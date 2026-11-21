"""Stronger membership inference: likelihood-ratio attack (LiRA) with shadow / reference models, TPR at low FPR, and a DP audit.

  python scripts/mia_lira.py stress [--candidates 600 --shadows 16 --steps 400]   -> results/mia_lira/stress/
  python scripts/mia_lira.py real   [--refs 8 --n 2000]                           -> results/mia_lira/real/
  python scripts/mia_lira.py rescore                                               recompute both summaries from the saved statistics

stress: the setting of scripts/mia_demo.py (small image-only head on nih features, 400 steps), attacked properly. K shadow models
  per arm, each trained on a random half of a candidate set of pool patients, so every patient is IN for about half of them.
  Each model in turn is the target and the others are its shadows (leave-one-out). Arms: no DP (overfit), no DP with early
  stopping, patient-level DP-SGD at eps 8, 3, 1. Attacks: loss threshold, offline LiRA, online LiRA. For DP arms an empirical
  epsilon lower bound is reported next to the claimed epsilon.
real: attacks the seed-0 final models of the full run (results/v2.1-final). Candidates are n training patients and n held-out VAL
  patients. Reference models (central, frozen config) are trained without any candidate, giving offline LiRA. References are
  non-DP central models for every target, so for FL and DP targets the attack is mismatched but still valid.
VAL only; TEST is never read. Existing result folders are not overwritten (use --force).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpmmfl.cli import config, parser  # noqa: E402
from dpmmfl.data.store import ClientData, gather_units, load_split, pos_weight_from  # noqa: E402
from dpmmfl.eval.lira import eps_lower_bound, lira_offline, lira_online, patient_stats, roc_summary  # noqa: E402
from dpmmfl.models.head import build_head  # noqa: E402
from dpmmfl.train.central import train_central  # noqa: E402
from dpmmfl.train.infer import val_score  # noqa: E402
from dpmmfl.train.losses import make_loss, weighted_mean  # noqa: E402
from dpmmfl.train.privacy import calibrate_sigma, dp_gradient, epsilon_for  # noqa: E402
from dpmmfl.utils import get_device, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
B = 64


def rows_of(meta, pats):
    return np.flatnonzero(meta.patient_id.isin(pats).to_numpy())


def train_shadow(cfg, store, members, arm, steps, seed, dev, es_rows):
    """One shadow model. arm: ('nodp', None) overfit | ('nodp_es', None) early stopped on es_rows | ('dp', eps)."""
    kind, eps = arm
    torch.manual_seed(seed)
    model = build_head(cfg).to(dev)
    loss_fn = make_loss(cfg, pos_weight_from(store, members))
    client = ClientData(store, members)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model.train()
    if kind != "dp":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state = -np.inf, None
        for s in range(1, steps + 1):
            opt.zero_grad()
            x, t, y, m, w = store.batch(client.sample_batch(gen, B))
            weighted_mean(loss_fn(model(x, t), y, m), w).backward()
            opt.step()
            if kind == "nodp_es" and s % 25 == 0:
                sc = val_score(model, store, es_rows, 20)
                if sc > best:
                    best, best_state = sc, {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.train()
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, None
    dp = cfg["dp"]
    q = B / client.n_patients
    sigma = calibrate_sigma(eps, dp["delta"], q, steps, "prv")
    opt = torch.optim.SGD(model.parameters(), lr=dp["lr"])
    for _ in range(steps):
        opt.zero_grad()
        s = client.sample_units(gen, q, "patient", dp["max_images"])
        dp_gradient(model, loss_fn, gather_units(store, *s) if s is not None else None, dp["clip"], sigma, B, dp["vmap_chunk"])
        opt.step()
    return model, epsilon_for(sigma, q, steps, dp["delta"])[0]


def arm_of(name):
    """'no DP (overfit)' -> ('nodp', None), 'no DP, early stopped' -> ('nodp_es', None), 'DP eps=3' -> ('dp', 3.0)."""
    return ("dp", float(name[len("DP eps="):])) if name.startswith("DP eps=") else ("nodp_es" if "early" in name else "nodp", None)


def stats_file(out, arm):
    return out / f"stats_{arm[0]}{'' if arm[1] is None else f'{arm[1]:g}'}.npz"


def attacks(S, L, IN, delta, eps):
    """Leave-one-out over the K models. Returns summary rows and ROC curves per attack."""
    K = len(S)
    sc = {"loss threshold": [], "LiRA offline": [], "LiRA online": []}
    lab = []
    for k in range(K):
        o = np.arange(K) != k
        sc["loss threshold"].append(-L[k])
        sc["LiRA offline"].append(lira_offline(S[k], S[o], ~IN[o]))
        sc["LiRA online"].append(lira_online(S[k], S[o], IN[o]))
        lab.append(IN[k])
    lab = np.concatenate(lab)
    rows, curves = [], []
    for name, v in sc.items():
        v = np.concatenate(v)
        r, fpr, tpr = roc_summary(v, lab)
        r["attack"] = name
        if eps is not None:
            r["eps_claimed"], r["eps_lower_bound"] = eps, eps_lower_bound(v, lab, delta)
        rows.append(r)
        keep = np.unique(np.r_[np.searchsorted(fpr, np.geomspace(1e-4, 1, 300)).clip(0, len(fpr) - 1), len(fpr) - 1])
        curves.append(pd.DataFrame({"attack": name, "fpr": fpr[keep], "tpr": tpr[keep]}))
    return rows, pd.concat(curves)


def stress(a, cfg, dev, out):
    cfg["model"]["head"] = "image_only"
    store = load_split(cfg, "train", dev, ["nih"])
    meta = store.meta
    rng = np.random.default_rng(a.seed)
    pool = np.sort(meta[meta.split == "pool"].patient_id.unique())
    val = rng.permutation(np.sort(meta[meta.split == "val"].patient_id.unique()))
    cand = rng.choice(pool, a.candidates, replace=False)
    cand_rows = rows_of(meta, cand)
    eval_rows, es_rows = rows_of(meta, val[:2000]), rows_of(meta, val[2000:3000])
    IN = rng.random((a.shadows, a.candidates)) < 0.5
    pidx = {p: i for i, p in enumerate(meta.patient_id.unique())}                 # store.pat index of each candidate
    cand_idx = np.array([pidx[p] for p in cand])
    arms = [("no DP (overfit)", ("nodp", None)), ("no DP, early stopped", ("nodp_es", None))] + \
           [(f"DP eps={e:g}", ("dp", e)) for e in a.eps]
    summary, curves, models = [], [], []
    for name, arm in arms:
        S, L = np.zeros((a.shadows, a.candidates)), np.zeros((a.shadows, a.candidates))
        accs, epss, t0 = [], [], time.time()
        for k in range(a.shadows):
            mem = rows_of(meta, cand[IN[k]])
            model, e_sp = train_shadow(cfg, store, mem, arm, a.steps, 100 * k + 7, dev, es_rows)
            loss_fn = make_loss(cfg, pos_weight_from(store, mem))
            pats, phi, loss = patient_stats(model, store, cand_rows, loss_fn)
            order = np.searchsorted(pats, cand_idx)
            S[k], L[k] = phi[order], loss[order]
            accs.append(val_score(model, store, eval_rows, 20))
            epss.append(e_sp)
            models.append(dict(arm=name, shadow=k, val_auroc=accs[-1], eps=e_sp, n_members=int(IN[k].sum())))
        rows, cur = attacks(S, L, IN, cfg["dp"]["delta"], arm[1])
        for r in rows:
            r.update(arm=name, val_auroc=float(np.mean(accs)))
        summary += rows
        curves.append(cur.assign(arm=name))
        np.savez_compressed(stats_file(out, arm), S=S, L=L, IN=IN, cand=cand)
        best = max(rows, key=lambda r: r["auc"])
        print(f"{name:22s} VAL AUROC {np.mean(accs):.3f}  best attack {best['attack']}: AUC {best['auc']:.3f} "
              f"TPR@1% {best['tpr@0.01']:.3f} TPR@0.1% {best['tpr@0.001']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    cols = ["arm", "attack", "auc", "tpr@0.01", "tpr@0.001", "eps_claimed", "eps_lower_bound", "val_auroc", "n", "n_members"]
    pd.DataFrame(summary).reindex(columns=cols).to_csv(out / "summary.csv", index=False)
    pd.concat(curves).to_csv(out / "roc.csv", index=False)
    pd.DataFrame(models).to_csv(out / "shadows.csv", index=False)
    save_json({"mode": "stress", "candidates": a.candidates, "shadows": a.shadows, "steps": a.steps, "batch": B, "seed": a.seed,
               "delta": cfg["dp"]["delta"], "head": "image_only, nih features", "statistic": "-log(mean per-patient loss)",
               "evaluation": "leave-one-out: each shadow is the target once"}, out / "meta.json")


def latest_final(tag, seed=0):
    runs = [d for d in sorted((ROOT / "results/v2.1-final/runs").glob(f"{tag}__*__s{seed}")) if (d / "DONE").exists()]
    return runs[-1] if runs else None


def real(a, dev, out):
    targets = {t: latest_final(t) for t in a.targets}
    targets = {t: d for t, d in targets.items() if d is not None}
    ref_dir = latest_final("E2")
    cfg = yaml.safe_load((ref_dir / "config.yaml").read_text())
    cd = Path(cfg["cache_dir"])
    cfg["cache_dir"] = str(cd if cd.is_absolute() else ROOT / cd)
    store = load_split(cfg, "train", dev)
    meta = store.meta
    train_mask = ((meta.split == "pool") & ~meta.local_val.astype(bool)).to_numpy()
    rng = np.random.default_rng(a.seed)
    c_in = rng.choice(np.sort(meta[train_mask].patient_id.unique()), a.n, replace=False)
    c_out = rng.choice(np.sort(meta[meta.split == "val"].patient_id.unique()), a.n, replace=False)
    cand = np.r_[c_in, c_out]
    member = np.r_[np.ones(a.n, bool), np.zeros(a.n, bool)]
    cand_rows = rows_of(meta, cand)
    pidx = {p: i for i, p in enumerate(meta.patient_id.unique())}
    cand_idx = np.array([pidx[p] for p in cand])
    ref_train = np.flatnonzero(train_mask & ~meta.patient_id.isin(c_in).to_numpy())
    ref_es = np.flatnonzero(((meta.split == "val") & ~meta.patient_id.isin(c_out)).to_numpy())
    loss_fn = make_loss(cfg, pos_weight_from(store, ref_train))

    def stats(model):
        pats, phi, loss = patient_stats(model, store, cand_rows, loss_fn)
        o = np.searchsorted(pats, cand_idx)
        return phi[o], loss[o]

    R = []
    for r in range(a.refs):
        t0 = time.time()
        model = build_head(cfg).to(dev)
        state, _, info = train_central(cfg, model, ref_train, ref_es, store, loss_fn, 1000 + r, lambda *_: None)
        model.load_state_dict(state)
        R.append(stats(model)[0])
        print(f"reference {r}: {info['rounds_run']} epochs, VAL {info['best_val_mean5']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    T = dict(R=np.array(R), member=member, cand=cand, tags=[], run_ids=[], phi=[], loss=[], eps=[], delta=cfg["dp"]["delta"])
    for tag, d in targets.items():
        tcfg = yaml.safe_load((d / "config.yaml").read_text())
        model = build_head(tcfg).to(dev)
        model.load_state_dict(torch.load(d / "head.pt", map_location=dev))
        phi, loss = stats(model)
        priv = json.loads((d / "privacy.json").read_text()) if (d / "privacy.json").exists() else None
        for k, v in (("tags", tag), ("run_ids", d.name), ("phi", phi), ("loss", loss), ("eps", max(priv["eps_spent_prv"]) if priv else np.nan)):
            T[k].append(v)
    np.savez_compressed(out / "stats.npz", **{k: np.asarray(v) for k, v in T.items()})
    score_real(out)
    save_json({"mode": "real", "n_each": a.n, "refs": a.refs, "seed": a.seed, "reference": f"central, config of {ref_dir.name}, "
               "trained without any candidate patient", "statistic": "-log(mean per-patient loss)"}, out / "meta.json")


def score_real(out):
    """Loss threshold and offline LiRA for every target in out/stats.npz -> summary.csv, roc.csv."""
    z = np.load(out / "stats.npz")
    member, summary, curves = z["member"], [], []
    for i, tag in enumerate(z["tags"]):
        eps = None if np.isnan(z["eps"][i]) else float(z["eps"][i])
        for name, sc in (("loss threshold", -z["loss"][i]), ("LiRA offline", lira_offline(z["phi"][i], z["R"]))):
            r, fpr, tpr = roc_summary(sc, member)
            r.update(target=str(tag), attack=name, run_id=str(z["run_ids"][i]), eps_claimed=eps,
                     eps_lower_bound=eps_lower_bound(sc, member, float(z["delta"])) if eps else None)
            summary.append(r)
            keep = np.unique(np.r_[np.searchsorted(fpr, np.geomspace(1e-4, 1, 300)).clip(0, len(fpr) - 1), len(fpr) - 1])
            curves.append(pd.DataFrame({"target": str(tag), "attack": name, "fpr": fpr[keep], "tpr": tpr[keep]}))
        print(f"{tag:10s} loss AUC {summary[-2]['auc']:.3f}  LiRA AUC {summary[-1]['auc']:.3f}  TPR@1% {summary[-1]['tpr@0.01']:.3f}", flush=True)
    cols = ["target", "attack", "auc", "tpr@0.01", "tpr@0.001", "eps_claimed", "eps_lower_bound", "n", "n_members", "run_id"]
    pd.DataFrame(summary).reindex(columns=cols).to_csv(out / "summary.csv", index=False)
    pd.concat(curves).to_csv(out / "roc.csv", index=False)


def rescore(root):
    """Recompute the attack summaries from saved statistics (no training)."""
    st = root / "stress"
    if (st / "shadows.csv").exists():
        sh, delta = pd.read_csv(st / "shadows.csv"), json.loads((st / "meta.json").read_text())["delta"]
        summary, curves = [], []
        for name in dict.fromkeys(sh.arm):
            arm = arm_of(name)
            z = np.load(stats_file(st, arm))
            rows, cur = attacks(z["S"], z["L"], z["IN"], delta, arm[1])
            for r in rows:
                r.update(arm=name, val_auroc=float(sh[sh.arm == name].val_auroc.mean()))
            summary += rows
            curves.append(cur.assign(arm=name))
        cols = ["arm", "attack", "auc", "tpr@0.01", "tpr@0.001", "eps_claimed", "eps_lower_bound", "val_auroc", "n", "n_members"]
        pd.DataFrame(summary).reindex(columns=cols).to_csv(st / "summary.csv", index=False)
        pd.concat(curves).to_csv(st / "roc.csv", index=False)
        print(f"rescored {st}")
    if (root / "real" / "stats.npz").exists():
        score_real(root / "real")


def main():
    ap = parser(__doc__)
    ap.add_argument("mode", choices=["stress", "real", "rescore"])
    ap.add_argument("--candidates", type=int, default=600, help="stress: candidate patients (each shadow trains on ~half)")
    ap.add_argument("--shadows", type=int, default=16, help="stress: shadow models per arm")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eps", type=float, nargs="+", default=[8, 3, 1])
    ap.add_argument("--refs", type=int, default=8, help="real: reference models")
    ap.add_argument("--n", type=int, default=2000, help="real: member and non-member candidates each")
    ap.add_argument("--targets", nargs="+", default=["E2", "E3", "E4_eps8", "E4_eps3", "E4_eps1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mia_lira")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.mode == "rescore":
        rescore(ROOT / a.out)
        return
    out = ROOT / a.out / a.mode
    if (out / "summary.csv").exists() and not a.force:
        sys.exit(f"{out}/summary.csv exists (use --force)")
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(a)
    dev = get_device(cfg)
    stress(a, cfg, dev, out) if a.mode == "stress" else real(a, dev, out)


if __name__ == "__main__":
    main()
