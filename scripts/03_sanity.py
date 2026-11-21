"""Sanity gate (docs/09). Exits non-zero on any failure. Uses TRAIN/VAL only - never TEST.

D1-D7 data checks; L1 linear-probe floor on VAL; L2 shuffled-label control (one epoch, mean of 3 shuffles, compared with
real labels under the same protocol); L3 overfit a tiny batch; L4 1-epoch train/VAL AUROC; L5 no VAL label below 0.45.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpmmfl.cli import parser, config
from dpmmfl.constants import LABELS
from dpmmfl.data.parse import make_targets
from dpmmfl.data.store import load_split, pos_weight_from
from dpmmfl.eval.metrics import safe_auroc, per_label
from dpmmfl.models.head import build_head
from dpmmfl.train.central import train_central
from dpmmfl.train.infer import predict_logits, val_score
from dpmmfl.train.losses import make_loss, weighted_mean
from dpmmfl.utils import seed_everything, get_device

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main():
    a = parser(__doc__).parse_args()
    cfg = config(a)
    sc = cfg["sanity"]
    seed_everything(0)
    dev = get_device(cfg)
    quiet = lambda s: None
    train = load_split(cfg, "train", dev)
    mt = train.meta
    print(f"features {cfg['features']['encoders']}: {tuple(train.x.shape)} on_gpu={train.on_gpu}")

    # ---- data checks
    raw = pd.read_csv(Path(cfg["data_root"]) / "CheXpert-v1.0-small" / "train.csv")
    ok = True
    for i in np.random.default_rng(0).choice(len(mt), 10, replace=False):
        r, csv = mt.iloc[i], raw.iloc[int(mt.iloc[i]["row_id"])]
        ok &= csv["Path"] == r["Path"] and csv["Path"].split("/")[2] == r["patient_id"]
        ok &= np.allclose(np.nan_to_num(csv[LABELS].to_numpy(float), nan=-9), np.nan_to_num(r[[f"R_{l}" for l in LABELS]].to_numpy(float), nan=-9))
    check("D1 path<->label alignment (10 rows by row_id)", bool(ok))
    check("D2 patient ids parsed from Path", bool(mt["patient_id"].str.match(r"^patient\d+$").all()))
    K = cfg["partition"]["K"]
    sets = {"val": set(mt[mt.split == "val"].patient_id), **{f"c{k}": set(mt[(mt.split == "pool") & (mt.client_id == k)].patient_id) for k in range(K)}}
    ov = [(x, y) for x in sets for y in sets if x < y and sets[x] & sets[y]]
    check("D3 patient overlap = 0 (VAL / clients)", not ov, str(ov))
    y, m = make_targets(raw.iloc[mt.row_id], cfg["uncertainty"], cfg["blank_as_negative"])
    yy, mm = train.y.cpu().numpy(), train.m.cpu().numpy()
    check("D4 targets/masks in {0,1} and re-derivable from the CSV", set(np.unique(yy)) <= {0., 1.} and np.array_equal(y, yy) and np.array_equal(m, mm))
    check("D5 label order", list(raw.columns[5:19]) == LABELS)
    prev = mt[[f"T_{l}" for l in LABELS]].mean()
    print("  D6 prevalence:", {l[:6]: round(float(v), 2) for l, v in zip(LABELS, prev)})
    check("D6 no label at 0%/100%", bool(((prev > 0) & (prev < 1)).all()))
    f = train.x[:20000].float()
    check("D7 features finite, non-constant per encoder", bool(torch.isfinite(f).all()) and bool((f.std(0).mean(-1) > 0.01).all()))

    pool = (mt.split == "pool") & ~mt.local_val
    tr = np.flatnonzero(pool.to_numpy())
    va = np.flatnonzero((mt.split == "val").to_numpy())
    pw = pos_weight_from(train, tr)
    loss_fn = make_loss({"model": {**cfg["model"], "loss": "bce_pw"}}, pw)
    mp = cfg["eval"]["min_pos_val"]

    # ---- L1 linear-probe floor (VAL)
    lin = {**cfg, "model": {**cfg["model"], "head": "linear", "ema": 0.0}}
    model = build_head(lin).to(dev)
    best, _, _ = train_central(lin, model, tr, va, train, loss_fn, 0, quiet, epochs=min(5, cfg["central"]["epochs"]))
    model.load_state_dict(best)
    e0 = val_score(model, train, va, mp)
    check("L1 linear probe VAL mean-5 AUROC", e0 >= sc["e0_min_auc"], f"= {e0:.4f} (min {sc['e0_min_auc']})")
    if e0 < sc["e0_min_auc"]:
        print("STOP: linear probe below threshold. Fix the pipeline (docs/09 s9.4)."); sys.exit(1)

    # ---- L2 shuffled-label control
    img = {**cfg, "model": {**cfg["model"], "head": "image_only", "ema": 0.0}}
    rng = np.random.default_rng(0)
    saved = (train.y.clone(), train.m.clone())
    tr_t = torch.as_tensor(tr, device=dev)
    sh = []
    for s in range(3):
        perm = torch.as_tensor(rng.permutation(len(tr)), device=dev)
        train.y[tr_t], train.m[tr_t] = saved[0][tr_t][perm], saved[1][tr_t][perm]
        model = build_head(img).to(dev)
        train_central(img, model, tr, va, train, loss_fn, s, quiet, epochs=1, select=False)
        train.y, train.m = saved[0].clone(), saved[1].clone()          # score VAL against the true labels
        sh.append(val_score(model, train, va, mp))
    train.y, train.m = saved
    auc_sh = float(np.mean(sh))
    seed_everything(0)
    model = build_head(img).to(dev)
    train_central(img, model, tr, va, train, loss_fn, 0, quiet, epochs=1, select=False)
    auc_real = val_score(model, train, va, mp)
    check("L2 shuffled-label control ~0.5 and far below real labels",
          abs(auc_sh - 0.5) <= sc["shuffle_tol"] and auc_real - auc_sh >= sc["l2_margin"],
          f"shuffled {auc_sh:.4f} ({', '.join(f'{x:.3f}' for x in sh)}; tol {sc['shuffle_tol']}) real {auc_real:.4f} (margin >= {sc['l2_margin']})")

    # ---- L3 overfit a tiny batch
    seed_everything(0)
    c3 = {**cfg, "model": {**cfg["model"], "head": "film", "p_drop": 0.0, "p_modality_drop": 0.0}}
    model = build_head(c3).to(dev)
    x, t, yb, mb, _ = train.batch(tr[:64])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(300):
        opt.zero_grad(); loss = weighted_mean(loss_fn(model(x, t), yb, mb)); loss.backward(); opt.step()
    check("L3 overfit 64 samples (loss < 0.05)", loss.item() < 0.05, f"final loss {loss.item():.4f}")

    # ---- L4 / L5 after one epoch
    seed_everything(0)
    c4 = {**cfg, "central": {**cfg["central"], "batch": 64}, "model": {**cfg["model"], "ema": 0.0}}
    model = build_head(c4).to(dev)
    best, _, _ = train_central(c4, model, tr, va, train, loss_fn, 0, quiet, epochs=1)
    model.load_state_dict(best)
    sub = np.random.default_rng(0).choice(tr, min(5000, len(tr)), replace=False)
    a_tr, a_va = val_score(model, train, sub, mp), val_score(model, train, va, mp)
    check("L4 1-epoch train/VAL AUROC above floor, small gap", a_tr > sc["l4_min_auc"] and a_va > sc["l4_min_auc"] and abs(a_tr - a_va) < sc["l4_gap"],
          f"train {a_tr:.3f} val {a_va:.3f}")
    yv, mv = train.numpy_targets(va)
    pl = per_label(safe_auroc, yv, predict_logits(model, train, va), mv, mp)
    bad = {k: round(v, 3) for k, v in pl.items() if not np.isnan(v) and v < 0.45}
    print("  L5 VAL per-label AUROC:", {k[:8]: round(v, 3) for k, v in pl.items() if not np.isnan(v)})
    check("L5 no VAL per-label AUROC < 0.45", not bad, str(bad))

    print("\nSANITY", "FAILED: " + ", ".join(FAILS) if FAILS else "PASSED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
