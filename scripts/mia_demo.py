"""Membership-inference stress test: does patient-level DP stop an overfit model from leaking who was in the training set?

Trains a small head on N member patients (nih features) with (a) plain AdamW for many steps (worst case: no early stopping, no DP) and
(b) patient-level DP-SGD at several epsilons, same steps and data. The attack is the loss-threshold audit of eval/mia.py:
members (training patients) vs the same number of held-out VAL patients. AUC 0.5 = no leakage. Uses VAL only; TEST is never read.

  python scripts/mia_demo.py [--members 300 --seeds 5 --steps 400]      -> results/mia_demo/
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpmmfl.cli import parser, config
from dpmmfl.data.store import ClientData, gather_units, load_split, pos_weight_from
from dpmmfl.eval.mia import patient_losses
from dpmmfl.models.head import build_head
from dpmmfl.train.infer import val_score
from dpmmfl.train.losses import make_loss, weighted_mean
from dpmmfl.train.privacy import calibrate_sigma, dp_gradient, epsilon_for
from dpmmfl.utils import get_device, save_json


def train_arm(cfg, store, members, eps, steps, seed, dev):
    torch.manual_seed(seed)
    model = build_head(cfg).to(dev)
    loss_fn = make_loss(cfg, pos_weight_from(store, members))
    client = ClientData(store, members)
    gen = torch.Generator(device=dev).manual_seed(seed)
    dp = cfg["dp"]
    B = 64
    model.train()
    if eps is None:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            x, t, y, m, w = store.batch(client.sample_batch(gen, B))
            weighted_mean(loss_fn(model(x, t), y, m), w).backward()
            opt.step()
        return model, None
    q = B / client.n_patients
    sigma = calibrate_sigma(eps, dp["delta"], q, steps, "prv")
    opt = torch.optim.SGD(model.parameters(), lr=dp["lr"])
    for _ in range(steps):
        opt.zero_grad()
        s = client.sample_units(gen, q, "patient", dp["max_images"])
        batch = gather_units(store, *s) if s is not None else None
        dp_gradient(model, loss_fn, batch, dp["clip"], sigma, B, dp["vmap_chunk"])
        opt.step()
    return model, epsilon_for(sigma, q, steps, dp["delta"])[0]


def main():
    ap = parser(__doc__)
    ap.add_argument("--members", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eps", type=float, nargs="+", default=[8, 3, 1])
    ap.add_argument("--out", default="results/mia_demo")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "summary.csv").exists() and not a.force:
        sys.exit(f"{out}/summary.csv exists (use --force)")
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(a)
    cfg["model"]["head"] = "image_only"
    dev = get_device(cfg)
    store = load_split(cfg, "train", dev, ["nih"])
    meta = store.meta
    pool = np.sort(meta[meta.split == "pool"].patient_id.unique())
    val = np.sort(meta[meta.split == "val"].patient_id.unique())
    rows_of = lambda pats: np.flatnonzero(meta.patient_id.isin(pats).to_numpy())
    res, losses = [], []
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        vperm = rng.permutation(val)
        mem_p = rng.choice(pool, a.members, replace=False)
        non_p, eval_p = vperm[:a.members], vperm[a.members:a.members + 2000]
        mem, non, ev = rows_of(mem_p), rows_of(non_p), rows_of(eval_p)
        for eps in [None] + list(a.eps):
            t0 = time.time()
            model, eps_spent = train_arm(cfg, store, mem, eps, a.steps, seed, dev)
            loss_fn = make_loss(cfg, pos_weight_from(store, mem))
            lm, ln = patient_losses(model, store, mem, loss_fn), patient_losses(model, store, non, loss_fn)
            auc = float(roc_auc_score(np.r_[np.ones(len(lm)), np.zeros(len(ln))], -np.r_[lm, ln]))
            arm = "no DP" if eps is None else f"DP eps={eps:g}"
            res.append(dict(arm=arm, seed=seed, auc=auc, advantage=abs(2 * auc - 1), member_loss=lm.mean(), nonmember_loss=ln.mean(),
                            val_auroc=val_score(model, store, ev, 20), eps=eps_spent, members=a.members))
            if seed == 0:
                losses.append(pd.DataFrame({"arm": arm, "member": np.r_[np.ones(len(lm)), np.zeros(len(ln))].astype(bool), "loss": np.r_[lm, ln]}))
            print(f"seed {seed} {arm:12s} MIA AUC {auc:.3f}  member loss {lm.mean():.3f} vs held-out {ln.mean():.3f}  "
                  f"VAL AUROC {res[-1]['val_auroc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(res).to_csv(out / "summary.csv", index=False)
    pd.concat(losses).to_csv(out / "losses_seed0.csv", index=False)
    save_json({"members": a.members, "steps": a.steps, "seeds": a.seeds, "delta": cfg["dp"]["delta"], "attack": "loss threshold (patient mean loss)",
               "nonmembers": "same number of held-out VAL patients", "head": "image_only, nih features"}, out / "meta.json")
    print(pd.DataFrame(res).groupby("arm", sort=False)[["auc", "member_loss", "nonmember_loss", "val_auroc"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
