"""Centralized head training (E0-E2) on cached features: AdamW, bf16 autocast, optional EMA, early stopping on VAL."""
import time
import numpy as np
import pandas as pd
import torch
from ..models.head import EMA
from .infer import EarlyStopper, val_score
from .losses import weighted_mean


def train_central(cfg, model, train_rows, val_rows, store, loss_fn, seed, log, epochs=None, select=True):
    c = cfg["central"]
    epochs = epochs or c["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    ema = EMA(model, cfg["model"].get("ema", 0.0))
    amp = c.get("amp", True) and store.device.type == "cuda"
    rng = np.random.default_rng(seed)
    stopper = EarlyStopper(c.get("patience", 0) if select else 0)
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(train_rows)
        tot = 0.0
        for i in range(0, len(perm), c["batch"]):
            x, t, y, m, w = store.batch(perm[i:i + c["batch"]])
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                z = model(x, t)
            loss = weighted_mean(loss_fn(z.float(), y, m), w)
            loss.backward(); opt.step(); ema.update(model)
            tot += loss.item() * len(x)
        target = ema.target(model)
        auc = val_score(target, store, val_rows, cfg["eval"]["min_pos_val"]) if select else np.nan
        hist.append({"round": ep, "train_loss": tot / len(perm), "val_mean5_auroc": auc, "sec": time.time() - t0})
        log(f"epoch {ep:3d} train loss {tot / len(perm):.4f} val mean5 AUROC {auc:.4f}")
        if select and stopper.update(auc, ep, target):
            log(f"early stop at epoch {ep} (best {stopper.best_step})")
            break
    if not select:
        stopper.state, stopper.best_step = {k: v.detach().clone() for k, v in ema.target(model).state_dict().items()}, epochs
    return stopper.state, pd.DataFrame(hist), {"best_round": stopper.best_step, "best_val_mean5": stopper.best,
                                                 "rounds_run": len(hist)}
