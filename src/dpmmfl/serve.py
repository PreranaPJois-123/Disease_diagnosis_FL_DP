"""Run discovery, model loading and single-image prediction for the dashboard (dashboard/app.py).

Supports v2.1 runs (results/<scale>/runs/<run_id>/) and the v2 full-scale runs (results/v2-large/<tag>/<seed>/).
Features for an uploaded image are computed with exactly the pipeline used for the cached features.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from .constants import COMP5, D_TAB, LABELS
from .eval.gradcam import gradcam
from .features.extract import build_encoder, encode, preprocess
from .models.head import FusionHead
from .utils import get_device

from .runs import ROOT, _scale_of, discover_runs  # noqa: F401  (moved to runs.py: no torch needed)


def _head_cfg(cfg):
    m = cfg["model"]
    encs = (cfg.get("features") or {}).get("encoders") or ["nih"]
    return encs, dict(head=m.get("head") or m.get("fusion", "film"), d=m["d"], p_drop=m["p_drop"], p_modality_drop=m["p_modality_drop"])


class Predictor:
    def __init__(self, run_dir, device=None):
        self.dir = Path(run_dir)
        cfg = yaml.safe_load((self.dir / "config.yaml").read_text())
        self.dev = device or get_device(cfg)
        self.encoders, kw = _head_cfg(cfg)
        self.head_kind = kw["head"]
        self.head = FusionHead(len(self.encoders), **kw).to(self.dev)
        self.head.load_state_dict(torch.load(self.dir / "head.pt", map_location=self.dev))
        self.head.eval()
        self.nets = [build_encoder(e, self.dev) for e in self.encoders]
        cal = self.dir / "calibration.json"
        self.temp = np.array([json.loads(cal.read_text())["temperature"][l] for l in LABELS]) if cal.exists() else np.ones(len(LABELS))
        self.calibrated = cal.exists()
        sd = Path(cfg.get("splits_dir") or ROOT / "splits")
        stats = sd / "tab_stats.json"
        self.age_stats = json.loads((stats if stats.exists() else ROOT / "splits" / "tab_stats.json").read_text())
        self.val_auroc = self._val_auroc()

    def _val_auroc(self):
        mv = self.dir / "metrics_val.json"
        if mv.exists():
            a = json.loads(mv.read_text()).get("auroc")
            if a:
                return {l: a.get(l) for l in LABELS}
        pl = self.dir / "per_label.csv"
        if pl.exists():
            return pd.read_csv(pl).set_index("label").val_auroc.to_dict()
        return {}

    def tab(self, age=None, sex=None, projection=None):
        """9-d metadata vector; all-None -> zeros (the 'unknown' input the model saw under modality dropout)."""
        if age is None and sex is None and projection is None:
            return torch.zeros(1, D_TAB, device=self.dev)
        miss = age is None
        z = 0.0 if miss else (np.clip(age, 18, 90) - self.age_stats["mean"]) / self.age_stats["std"]
        v = [z, float(miss), sex == "Male", sex == "Female", sex not in ("Male", "Female"),
             projection == "AP", projection == "PA", projection not in ("AP", "PA"), 0.0]
        return torch.tensor([v], dtype=torch.float32, device=self.dev)

    @torch.no_grad()
    def predict(self, gray, **meta):
        """gray: 2-D uint8 array. Returns DataFrame (label, prob, val_auroc, comp5) and the preprocessed tensor."""
        x = preprocess(gray)[None].to(self.dev)
        f = torch.stack([encode(n, x, amp=False) for n in self.nets], dim=1)
        z = self.head(f, self.tab(**meta))[0].float().cpu().numpy() / self.temp
        df = pd.DataFrame(dict(label=LABELS, prob=1 / (1 + np.exp(-z)), val_auroc=[self.val_auroc.get(l) for l in LABELS],
                               comp5=[l in COMP5 for l in LABELS]))
        return df, x

    def cam(self, x, label, **meta):
        return gradcam(self.nets, self.head, x, self.tab(**meta), LABELS.index(label))


def live_mia(run_dir, n=1000, seed=0, device=None):
    """Loss-threshold membership attack on a finished run: training patients (pool, not local-val) vs held-out VAL patients.
    Returns dict(auc, tpr1 (TPR at 1% FPR), member_loss, nonmember_loss (per-patient arrays), n). Uses cached features only; TEST is never touched."""
    from sklearn.metrics import roc_auc_score
    from .data.store import load_split, pos_weight_from
    from .eval.mia import patient_losses, tpr_at_fpr
    from .train.losses import make_loss
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cd = Path(cfg["cache_dir"])
    cfg["cache_dir"] = str(cd if cd.is_absolute() else ROOT / cd)
    dev = device or get_device(cfg)
    encs, kw = _head_cfg(cfg)
    cfg["features"] = {**(cfg.get("features") or {}), "encoders": encs}
    store = load_split(cfg, "train", dev, encs)
    meta = store.meta
    members = np.flatnonzero(((meta.split == "pool") & ~meta.local_val.astype(bool)).to_numpy())
    heldout = np.flatnonzero((meta.split == "val").to_numpy())
    head = FusionHead(len(encs), **kw).to(dev)
    head.load_state_dict(torch.load(run_dir / "head.pt", map_location=dev))
    loss_fn = make_loss(cfg, pos_weight_from(store, members))
    lm, ln = patient_losses(head, store, members, loss_fn), patient_losses(head, store, heldout, loss_fn)
    rng = np.random.default_rng(seed)
    k = min(n, len(lm), len(ln))
    lm, ln = rng.choice(lm, k, replace=False), rng.choice(ln, k, replace=False)
    lab, sc = np.r_[np.ones(k), np.zeros(k)], -np.r_[lm, ln]
    return dict(auc=float(roc_auc_score(lab, sc)), tpr1=tpr_at_fpr(lab, sc), member_loss=lm, nonmember_loss=ln, n=k)
