"""Cached-feature store (all encoders) and per-client patient indices. No DataLoader: every batch is one tensor gather.

Features live on the GPU as float16 [N, E, 1024] when they fit, else in pinned CPU memory (gather on CPU, async copy).
ClientData pre-pads each client's patient -> image-row table once, so a DP batch is Poisson mask + top-k + gather on device.
"""
import os
import numpy as np
import pandas as pd
import torch
from ..constants import LABELS, TAB_COLS, ENCODERS


def label_cols(prefix):
    return [f"{prefix}_{l}" for l in LABELS]


def feature_path(cache_dir, split, enc):
    """New layout feat_<split>_<enc>.npy; the v2 nih cache keeps its legacy name feat_<split>.npy."""
    p = os.path.join(cache_dir, f"feat_{split}_{enc}.npy")
    legacy = os.path.join(cache_dir, f"feat_{split}.npy")
    return p if os.path.exists(p) or enc != "nih" or not os.path.exists(legacy) else legacy


def load_features(cache_dir, split, encoders, meta):
    out = []
    for enc in encoders:
        assert enc in ENCODERS, f"unknown encoder {enc}"
        p = feature_path(cache_dir, split, enc)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing features {p} (run scripts/02_extract.py)")
        f = np.load(p, mmap_mode="r")
        assert len(f) == len(meta), f"{p}: {len(f)} rows vs meta {len(meta)}"
        rid = p.replace(".npy", ".rowid.npy")
        if os.path.exists(rid):
            assert np.array_equal(np.load(rid), meta["row_id"].to_numpy()), f"{p}: row_id order differs from meta"
        out.append(f)
    return out


class Store:
    """All tensors for one split. Row positions index into these tensors."""

    def __init__(self, feats, meta, device, on_gpu="auto"):
        self.meta = meta.reset_index(drop=True)
        self.device = device
        n, E = len(self.meta), len(feats)
        x = np.empty((n, E, feats[0].shape[1]), np.float16)
        for e, f in enumerate(feats):
            x[:, e] = f
        x = torch.from_numpy(x)
        if on_gpu == "auto":
            on_gpu = device.type == "cuda" and x.numel() * 2 < 0.45 * torch.cuda.mem_get_info(device)[0]
        self.on_gpu = bool(on_gpu) and device.type == "cuda"
        self.x = x.to(device) if self.on_gpu else (x.pin_memory() if device.type == "cuda" else x)
        self.tab = torch.tensor(self.meta[TAB_COLS].to_numpy(np.float32), device=device)
        self.y = torch.tensor(self.meta[label_cols("T")].to_numpy(np.float32), device=device)
        self.m = torch.tensor(self.meta[label_cols("M")].to_numpy(np.float32), device=device)
        self.pat = pd.factorize(self.meta["patient_id"])[0]
        self.w = torch.ones(n, device=device)                       # per-image loss weights (fairness F1)

    def __len__(self):
        return len(self.meta)

    def _idx(self, rows):
        return rows if torch.is_tensor(rows) else torch.as_tensor(np.asarray(rows), dtype=torch.long)

    def feats(self, rows):
        r = self._idx(rows)
        if self.on_gpu:
            return self.x[r.to(self.device)].float()
        return self.x[r.cpu()].to(self.device, non_blocking=True).float()

    def batch(self, rows):
        r = self._idx(rows).to(self.device)
        return self.feats(r), self.tab[r], self.y[r], self.m[r], self.w[r]

    def numpy_targets(self, rows):
        return self.y[rows].cpu().numpy(), self.m[rows].cpu().numpy()


class ClientData:
    """One client's training rows, grouped by patient once (padded index table on device)."""

    def __init__(self, store, rows):
        self.store = store
        self.rows_np = np.asarray(rows)
        self.rows = torch.as_tensor(self.rows_np, device=store.device)
        pat = store.pat[self.rows_np]
        order = np.argsort(pat, kind="stable")
        bounds = np.flatnonzero(np.diff(pat[order])) + 1
        groups = np.split(self.rows_np[order], bounds) if len(self.rows_np) else []
        self.counts = torch.as_tensor([len(g) for g in groups], device=store.device)
        L = max([len(g) for g in groups] + [1])
        tbl = np.zeros((len(groups), L), np.int64)
        for i, g in enumerate(groups):
            tbl[i, :len(g)] = g
        self.table = torch.as_tensor(tbl, device=store.device)

    @property
    def n_patients(self): return len(self.counts)

    @property
    def n_images(self): return len(self.rows_np)

    @property
    def max_images(self): return int(self.counts.max())

    def sample_batch(self, gen, B):
        return self.rows[torch.randint(0, len(self.rows), (min(B, len(self.rows)),), device=self.rows.device, generator=gen)]

    def sample_units(self, gen, q, unit, M):
        """Poisson-sample privacy units. Returns (row index [b, M'], image mask [b, M']) or None if empty.
        patient: each sampled patient contributes <= M random images; image: one image per unit."""
        dev = self.rows.device
        if unit == "image":
            sel = self.rows[torch.rand(len(self.rows), device=dev, generator=gen) < q]
            return (sel[:, None], torch.ones(len(sel), 1, device=dev)) if len(sel) else None
        sel = torch.nonzero(torch.rand(self.n_patients, device=dev, generator=gen) < q).flatten()
        if len(sel) == 0:
            return None
        L = self.table.shape[1]
        keys = torch.rand(len(sel), L, device=dev, generator=gen)
        valid = torch.arange(L, device=dev)[None] < self.counts[sel][:, None]
        keys = keys.masked_fill(~valid, 2.0)
        k = min(M, L)
        pos = keys.topk(k, dim=1, largest=False).indices                    # random <=M images per patient
        im = torch.gather(valid, 1, pos).float()
        idx = torch.gather(self.table[sel], 1, pos)
        return idx, im


def gather_units(store, idx, im):
    """[b, M] row indices -> F [b,M,E,1024], T [b,M,9], Y, Mk [b,M,14], IM [b,M], W [b,M]"""
    b, M = idx.shape
    flat = idx.flatten()
    F, T, Y, Mk, W = store.batch(flat)
    sh = lambda t: t.view(b, M, *t.shape[1:])
    return sh(F), sh(T), sh(Y), sh(Mk), im, sh(W)


def pos_weight_from(store, rows):
    r = torch.as_tensor(np.asarray(rows), device=store.device)
    y, m = store.y[r], store.m[r]
    pos, neg = (m * y).sum(0), (m * (1 - y)).sum(0)
    return (neg / pos.clamp(min=1)).clamp(1, 10)


def load_split(cfg, split, device, encoders=None):
    enc = encoders or cfg["features"]["encoders"]
    meta = pd.read_parquet(os.path.join(cfg["cache_dir"], f"meta_{split}.parquet"))
    return Store(load_features(cfg["cache_dir"], split, enc, meta), meta, device, cfg["features"].get("on_gpu", "auto"))
