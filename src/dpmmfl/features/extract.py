"""Frozen TorchXRayVision encoders -> cached 1024-d features (docs/03 s3.7).

One pass: each frontal JPEG is decoded and preprocessed once, then fed to every requested encoder.
Resumable (per-chunk files per encoder), float16, row order identical to meta (row_id saved alongside and checked).
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ..constants import ENCODERS, FORBIDDEN_WEIGHTS


def preprocess(img):
    """2-D (or RGB) uint8 array -> [1,224,224] float tensor, exactly as used for the cached features."""
    import torchxrayvision as xrv
    img = xrv.datasets.normalize(img, 255)
    img = img[None] if img.ndim == 2 else img.mean(2)[None]
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    return torch.from_numpy(img).float()


class ImgDS(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import skimage.io
        return preprocess(skimage.io.imread(self.paths[i]))


def check_weights(enc):
    w = ENCODERS[enc]
    assert not any(f"-{b}" in w or w.endswith(b) for b in FORBIDDEN_WEIGHTS), f"forbidden weights {w}"
    return w


def build_encoder(enc, device):
    import torchxrayvision as xrv
    return xrv.models.DenseNet(weights=check_weights(enc)).eval().to(device)


@torch.no_grad()
def encode(net, x, amp):
    with torch.autocast("cuda", enabled=amp and x.is_cuda):
        f = torch.nn.functional.relu(net.features(x))
        return torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1).float()


def out_path(cache_dir, split, enc):
    return os.path.join(cache_dir, f"feat_{split}_{enc}.npy")


def extract_split(meta, split, encoders, cfg, device, log=print):
    """Writes cache/feat_<split>_<enc>.npy (+ .rowid.npy) for every encoder not yet finished."""
    ec, cd = cfg["extract"], cfg["cache_dir"]
    todo = [e for e in encoders if not os.path.exists(out_path(cd, split, e))]
    if not todo:
        log(f"[{split}] all features exist for {encoders}")
        return
    nets = {e: build_encoder(e, device) for e in todo}
    for e, net in nets.items():
        log(f"[{split}] {e}: {ENCODERS[e]} weight checksum {float(sum(p.double().sum() for p in net.parameters())):.4f}")
    paths = meta["abs_path"].to_numpy()
    n, cs = len(paths), ec["chunk"]
    nch = (n + cs - 1) // cs
    cdirs = {e: out_path(cd, split, e).replace(".npy", "_chunks") for e in todo}
    for d in cdirs.values():
        os.makedirs(d, exist_ok=True)
    pending = [ci for ci in range(nch) if any(not os.path.exists(os.path.join(cdirs[e], f"{ci:05d}.npy")) for e in todo)]
    bs = ec["batch_size"]
    batches = [list(range(s, min(s + bs, (ci + 1) * cs, n))) for ci in pending for s in range(ci * cs, min((ci + 1) * cs, n), bs)]
    # one loader for the whole pass (re-spawning Windows workers per chunk costs ~10 s each)
    dl = DataLoader(ImgDS(paths), batch_sampler=batches, num_workers=ec["num_workers"], pin_memory=device.type == "cuda")
    it = iter(dl)
    for ci in pending:
        need = [e for e in todo if not os.path.exists(os.path.join(cdirs[e], f"{ci:05d}.npy"))]
        outs = {e: [] for e in todo}
        for _ in range((min((ci + 1) * cs, n) - ci * cs + bs - 1) // bs):
            b = next(it).to(device, non_blocking=True)
            for e in need:
                outs[e].append(encode(nets[e], b, ec["amp"]).cpu().numpy())
        for e in need:
            f = os.path.join(cdirs[e], f"{ci:05d}.npy")
            np.save(f + ".tmp.npy", np.concatenate(outs[e]).astype(np.float16))
            os.replace(f + ".tmp.npy", f)                      # atomic: a crash never leaves a truncated chunk
        log(f"[{split}] chunk {ci + 1}/{nch} ({', '.join(need)})")
    for e in todo:
        feats = np.concatenate([np.load(os.path.join(cdirs[e], f"{i:05d}.npy")) for i in range(nch)])
        assert len(feats) == n, f"{e}: {len(feats)} != {n}"
        np.save(out_path(cd, split, e).replace(".npy", ".rowid.npy"), meta["row_id"].to_numpy())
        np.save(out_path(cd, split, e), feats)


def verify(meta, feats, enc, cfg, device, n_check=16, seed=0):
    """Integrity: length, finite, non-constant, row order, spot re-extraction. Returns max re-extraction diff."""
    assert len(feats) == len(meta), "feat/meta length mismatch"
    f32 = np.asarray(feats, dtype=np.float32)
    assert np.isfinite(f32).all(), "NaN/Inf in features"
    assert f32.std(0).mean() > 0.01, "features nearly constant"
    idx = np.random.default_rng(seed).choice(len(meta), size=min(n_check, len(meta)), replace=False)
    ds = ImgDS(meta["abs_path"].to_numpy()[idx])
    x = torch.stack([ds[i] for i in range(len(ds))]).to(device)
    diff = float(np.abs(encode(build_encoder(enc, device), x, cfg["extract"]["amp"]).cpu().numpy() - f32[idx]).max())
    assert diff < 2e-2, f"{enc}: re-extraction mismatch {diff}"
    return diff
