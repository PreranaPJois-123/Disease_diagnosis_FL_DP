"""Extract + verify cached features for the encoders in cfg.extract.encoders (one decode per image). Resumable.

  python scripts/02_extract.py                 # full: test then train, mimic_ch + pc (nih already cached)
  python scripts/02_extract.py --smoke         # smoke rows only (train split), into cache/smoke
  python scripts/02_extract.py --limit 500     # probe the first N rows into <cache>/probe/ (deleted first)
"""
import os, shutil, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpmmfl.cli import parser, config
from dpmmfl.data.store import load_features
from dpmmfl.features.extract import extract_split, verify
from dpmmfl.utils import get_device


def main():
    ap = parser(__doc__)
    ap.add_argument("--limit", type=int, default=0, help="probe: first N rows only, written to <cache>/probe/")
    ap.add_argument("--encoders", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = config(a)
    if a.workers is not None:
        cfg["extract"]["num_workers"] = a.workers
    enc = a.encoders or cfg["extract"]["encoders"]
    dev = get_device(cfg)
    src = cfg["cache_dir"]
    splits = ["train"] if cfg["smoke"] else ["test", "train"]
    if a.limit:
        cfg["cache_dir"] = os.path.join(src, "probe")
        shutil.rmtree(cfg["cache_dir"], ignore_errors=True)
        os.makedirs(cfg["cache_dir"])
    for split in splits:
        meta = pd.read_parquet(os.path.join(src, f"meta_{split}.parquet"))
        if a.limit:
            meta = meta.iloc[:a.limit].reset_index(drop=True)
        t = time.time()
        extract_split(meta, split, enc, cfg, dev)
        dt = time.time() - t
        for e, f in zip(enc, load_features(cfg["cache_dir"], split, enc, meta)):
            d = verify(meta, f, e, cfg, dev)
            print(f"[{split}] {e}: {f.shape} float16, row_id order OK, finite, re-extraction max diff {d:.2e}")
        print(f"[{split}] {len(meta)} images in {dt:.0f}s ({len(meta) / max(dt, 1e-6):.1f} img/s for {len(enc)} encoders)")
    print("EXTRACTION OK")


if __name__ == "__main__":
    main()
