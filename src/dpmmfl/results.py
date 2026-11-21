"""Results writer.

Layout (one root per campaign, e.g. results/v2.1-final or results/smoke):
  <root>/runs/<run_id>/           run_id = <tag>__<config-hash>__s<seed>, unique per (config, seed)
      manifest.json               status, timings, config hash, versions, host
      config.yaml                 resolved config
      metrics_val.json            VAL metrics (always)
      metrics_test.json           TEST metrics (final runs only)
      preds_val.parquet / preds_test.parquet   per-sample logits + calibrated probabilities
      history.csv, fairness.json, privacy.json, comms.json, mia.json, head.pt
      DONE                        written last; a run without it is incomplete
  <root>/test_access.jsonl        one line per TEST evaluation (run_id, time) - TEST is evaluated once per final run
  <root>/logs/                    driver logs
"""
import json, platform, shutil, socket, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from .constants import LABELS
from .utils import config_hash, save_json


class RunExists(Exception):
    pass


class TestAccessDenied(RuntimeError):
    pass


def versions():
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform()}
    for mod in ("torchxrayvision", "opacus", "numpy", "sklearn"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


PERF_KEYS = {("dp", "vmap_chunk"), ("features", "on_gpu"), ("extract", "num_workers"), ("extract", "batch_size")}


def hashable(cfg):
    """Config minus paths and performance-only knobs, so retries with e.g. a smaller vmap chunk keep the same run_id."""
    out = {k: v for k, v in cfg.items() if k not in ("results_dir", "cache_dir", "splits_dir", "data_root")}
    for a, b in PERF_KEYS:
        if isinstance(out.get(a), dict):
            out[a] = {k: v for k, v in out[a].items() if k != b}
    return out


class RunWriter:
    def __init__(self, cfg, tag, seed, resume=False):
        self.cfg, self.tag, self.seed = cfg, tag, seed
        self.root = Path(cfg["results_dir"])
        self.chash = config_hash(hashable(cfg))
        self.run_id = f"{tag}__{self.chash}__s{seed}"
        self.dir = self.root / "runs" / self.run_id
        self.done = (self.dir / "DONE").exists()
        if self.done and not resume:
            raise RunExists(f"{self.run_id} is complete; refusing to overwrite (pass --resume to skip it)")
        if self.dir.exists() and not self.done:
            if not resume:
                raise RunExists(f"{self.dir} exists but is incomplete; pass --resume to redo it")
            shutil.rmtree(self.dir)                      # partial run from a crash: redo from scratch
        self.t0 = time.time()

    def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False)
        self.manifest = {"run_id": self.run_id, "tag": self.tag, "seed": self.seed, "config_hash": self.chash,
                         "smoke": self.cfg.get("smoke", False), "status": "running",
                         "started": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
                         "versions": versions(), "encoders": self.cfg["features"]["encoders"]}
        save_json(self.manifest, self.dir / "manifest.json")

    def json(self, name, obj):
        save_json(obj, self.dir / name)

    def preds(self, name, meta, logits, temps):
        df = pd.DataFrame({"row_id": meta["row_id"].to_numpy(), "patient_id": meta["patient_id"].to_numpy()})
        prob = 1 / (1 + np.exp(-logits / temps[None]))
        for j, l in enumerate(LABELS):
            df[f"logit_{l}"] = logits[:, j].astype(np.float32)
            df[f"prob_{l}"] = prob[:, j].astype(np.float32)
        df.to_parquet(self.dir / name, index=False)

    def finish(self, extra=None):
        self.manifest.update({"status": "done", "finished": datetime.now().isoformat(timespec="seconds"),
                              "wall_sec": round(time.time() - self.t0, 1), **(extra or {})})
        save_json(self.manifest, self.dir / "manifest.json")
        (self.dir / "DONE").write_text(self.manifest["finished"])


class TestGuard:
    """Gatekeeper for the official TEST set. Smoke mode or non-final runs can never touch it; a final run_id can read it once."""

    def __init__(self, cfg, writer, final):
        self.cfg, self.writer, self.final = cfg, writer, final
        self.log = Path(cfg["results_dir"]) / "test_access.jsonl"

    def check(self):
        if self.cfg.get("smoke"):
            raise TestAccessDenied("TEST evaluation is disabled in --smoke mode")
        if not self.final:
            raise TestAccessDenied("TEST is only evaluated by final runs (--final)")
        if self.log.exists():
            seen = {json.loads(l)["run_id"] for l in self.log.read_text().splitlines() if l.strip()}
            if self.writer.run_id in seen:
                raise TestAccessDenied(f"TEST was already evaluated for {self.writer.run_id}")

    def record(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a") as f:
            f.write(json.dumps({"run_id": self.writer.run_id, "time": datetime.now().isoformat(timespec="seconds")}) + "\n")
