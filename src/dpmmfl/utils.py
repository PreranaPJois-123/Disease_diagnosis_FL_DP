import copy, hashlib, json, logging, random, sys
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def deep_update(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_sets(cfg, sets):
    """--set a.b=value overrides (value parsed as YAML)."""
    for kv in sets or []:
        k, v = kv.split("=", 1)
        node = cfg
        for part in k.split(".")[:-1]:
            node = node.setdefault(part, {})
        node[k.split(".")[-1]] = yaml.safe_load(v)
    return cfg


def load_config(*paths, smoke=False, sets=None):
    """base.yaml <- override files <- smoke.yaml (if smoke) <- --set overrides. Paths become absolute."""
    files = [ROOT / "configs" / "base.yaml", *[Path(x) for x in paths]]
    if smoke:
        files.append(ROOT / "configs" / "smoke.yaml")
    cfg = {}
    for p in files:
        with open(p) as f:
            cfg = deep_update(cfg, yaml.safe_load(f))
    cfg = apply_sets(cfg, sets)
    cfg["smoke"] = bool(smoke)
    for k in ("cache_dir", "splits_dir", "results_dir"):
        cfg[k] = str((ROOT / cfg[k]).resolve())
    return cfg


def config_hash(obj, n=8):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    d = cfg.get("device", "auto")
    return torch.device("cuda" if d == "auto" and torch.cuda.is_available() else ("cpu" if d == "auto" else d))


def get_logger(name="dpmmfl", file=None):
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(h); log.setLevel(logging.INFO); log.propagate = False
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(fh)
    return log


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)
