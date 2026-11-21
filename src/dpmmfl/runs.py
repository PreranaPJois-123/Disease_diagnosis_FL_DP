"""Run discovery without importing torch (keeps the dashboard light until a prediction is made)."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _scale_of(p):
    parts = p.relative_to(ROOT / "results").parts
    return {"scale_small": "small", "scale_medium": "medium", "v2.1-final": "large (v2.1)", "v2-large": "large (v2)"}.get(parts[0], parts[0])


def discover_runs(results=ROOT / "results", models=ROOT / "models"):
    """One row per run that has a head.pt: scale, config tag, seed, val/test mean-5 AUROC, path.
    Also lists models saved by scripts/export_model.py (scale "saved (<name>)").
    `predictable` is False for archived v2 runs: their head layout predates the v2.1 FusionHead (results only)."""
    rows = []
    heads = [(h, None) for h in sorted(results.glob("**/head.pt"))] + [(h, "saved") for h in sorted(models.glob("*/*/head.pt"))]
    for head, kind in heads:
        d = head.parent
        if "smoke" in d.parts or "archive" in d.parts:
            continue
        man = d / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            tag, seed, val, test = m["tag"], m["seed"], m.get("val_mean5"), m.get("test_mean5")
            if m.get("status") != "done" or not (d / "DONE").exists():
                continue
        else:                                                    # v2 layout: results/v2-large/<tag>/<seed>/
            tag, seed = d.parent.name, int(d.name)
            mv, mt = d / "metrics_val.json", d / "metrics_test.json"
            val = json.loads(mv.read_text()).get("mean5_auroc") if mv.exists() else None
            test = json.loads(mt.read_text()).get("mean5_auroc") if mt.exists() else None
        rows.append(dict(scale=_scale_of(d) if kind is None else f"saved ({d.parent.name})", tag=tag, seed=seed, val=val, test=test, path=str(d), predictable=man.exists()))
    return pd.DataFrame(rows)


