"""Copy the final large-scale models into models/<name>/ for demos and safekeeping.

  python scripts/export_model.py [--root results/v2.1-final] [--name v2.1-large] [--tags E4_eps3 E3 E2 E1] [--seed 0]

Each model folder holds head.pt, config.yaml, calibration.json, the VAL/TEST metrics, privacy.json (DP runs) and the manifest:
everything dpmmfl.serve.Predictor needs. models/<name>/models.json lists them. Refuses to overwrite an existing folder.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("head.pt", "config.yaml", "calibration.json", "metrics_val.json", "metrics_test.json", "privacy.json", "comms.json", "DONE",
         "fairness.json", "mia.json", "manifest.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/v2.1-final")
    ap.add_argument("--name", default="v2.1-large")
    ap.add_argument("--tags", nargs="+", default=["E4_eps3", "E3", "E2", "E1"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "models" / a.name
    if out.exists():
        sys.exit(f"{out} exists; not overwriting")
    index = []
    for tag in a.tags:
        runs = [d for d in sorted((ROOT / a.root / "runs").glob(f"{tag}__*__s{a.seed}"))
                if (d / "DONE").exists() and (d / "metrics_test.json").exists()]
        if not runs:
            print(f"skip {tag}: no finished final run for seed {a.seed}")
            continue
        src = runs[-1]
        dst = out / f"{tag}_s{a.seed}"
        dst.mkdir(parents=True)
        for f in FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        man = json.loads((src / "manifest.json").read_text())
        index.append({"model": dst.name, "run_id": man["run_id"], "encoders": man.get("encoders"), "val_mean5": man.get("val_mean5"),
                      "test_mean5": man.get("test_mean5"), "eps_spent_prv": man.get("eps_spent_prv")})
        print(f"{tag}: {src.name} -> {dst}")
    if not index:
        sys.exit("nothing exported")
    (out / "models.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} models -> {out}")


if __name__ == "__main__":
    main()
