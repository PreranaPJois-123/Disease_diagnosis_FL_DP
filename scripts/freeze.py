"""Freeze VAL-selected choices into <results>/frozen.yaml (used by the final runs via --config).

  python scripts/freeze.py --stage model   # best of the trial_* runs: features.encoders, model.head/loss/ema
  python scripts/freeze.py --stage dp      # best of the dpsweep_* runs: dp.lr / dp.clip / dp.batch
Selection uses VAL mean-5 AUROC only.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml
from dpmmfl.cli import parser, config


def best(root, prefix):
    rows = []
    for d in (root / "runs").glob(f"*_{prefix}*"):          # run dirs are <exp>[_eps..]_<tag>__<hash>__s<seed>
        if (d / "DONE").exists():
            rows.append((json.loads((d / "metrics_val.json").read_text())["mean5_auroc"], d))
    if not rows:
        sys.exit(f"no completed {prefix}* runs under {root}")
    rows.sort(key=lambda r: -r[0] if r[0] == r[0] else 0)
    for v, d in rows:
        print(f"  {v:.4f}  {d.name}")
    return rows[0][0], yaml.safe_load((rows[0][1] / "config.yaml").read_text()), rows[0][1].name


def main():
    ap = parser(__doc__)
    ap.add_argument("--stage", choices=["model", "dp"], required=True)
    a = ap.parse_args()
    root = Path(config(a)["results_dir"])
    out = root / "frozen.yaml"
    frozen = yaml.safe_load(out.read_text()) if out.exists() else {}
    if a.stage == "model":
        v, c, name = best(root, "trial_")
        frozen["features"] = {"encoders": c["features"]["encoders"], "on_gpu": "auto"}
        frozen["model"] = {k: c["model"][k] for k in ("head", "loss", "ema")}
    else:
        v, c, name = best(root, "dpsweep_")
        frozen["dp"] = {k: c["dp"][k] for k in ("lr", "clip", "batch")}
    frozen.setdefault("_selected", {})[a.stage] = {"run": name, "val_mean5": round(v, 4)}
    out.write_text(yaml.safe_dump(frozen, sort_keys=False))
    print(f"frozen {a.stage}: {name} (VAL {v:.4f}) -> {out}")


if __name__ == "__main__":
    main()
