"""Collect demo images from CheXpert train.csv: a few per finding, with patient info, for the dashboard.

Writes demo/demo_samples.csv and copies the JPEGs to demo/images/. Held-out images come from VAL patients (never trained on);
one 'member' image per finding comes from the training pool (used for the membership-inference demo). TEST (valid.csv) is never read.

  python scripts/make_demo_set.py [--per-label 3]
"""
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpmmfl.cli import parser, config
from dpmmfl.constants import LABELS
from dpmmfl.data.parse import parse_csv


def main():
    ap = parser(__doc__)
    ap.add_argument("--per-label", type=int, default=3, help="held-out images per finding")
    ap.add_argument("--out", default="demo")
    a = ap.parse_args()
    cfg = config(a)
    root = Path(cfg["data_root"])
    df = parse_csv(root / "CheXpert-v1.0-small" / "train.csv", root)
    split = pd.read_parquet(Path(cfg["cache_dir"]) / "meta_train.parquet", columns=["patient_id", "split"]).drop_duplicates("patient_id")
    df = df.merge(split, on="patient_id", how="inner")
    df["n_pos"] = (df[LABELS] == 1).sum(1)
    rng = np.random.default_rng(0)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, used = [], set()
    for lab in LABELS:
        pos = df[(df[lab] == 1) & df["Sex"].isin(["Male", "Female"]) & df["Age"].notna() & df["AP/PA"].isin(["AP", "PA"])]
        pos = pos[pos.n_pos <= (1 if lab == "No Finding" else 3)]          # clean cases: few co-occurring findings
        pos = pos[~pos.patient_id.isin(used)].drop_duplicates("patient_id")
        for sp, k in (("val", a.per_label), ("pool", 1)):
            cand = pos[pos["split"] == sp]
            for r in cand.iloc[rng.permutation(len(cand))[:k]].itertuples():
                used.add(r.patient_id)
                dest = out / "images" / f"{lab.replace(' ', '_')}_{r.patient_id}_{sp}.jpg"
                shutil.copy(r.abs_path, dest)
                rec = dict(sample_id=dest.stem, finding=lab, image=str(dest).replace("\\", "/"), patient_id=r.patient_id, study=r.study_num,
                           sex=r.Sex, age=int(r.Age), projection=df.loc[r.Index, "AP/PA"],
                           source="held-out (VAL, never trained on)" if sp == "val" else "training pool (member)", split=sp)
                rec.update({f"label_{l}": df.loc[r.Index, l] for l in LABELS})
                rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(out / "demo_samples.csv", index=False)
    print(d.groupby(["finding", "split"]).size().unstack(fill_value=0).to_string())
    print(f"{len(d)} images -> {out}/demo_samples.csv")


if __name__ == "__main__":
    main()
