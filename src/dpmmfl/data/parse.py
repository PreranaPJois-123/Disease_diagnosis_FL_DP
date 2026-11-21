"""CSV -> meta. patient/study/view come from the Path string ONLY (docs/03 s3.2)."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from ..constants import LABELS, COMP5_IDX, TAB_COLS, DEFAULT_POLICY

PATH_RE = re.compile(r"(?:^|/)(?:train|valid)/(patient\d+)/study(\d+)/view\d+_(frontal|lateral)\.jpg$")


def parse_csv(csv_path, root):
    df = pd.read_csv(csv_path)
    assert list(df.columns[5:19]) == LABELS, "label column order differs from constants.LABELS"   # D5
    df.insert(0, "row_id", np.arange(len(df)))                                                   # original row index
    ex = df["Path"].str.extract(PATH_RE)
    assert ex.notna().all().all(), "D2: Path regex failed on some rows"                          # D2
    df["patient_id"] = ex[0]
    df["study_num"] = ex[1].astype(int)
    df["view"] = ex[2]
    assert (df["view"].str.capitalize() == df["Frontal/Lateral"]).all(), "Path view != CSV Frontal/Lateral"
    df = df[df["view"] == "frontal"].copy()
    df["abs_path"] = [str(Path(root) / p) for p in df["Path"]]
    for p in df.sample(min(1000, len(df)), random_state=0)["abs_path"]:
        assert Path(p).exists(), f"missing image {p}"
    return df.reset_index(drop=True)


def make_targets(df, policy=None, blank_as_negative=True):
    """Returns (y, m) float32 [N,14] per the uncertainty policy in docs/03 s3.3."""
    policy = DEFAULT_POLICY if policy is None else policy
    raw = df[LABELS].to_numpy(dtype=float)
    y = (raw == 1).astype(np.float32)
    m = np.ones_like(y)
    for j, l in enumerate(LABELS):
        unc = raw[:, j] == -1
        pol = policy.get(l, "ignore")
        if pol == "ones":
            y[unc, j] = 1
        elif pol == "zeros":
            y[unc, j] = 0
        else:
            m[unc, j] = 0
        if not blank_as_negative:
            m[np.isnan(raw[:, j]), j] = 0
    assert set(np.unique(y)) <= {0.0, 1.0} and set(np.unique(m)) <= {0.0, 1.0}                   # D4
    return y, m


def age_stats(df):
    a = pd.to_numeric(df["Age"], errors="coerce")
    a = a[a > 0].clip(18, 90)
    return {"mean": float(a.mean()), "std": float(a.std() + 1e-6)}


def make_tab(df, stats):
    age = pd.to_numeric(df["Age"], errors="coerce").to_numpy(float)
    missing = ~(age > 0)
    z = (np.clip(np.nan_to_num(age), 18, 90) - stats["mean"]) / stats["std"]
    z[missing] = 0.0
    sex = df["Sex"].fillna("Unknown").to_numpy()
    proj = df["AP/PA"].fillna("Other").to_numpy()
    prior = np.log1p(df["study_num"].to_numpy(float) - 1)
    tab = np.stack([z, missing.astype(float), sex == "Male", sex == "Female",
                    ~np.isin(sex, ["Male", "Female"]), proj == "AP", proj == "PA",
                    ~np.isin(proj, ["AP", "PA"]), prior], 1).astype(np.float32)
    assert tab.shape[1] == len(TAB_COLS)
    return tab


def comp5_counts(df):
    return (df[[LABELS[i] for i in COMP5_IDX]].to_numpy(float) == 1).astype(int)
