"""Parse CSVs, build patient-disjoint splits and the non-IID partition (docs/03).

Full:  writes cache/meta_{train,test}.parquet (refuses to overwrite without --force).
Smoke: subsamples the full meta (1,500 pool + <=1,000 VAL patients), re-partitions, and slices the cached nih features
       into cache/smoke/. No TEST meta is written in smoke mode.
"""
import os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpmmfl.cli import parser, config
from dpmmfl.constants import LABELS, COMP5, TAB_COLS
from dpmmfl.data.parse import parse_csv, make_targets, age_stats, make_tab, comp5_counts
from dpmmfl.data.partition import assign_clients, partition_table
from dpmmfl.data.splits import assign_splits, assert_disjoint
from dpmmfl.data.store import feature_path
from dpmmfl.utils import load_config, save_json


def attach(df, y, m, tab):
    for j, l in enumerate(LABELS):
        df[f"R_{l}"] = df[l].to_numpy(float)
        df[f"T_{l}"], df[f"M_{l}"] = y[:, j], m[:, j]
    for j, c in enumerate(TAB_COLS):
        df[c] = tab[:, j]
    return df.drop(columns=LABELS)


def check_disjoint(tr, te, K):
    sets = {"val": set(tr[tr.split == "val"].patient_id)}
    if te is not None:
        sets["test"] = set(te.patient_id)
    for k in range(K):
        sets[f"client{k}"] = set(tr[(tr.split == "pool") & (tr.client_id == k)].patient_id)
    assert_disjoint(sets)
    print("patient overlap across " + "/".join(sets) + ": 0")


def prepare_full(cfg, force):
    out_tr = os.path.join(cfg["cache_dir"], "meta_train.parquet")
    if os.path.exists(out_tr) and not force:
        print(f"{out_tr} exists; not overwriting (use --force)"); return
    root = Path(cfg["data_root"])
    cs = root / "CheXpert-v1.0-small"
    tr, te = parse_csv(cs / "train.csv", root), parse_csv(cs / "valid.csv", root)
    print(f"frontal rows: train={len(tr)} ({tr.patient_id.nunique()} pts)  test={len(te)} ({te.patient_id.nunique()} pts)")
    tr = assign_splits(tr, cfg["val_frac"], cfg["seed"], cfg["subset_patients"])
    c5 = comp5_counts(tr)
    stats = age_stats(tr[tr["split"] == "pool"])
    y, m = make_targets(tr, cfg["uncertainty"], cfg["blank_as_negative"])
    tr = attach(tr.copy(), y, m, make_tab(tr, stats))
    p = cfg["partition"]
    tr, part = assign_clients(tr, c5, p["K"], p["alpha"], p["min_patients"], cfg["local_val_frac"], cfg["seed"])
    yt, mt = make_targets(te, cfg["uncertainty"], cfg["blank_as_negative"])
    assert mt.all(), "valid.csv must have no uncertain labels"
    te["split"], te["client_id"], te["local_val"] = "test", -1, False
    te = attach(te.copy(), yt, mt, make_tab(te, stats))
    check_disjoint(tr, te, p["K"])
    print(partition_table(tr, [f"T_{l}" for l in COMP5]).to_string(index=False))
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    tr.to_parquet(out_tr)
    te.to_parquet(os.path.join(cfg["cache_dir"], "meta_test.parquet"))
    save_json(part, os.path.join(cfg["splits_dir"], "partition.json"))
    save_json(stats, os.path.join(cfg["splits_dir"], "tab_stats.json"))


def prepare_subset(cfg, a):
    """Subsample the full meta/features into cfg.cache_dir (smoke, or a scale config with `subset:`)."""
    full = load_config()                                  # base config: the full cache is the source
    meta = pd.read_parquet(os.path.join(full["cache_dir"], "meta_train.parquet"))
    rng = np.random.default_rng(cfg["seed"])
    ss = cfg.get("subset") or cfg["smoke_subset"]
    pool_p = np.sort(meta[meta.split == "pool"].patient_id.unique())
    val_p = np.sort(meta[meta.split == "val"].patient_id.unique())
    keep = set(rng.choice(pool_p, min(ss["pool_patients"], len(pool_p)), replace=False)) | \
        set(rng.choice(val_p, min(ss["val_patients"], len(val_p)), replace=False))
    pos = np.flatnonzero(meta.patient_id.isin(keep).to_numpy())          # row positions in the full cache
    sm = meta.iloc[pos].drop(columns=["client_id", "local_val"]).reset_index(drop=True)
    c5 = (sm[[f"R_{l}" for l in COMP5]].to_numpy(float) == 1).astype(int)
    p = cfg["partition"]
    sm, part = assign_clients(sm, c5, p["K"], p["alpha"], p["min_patients"], cfg["local_val_frac"], cfg["seed"])
    check_disjoint(sm, None, p["K"])
    print(partition_table(sm, [f"T_{l}" for l in COMP5]).to_string(index=False))
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    sm.to_parquet(os.path.join(cfg["cache_dir"], "meta_train.parquet"))
    nih = np.load(feature_path(full["cache_dir"], "train", "nih"), mmap_mode="r")
    np.save(os.path.join(cfg["cache_dir"], "feat_train_nih.npy"), np.asarray(nih[pos]))
    np.save(os.path.join(cfg["cache_dir"], "feat_train_nih.rowid.npy"), sm["row_id"].to_numpy())
    save_json(part, os.path.join(cfg["splits_dir"], "partition.json"))
    if not cfg["smoke"]:                                  # scale runs evaluate TEST (final runs only): copy meta + test features
        shutil.copy(os.path.join(full["cache_dir"], "meta_test.parquet"), os.path.join(cfg["cache_dir"], "meta_test.parquet"))
        shutil.copy(feature_path(full["cache_dir"], "test", "nih"), os.path.join(cfg["cache_dir"], "feat_test_nih.npy"))
        te = pd.read_parquet(os.path.join(cfg["cache_dir"], "meta_test.parquet"))
        np.save(os.path.join(cfg["cache_dir"], "feat_test_nih.rowid.npy"), te["row_id"].to_numpy())
    print(f"subset: {len(sm)} images, pool {sm[sm.split == 'pool'].patient_id.nunique()} / val {sm[sm.split == 'val'].patient_id.nunique()} patients -> {cfg['cache_dir']}")


def main():
    ap = parser(__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing full meta cache")
    a = ap.parse_args()
    cfg = config(a)
    prepare_subset(cfg, a) if (cfg["smoke"] or cfg.get("subset")) else prepare_full(cfg, a.force)


if __name__ == "__main__":
    main()
