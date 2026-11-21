"""Patient-disjoint splits (docs/03 s3.5)."""
import numpy as np


def assign_splits(meta, val_frac, seed, subset_patients=None):
    """meta: frontal train rows. Returns meta (possibly patient-subset) with split in {val,pool}."""
    pats = np.array(sorted(meta["patient_id"].unique()))
    rng = np.random.default_rng(seed)
    if subset_patients:
        pats = rng.choice(pats, size=min(subset_patients, len(pats)), replace=False)
        meta = meta[meta["patient_id"].isin(set(pats))].copy()
    pats = rng.permutation(np.array(sorted(pats)))
    n_val = max(1, int(round(len(pats) * val_frac)))
    val = set(pats[:n_val])
    meta = meta.copy()
    meta["split"] = np.where(meta["patient_id"].isin(val), "val", "pool")
    return meta


def assert_disjoint(sets: dict):
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = sets[a] & sets[b]
            assert not ov, f"patient overlap {a} vs {b}: {len(ov)}"                              # D3
