"""Dirichlet non-IID partition of PATIENTS across K clients (docs/03 s3.6)."""
import numpy as np
import pandas as pd


def partition_patients(patients, counts, K, alpha, min_patients, seed):
    """patients: ids; counts: [P,5] comp-5 positive counts. Returns client id per patient."""
    rng = np.random.default_rng(seed)
    P = len(patients)
    assert K * min_patients <= P, f"need >= {K * min_patients} patients, have {P}"
    cls = np.where(counts.max(1) > 0, counts.argmax(1), counts.shape[1])   # proxy class (5 = none)
    client = np.zeros(P, int)
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        client[idx] = rng.choice(K, size=len(idx), p=rng.dirichlet(alpha * np.ones(K)))
    while True:                                                            # enforce min size
        sizes = np.bincount(client, minlength=K)
        k = sizes.argmin()
        if sizes[k] >= min_patients:
            break
        client[rng.choice(np.where(client == sizes.argmax())[0])] = k
    return client


def assign_clients(meta, counts_by_row, K, alpha, min_patients, local_val_frac, seed):
    """Adds client_id (-1 outside pool) and local_val to meta. counts_by_row: [N,5] aligned with meta."""
    meta = meta.copy()
    is_pool = (meta["split"] == "pool").to_numpy()
    pool_pat = meta.loc[is_pool, "patient_id"].to_numpy()
    patients, inv = np.unique(pool_pat, return_inverse=True)
    counts = np.zeros((len(patients), counts_by_row.shape[1]), int)
    np.add.at(counts, inv, counts_by_row[is_pool])
    client = partition_patients(patients, counts, K, alpha, min_patients, seed)
    rng = np.random.default_rng(seed + 1)
    cmap, lv = dict(zip(patients, client)), set()
    for k in range(K):
        pk = patients[client == k]
        lv |= set(rng.choice(pk, size=max(1, int(round(len(pk) * local_val_frac))), replace=False))
    meta["client_id"] = np.where(is_pool, meta["patient_id"].map(cmap).fillna(-1), -1).astype(int)
    meta["local_val"] = meta["patient_id"].isin(lv) & is_pool
    return meta, {int(k): [str(p) for p in patients[client == k]] for k in range(K)}


def partition_table(meta, label_cols):
    pool = meta[meta["split"] == "pool"]
    n_tot = pool["patient_id"].nunique()
    rows = []
    for k, d in pool.groupby("client_id"):
        r = {"client": k, "patients": d["patient_id"].nunique(), "images": len(d)}
        r["p_k"] = round(r["patients"] / n_tot, 3)
        r.update({c: round(float(d[c].mean()), 3) for c in label_cols})
        rows.append(r)
    return pd.DataFrame(rows)
