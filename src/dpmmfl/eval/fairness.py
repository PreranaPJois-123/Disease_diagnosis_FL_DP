"""Fairness audit and mitigations (docs/07 s7.3).

Audit per attribute (sex, age bin, AP/PA; clients on their local-val): mean comp-5 AUROC with patient-level bootstrap CI,
worst-group AUROC, max gap, equalised-odds TPR/FPR gaps at the operating thresholds, per-group ECE.
F1: group-reweighted loss, weights <= 1 (so the per-patient DP sensitivity bound is untouched).
F2: per-group decision thresholds (post-processing), fit on one half of VAL patients and evaluated on the other half.
"""
import numpy as np
import pandas as pd
from ..constants import COMP5, COMP5_IDX, AGE_BINS, AGE_LABELS
from .metrics import bootstrap_ci, safe_auroc, mean_comp5, ece, youden_thresholds


def attributes(meta):
    age = pd.to_numeric(meta["Age"], errors="coerce")
    return {"sex": meta["Sex"].where(meta["Sex"].isin(["Male", "Female"])),
            "age": pd.cut(age, AGE_BINS, right=False, labels=AGE_LABELS).astype(object),
            "projection": meta["AP/PA"].where(meta["AP/PA"].isin(["AP", "PA"]))}


def _mean5(y, p, m, min_pos):
    return mean_comp5({l: safe_auroc(y[:, j], p[:, j], m[:, j], min_pos) for l, j in zip(COMP5, COMP5_IDX)})


def _rates(y, p, m, thr):
    """Mean comp-5 TPR and FPR at per-label thresholds (thr may be [14] or [N,14])."""
    tpr, fpr = [], []
    for j in COMP5_IDX:
        k = m[:, j] > 0
        t = thr[k, j] if np.ndim(thr) == 2 else thr[j]
        pr, yy = p[k, j] >= t, y[k, j]
        if (yy == 1).any():
            tpr.append(pr[yy == 1].mean())
        if (yy == 0).any():
            fpr.append(pr[yy == 0].mean())
    return (float(np.mean(tpr)) if tpr else np.nan, float(np.mean(fpr)) if fpr else np.nan)


def _boot_ci(y, p, m, pats, n, min_pos, seed):
    return bootstrap_ci(y, p, m, pats, n, seed, min_pos)


def summarize(res):
    """res: {group: {auroc, tpr, fpr, ece}} -> attribute-level gaps."""
    g = {k: v for k, v in res.items() if not np.isnan(v["auroc"])}
    if len(g) < 2:
        return {"worst_group": None, "worst_auroc": np.nan, "max_gap": np.nan, "eo_tpr_gap": np.nan, "eo_fpr_gap": np.nan, "ece_gap": np.nan}
    span = lambda key: float(np.nanmax([v[key] for v in g.values()]) - np.nanmin([v[key] for v in g.values()]))
    worst = min(g, key=lambda k: g[k]["auroc"])
    return {"worst_group": worst, "worst_auroc": g[worst]["auroc"], "max_gap": span("auroc"),
            "eo_tpr_gap": span("tpr"), "eo_fpr_gap": span("fpr"), "ece_gap": span("ece")}


def group_report(y, p, m, pats, groups, thr, n_boot=200, min_pos=5, seed=0):
    """groups: Series/array of group labels aligned with rows (NaN = excluded)."""
    g = pd.Series(np.asarray(groups, dtype=object))
    res = {}
    for name in sorted(g.dropna().unique(), key=str):
        idx = np.flatnonzero((g == name).to_numpy())
        tpr, fpr = _rates(y[idx], p[idx], m[idx], thr[idx] if np.ndim(thr) == 2 else thr)
        res[str(name)] = {"n": int(len(idx)), "auroc": _mean5(y[idx], p[idx], m[idx], min_pos),
                          "ci95": _boot_ci(y[idx], p[idx], m[idx], pats[idx], n_boot, min_pos, seed),
                          "tpr": tpr, "fpr": fpr, "ece": ece(y[idx], p[idx], m[idx])}
    return {"groups": res, **summarize(res)}


def subgroup_audit(meta, y, p, m, thr, n_boot=200, min_pos=5):
    pats = meta["patient_id"].to_numpy()
    return {a: group_report(y, p, m, pats, g.to_numpy(), thr, n_boot, min_pos) for a, g in attributes(meta).items()}


def client_audit(meta_all, rows_by_client, predict_fn, y_fn, thr, n_boot=200, min_pos=5):
    """Global model on each client's local-val rows (one group per client)."""
    ys, ps, ms, pats, grp = [], [], [], [], []
    for k, rows in rows_by_client.items():
        if len(rows) == 0:
            continue
        y, m = y_fn(rows)
        ys.append(y); ms.append(m); ps.append(predict_fn(rows))
        pats.append(meta_all["patient_id"].to_numpy()[rows]); grp += [str(k)] * len(rows)
    if not ys:
        return {}
    return group_report(np.concatenate(ys), np.concatenate(ps), np.concatenate(ms), np.concatenate(pats),
                        np.array(grp, dtype=object), thr, n_boot, min_pos)


# ---------------------------------------------------------------- F1: group reweighting (w <= 1)
def group_keys(meta, attrs):
    age = pd.to_numeric(meta["Age"], errors="coerce")
    parts = {"age": pd.cut(age, AGE_BINS, right=False, labels=AGE_LABELS).astype(object).fillna("unk"),
             "projection": meta["AP/PA"].where(meta["AP/PA"].isin(["AP", "PA"]), "Other"),
             "sex": meta["Sex"].where(meta["Sex"].isin(["Male", "Female"]), "Unknown")}
    key = None
    for a in attrs:
        key = parts[a].astype(str) if key is None else key + "|" + parts[a].astype(str)
    return key


def group_weights(meta, rows, attrs=("age", "projection"), wmin=0.2, min_share=0.01):
    """Inverse-frequency weights scaled so the rarest (non-tiny) group gets 1 and every weight is in [wmin, 1].
    Groups smaller than min_share of rows are capped at weight 1. Returns (weights aligned with rows, group sizes)."""
    key = group_keys(meta.iloc[rows], attrs)
    n = key.value_counts()
    ref = n[n >= min_share * len(key)].min()
    w = np.clip(ref / key.map(n).to_numpy(float), wmin, 1.0)
    return w.astype(np.float32), n.to_dict()


# ---------------------------------------------------------------- F2: per-group thresholds
def group_thresholds(meta, y, p, m, attr, seed=0):
    """Fit global and per-group Youden thresholds on half A of VAL patients; report EO gaps on half B for both.
    Returns (report dict, {group: thresholds[14]})."""
    pats = meta["patient_id"].to_numpy()
    u = np.unique(pats)
    a_set = set(np.random.default_rng(seed).permutation(u)[: len(u) // 2])
    A = np.array([q in a_set for q in pats])
    g = attributes(meta)[attr].to_numpy(dtype=object)
    thr_global = youden_thresholds(y[A], p[A], m[A])
    thr_g = {}
    for name in pd.Series(g).dropna().unique():
        idx = A & (g == name)
        thr_g[str(name)] = youden_thresholds(y[idx], p[idx], m[idx]) if idx.sum() >= 50 else thr_global
    B = ~A
    per_row = np.stack([thr_g.get(str(x), thr_global) if x == x and x is not None else thr_global for x in g[B]])
    rep_global = group_report(y[B], p[B], m[B], pats[B], g[B], thr_global, 0)
    rep_group = group_report(y[B], p[B], m[B], pats[B], g[B], per_row, 0)
    keys = ("eo_tpr_gap", "eo_fpr_gap")
    return ({"attr": attr, "eval": "VAL half B (thresholds fit on half A)",
             "global_thresholds": {k: rep_global[k] for k in keys},
             "group_thresholds": {k: rep_group[k] for k in keys},
             "groups_global": {k: {"tpr": v["tpr"], "fpr": v["fpr"]} for k, v in rep_global["groups"].items()},
             "groups_group": {k: {"tpr": v["tpr"], "fpr": v["fpr"]} for k, v in rep_group["groups"].items()}},
            {k: v.tolist() for k, v in thr_g.items()})
