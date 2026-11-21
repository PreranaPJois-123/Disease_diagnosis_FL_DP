"""Metrics (docs/07 s7.2). AUROC only where both classes exist; never feeds -1/NaN.

AUROC is computed from sorted scores (Mann-Whitney U, ties count 1/2): the same value as sklearn roc_auc_score but about 10x faster,
which matters because the bootstrap CIs need tens of thousands of AUROCs per run. Bootstraps are vectorised: a patient-level resample
is a vector of row multiplicities, and the AUROC of a resample is the weighted AUROC under those multiplicities (no re-sorting).
"""
import numpy as np
from sklearn.metrics import average_precision_score
from ..constants import LABELS, COMP5, COMP5_IDX


def weighted_auroc(y, s, W, min_pos=1):
    """AUROC for each row-weight vector in W [B, n] (bootstrap multiplicities) -> [B]; NaN where a class is missing or has
    fewer than min_pos (weighted) positives. Equals the AUROC of the resampled rows: sort once, group tied scores, then
    P(pos > neg) + P(tie)/2 from weighted cumulative counts."""
    if len(s) == 0:
        return np.full(len(W), np.nan)
    order = np.argsort(s, kind="stable")
    ss, pos = s[order], (y[order] == 1)
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    Wo = W[:, order]
    P = np.add.reduceat(Wo * pos, starts, axis=1)
    N = np.add.reduceat(Wo * ~pos, starts, axis=1)
    tp, tn = P.sum(1), N.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (P * (np.cumsum(N, 1) - 0.5 * N)).sum(1) / (tp * tn)
    auc[(tp < min_pos) | (tn == 0)] = np.nan
    return auc


def safe_auroc(y, s, m=None, min_pos=1):
    if m is not None:
        keep = m > 0
        y, s = y[keep], s[keep]
    return float(weighted_auroc(y, s, np.ones((1, len(y))), min_pos)[0])


def boot_weights(patients, n, seed=0):
    """[n, rows] multiplicity of every row in n patient-level bootstrap resamples (patients drawn with replacement)."""
    _, inv = np.unique(patients, return_inverse=True)
    k = int(inv.max()) + 1 if len(inv) else 0
    cnt = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=n) if k else np.zeros((n, 0), int)
    return cnt[:, inv].astype(np.float64)


def boot_mean5(y, p, m, W, min_pos=1, max_cells=4_000_000):
    """Mean comp-5 AUROC (undefined labels skipped) for every weight vector in W [B, N] -> [B], in memory-bounded blocks."""
    out = []
    step = max(1, max_cells // max(1, y.shape[0]))
    for b in range(0, len(W), step):
        Wb = W[b:b + step]
        A = np.stack([weighted_auroc(y[k, j], p[k, j], Wb[:, k], min_pos) for j in COMP5_IDX for k in [m[:, j] > 0]], 1)
        ok = ~np.isnan(A)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(ok.any(1), np.nansum(A, 1) / ok.sum(1), np.nan))
    return np.concatenate(out) if out else np.zeros(0)


def per_label(fn, y, p, m, min_pos=1):
    return {l: fn(y[:, j], p[:, j], m[:, j], min_pos) for j, l in enumerate(LABELS)}


def _auprc(y, s, m, min_pos):
    keep = m > 0
    y, s = y[keep], s[keep]
    return float(average_precision_score(y, s)) if (y == 1).sum() >= min_pos and (y == 0).any() else np.nan


def mean_comp5(d):
    v = [d[l] for l in COMP5 if not np.isnan(d[l])]
    return float(np.mean(v)) if v else np.nan


def bootstrap_ci(y, p, m, patients, n=1000, seed=0, min_pos=1):
    """Patient-level bootstrap 95% CI of mean comp-5 AUROC ([nan, nan] if no resample is defined)."""
    if n <= 0 or len(y) == 0:
        return [np.nan, np.nan]
    vals = boot_mean5(y, p, m, boot_weights(patients, n, seed), min_pos)
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [np.nan, np.nan]


def youden_thresholds(y, p, m):
    thr = np.full(len(LABELS), 0.5)
    for j in range(len(LABELS)):
        k = m[:, j] > 0
        yy, pp = y[k, j], p[k, j]
        if (yy == 1).sum() < 1 or (yy == 0).sum() < 1:
            continue
        cand = np.unique(np.quantile(pp, np.linspace(0.01, 0.99, 99)))
        j_stat = [((pp >= t)[yy == 1].mean() + (pp < t)[yy == 0].mean() - 1) for t in cand]
        thr[j] = cand[int(np.argmax(j_stat))]
    return thr


def threshold_metrics(y, p, m, thr):
    out = {}
    for j, l in enumerate(LABELS):
        k = m[:, j] > 0
        yy, pr = y[k, j], p[k, j] >= thr[j]
        if (yy == 1).sum() < 1 or (yy == 0).sum() < 1:
            continue
        tp, fp, fn, tn = ((pr & (yy == 1)).sum(), (pr & (yy == 0)).sum(), (~pr & (yy == 1)).sum(), (~pr & (yy == 0)).sum())
        out[l] = {"sens": float(tp / max(1, tp + fn)), "spec": float(tn / max(1, tn + fp)),
                  "f1": float(2 * tp / max(1, 2 * tp + fp + fn))}
    return out


def ece(y, p, m, bins=15):
    vals = []
    for j in range(len(LABELS)):
        k = m[:, j] > 0
        yy, pp = y[k, j], p[k, j]
        if len(yy) == 0:
            continue
        b = np.minimum((pp * bins).astype(int), bins - 1)
        vals.append(sum(abs(yy[b == i].mean() - pp[b == i].mean()) * (b == i).mean() for i in np.unique(b)))
    return float(np.mean(vals))


T_GRID = np.exp(np.linspace(-1.5, 1.5, 61))


def fit_temperature(z, y, m, grid=T_GRID):
    """Per-label temperature T_j minimising masked BCE of sigmoid(z/T_j) (fit on VAL only). Monotone: AUROC unchanged."""
    T = np.ones(z.shape[1])
    for j in range(z.shape[1]):
        k = m[:, j] > 0
        if k.sum() == 0 or len(np.unique(y[k, j])) < 2:
            continue
        zz, yy = z[k, j][:, None] / grid[None], y[k, j][:, None]
        nll = (np.logaddexp(0, zz) - yy * zz).mean(0)
        T[j] = grid[int(np.argmin(nll))]
    return T


def evaluate(y, p, m, patients, min_pos, n_boot=0, thr=None):
    """y,p,m: numpy [N,14]. Returns dict with per-label metrics and comp-5 mean."""
    auc = per_label(safe_auroc, y, p, m, min_pos)
    res = {"auroc": auc, "auprc": per_label(_auprc, y, p, m, min_pos), "mean5_auroc": mean_comp5(auc),
           "mean_all_auroc": float(np.nanmean([v for v in auc.values()])) if any(~np.isnan(list(auc.values()))) else np.nan,
           "ece": ece(y, p, m)}
    if n_boot:
        res["mean5_auroc_ci95"] = bootstrap_ci(y, p, m, patients, n_boot)
    if thr is not None:
        res["threshold_metrics"] = threshold_metrics(y, p, m, thr)
    return res
