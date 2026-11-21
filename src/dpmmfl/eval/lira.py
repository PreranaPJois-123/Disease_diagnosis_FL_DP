"""Likelihood-ratio membership inference (LiRA, Carlini et al. 2022) at patient level, plus the DP audit quantities.

The per-patient statistic is -log(mean training loss over the patient's images), a log transform that makes it roughly Gaussian
across models (the point of LiRA). On the overfit shadow models it was a much stronger signal than the mean signed logit
(online LiRA TPR at 1% FPR 0.43 vs 0.09); an attacker can make that choice with their own shadow models. Shadow (or reference) models trained with and
without a patient give that patient's IN and OUT distributions; the attack scores how much more likely the target model's
statistic is under IN than under OUT. Online LiRA uses IN and OUT models; offline LiRA uses OUT models only.

DP quantities: the (eps, delta) bound on any attack's ROC curve, and an empirical epsilon lower bound from an attack's
error rates (Clopper-Pearson), which must stay below the claimed epsilon if the DP implementation is correct.
"""
import numpy as np
import torch
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve


@torch.no_grad()
def patient_stats(model, store, rows, loss_fn, bs=8192):
    """Per-patient (patient index, attack statistic -log(mean loss), mean loss) for `rows`. Patients are store.pat indices."""
    model.eval()
    rows = np.asarray(rows)
    loss = []
    for i in range(0, len(rows), bs):
        x, t, y, m, _ = store.batch(rows[i:i + bs])
        loss.append(loss_fn(model(x, t).float(), y, m).cpu().numpy())
    pats, inv = np.unique(store.pat[rows], return_inverse=True)
    loss = np.bincount(inv, np.concatenate(loss)) / np.bincount(inv)
    return pats, -np.log(loss + 1e-8), loss


def _logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


def lira_online(target, S, IN):
    """target [P]; S [K, P] shadow statistics; IN [K, P] bool. Per-patient means, global (pooled) variance (Carlini et al.,
    fixed-variance variant, robust with few shadows). Patients without an IN or OUT shadow get nan."""
    n_in, n_out = IN.sum(0), (~IN).sum(0)
    mu_in = np.where(IN, S, 0).sum(0) / np.maximum(n_in, 1)
    mu_out = np.where(~IN, S, 0).sum(0) / np.maximum(n_out, 1)
    sd_in = np.sqrt((np.where(IN, S - mu_in, 0) ** 2).sum() / max(IN.sum() - 1, 1)) + 1e-6
    sd_out = np.sqrt((np.where(~IN, S - mu_out, 0) ** 2).sum() / max((~IN).sum() - 1, 1)) + 1e-6
    sc = _logpdf(target, mu_in, sd_in) - _logpdf(target, mu_out, sd_out)
    return np.where((n_in > 0) & (n_out > 0), sc, np.nan)


def lira_offline(target, S, OUT=None):
    """target [P]; S [R, P] statistics of reference models; OUT [R, P] marks which of them did not train on each patient
    (default: none did). One-sided z-score against the OUT distribution, pooled variance."""
    OUT = np.ones(S.shape, bool) if OUT is None else OUT
    n = OUT.sum(0)
    mu = np.where(OUT, S, 0).sum(0) / np.maximum(n, 1)
    sd = np.sqrt((np.where(OUT, S - mu, 0) ** 2).sum() / max(OUT.sum() - 1, 1)) + 1e-6
    return np.where(n > 0, (target - mu) / sd, np.nan)


def roc_summary(scores, member, fprs=(0.01, 0.001)):
    """AUC, TPR at fixed low FPRs (the metric that matters for privacy) and the ROC curve. Higher score = 'member'."""
    ok = np.isfinite(scores)
    s, y = scores[ok], member[ok].astype(int)
    fpr, tpr, _ = roc_curve(y, s)
    out = {"auc": float(roc_auc_score(y, s)), "n": int(len(s)), "n_members": int(y.sum())}
    for f in fprs:
        out[f"tpr@{f:g}"] = float(tpr[np.searchsorted(fpr, f, side="right") - 1])
    return out, fpr, tpr


def dp_roc_bound(eps, delta, fpr):
    """Highest TPR any attack can reach at a given FPR against an (eps, delta)-DP mechanism (Kairouz et al. 2015)."""
    fpr = np.asarray(fpr, float)
    return np.clip(np.minimum(np.exp(eps) * fpr + delta, 1 - np.exp(-eps) * (1 - delta - fpr)), 0, 1)


def _cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def _cp_bound(s, y, t, delta, alpha):
    """Epsilon lower bound at threshold t from Clopper-Pearson upper bounds on FPR and FNR (Bonferroni over the two rates)."""
    fpr_u = _cp_upper(int((s[~y] >= t).sum()), int((~y).sum()), alpha / 2)
    fnr_u = _cp_upper(int((s[y] < t).sum()), int(y.sum()), alpha / 2)
    return max(np.log(max(1 - delta - fpr_u, 1e-12) / max(fnr_u, 1e-12)), np.log(max(1 - delta - fnr_u, 1e-12) / max(fpr_u, 1e-12)), 0.0)


def eps_lower_bound(scores, member, delta, alpha=0.05, seed=0):
    """Empirical epsilon lower bound (Jagielski et al. 2020): eps_lb = max(log((1 - delta - FPR_u) / FNR_u),
    log((1 - delta - FNR_u) / FPR_u)) with 95% Clopper-Pearson upper bounds on the error rates. The threshold is the one with the
    best bound on a random half of the trials; the bound is then computed on the other half, so selection does not inflate it.
    Trials across shadow models are not fully independent, so treat it as an estimate."""
    ok = np.isfinite(scores)
    s, y = scores[ok], member[ok].astype(bool)
    idx = np.random.default_rng(seed).permutation(len(s))
    a, b = idx[: len(s) // 2], idx[len(s) // 2:]
    cand = np.quantile(s[a], np.linspace(0.001, 0.999, 999))
    t = cand[int(np.argmax([_cp_bound(s[a], y[a], c, delta, alpha) for c in cand]))]
    return float(_cp_bound(s[b], y[b], t, delta, alpha))
