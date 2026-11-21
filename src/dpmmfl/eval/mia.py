"""E9: loss-threshold membership-inference audit (docs/06 s6.6): per-patient mean loss, training patients vs VAL patients."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


@torch.no_grad()
def patient_losses(model, store, rows, loss_fn, bs=8192):
    model.eval()
    rows = np.asarray(rows)
    l = []
    for i in range(0, len(rows), bs):
        x, t, y, m, _ = store.batch(rows[i:i + bs])
        l.append(loss_fn(model(x, t).float(), y, m).cpu().numpy())
    l = np.concatenate(l)
    _, inv = np.unique(store.pat[rows], return_inverse=True)
    return np.bincount(inv, l) / np.bincount(inv)


def tpr_at_fpr(member, score, fpr_max=0.01):
    """Share of members flagged when at most `fpr_max` of non-members are (the low-FPR regime that matters for privacy)."""
    fpr, tpr, _ = roc_curve(member, score)
    return float(tpr[np.searchsorted(fpr, fpr_max, side="right") - 1])


def mia_audit(model, store, member_rows, nonmember_rows, loss_fn, seed=0):
    """Balanced attack AUC (members subsampled to the number of non-member patients). 0.5 = no signal."""
    lm, ln = patient_losses(model, store, member_rows, loss_fn), patient_losses(model, store, nonmember_rows, loss_fn)
    n = min(len(lm), len(ln))
    rng = np.random.default_rng(seed)
    lm, ln = rng.choice(lm, n, replace=False), rng.choice(ln, n, replace=False)
    lab, sc = np.r_[np.ones(n), np.zeros(n)], -np.r_[lm, ln]
    auc = float(roc_auc_score(lab, sc))
    return {"mia_auc": auc, "advantage": abs(2 * auc - 1), "tpr_at_1pct_fpr": tpr_at_fpr(lab, sc), "n_patients_each": int(n),
            "member_loss_mean": float(lm.mean()), "nonmember_loss_mean": float(ln.mean())}
