"""Batched inference and VAL scoring on cached features."""
import numpy as np
import torch
from ..eval.metrics import mean_comp5, per_label, safe_auroc


@torch.no_grad()
def predict_logits(model, store, rows=None, bs=8192):
    model.eval()
    rows = np.arange(len(store)) if rows is None else np.asarray(rows)
    out = []
    for i in range(0, len(rows), bs):
        x, t, _, _, _ = store.batch(rows[i:i + bs])
        out.append(model(x, t).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, store.y.shape[1]), np.float32)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def val_score(model, store, rows, min_pos):
    """Mean comp-5 AUROC on `rows` (model selection / early stopping)."""
    z = predict_logits(model, store, rows)
    y, m = store.numpy_targets(rows)
    return mean_comp5(per_label(safe_auroc, y, z, m, min_pos))


class EarlyStopper:
    def __init__(self, patience):
        self.patience, self.best, self.bad, self.best_step, self.state = patience, -np.inf, 0, None, None

    def update(self, score, step, model):
        """Returns True when training should stop."""
        if np.isfinite(score) and score > self.best:
            self.best, self.bad, self.best_step = score, 0, step
            self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.patience > 0 and self.bad >= self.patience
