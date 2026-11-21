"""Masked multi-label losses (docs/04 s4.3). All are per-image and vmap-safe (pure tensor ops)."""
import torch
import torch.nn.functional as F

LOSSES = ("bce_pw", "asl")


def masked_bce_per_image(logits, y, m, pos_weight):
    """[..., 14] -> [...]: sum_j m*BCE / max(1, sum_j m)."""
    l = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    return (l * m).sum(-1) / m.sum(-1).clamp(min=1)


def masked_asl_per_image(logits, y, m, gamma_neg=4.0, gamma_pos=0.0, clip=0.05):
    """Asymmetric loss (Ridnik et al., 2021): focusing on positives/negatives separately, probability-shifted negatives."""
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip > 0 else p
    lp = y * (1 - p).pow(gamma_pos) * F.logsigmoid(logits)
    ln = (1 - y) * p_neg.pow(gamma_neg) * torch.log((1 - p_neg).clamp(min=1e-8))
    return (-(lp + ln) * m).sum(-1) / m.sum(-1).clamp(min=1)


def make_loss(cfg, pos_weight):
    """-> per_image(logits, y, m) -> [...]."""
    mc = cfg["model"]
    if mc["loss"] == "bce_pw":
        return lambda z, y, m: masked_bce_per_image(z, y, m, pos_weight)
    if mc["loss"] == "asl":
        a = mc["asl"]
        return lambda z, y, m: masked_asl_per_image(z, y, m, a["gamma_neg"], a["gamma_pos"], a["clip"])
    raise ValueError(mc["loss"])


def weighted_mean(per_image, w=None):
    """w: optional per-image weights (fairness F1, all <= 1)."""
    return (per_image if w is None else per_image * w).mean()
