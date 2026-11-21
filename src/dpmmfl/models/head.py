"""Multi-encoder head (docs/04 s4.2). No BatchNorm anywhere (DP-safe).

Each encoder's 1024-d feature gets its own LayerNorm -> Linear(1024, d); the projections are concatenated.
  linear     : concat -> Linear(14). A linear probe (no metadata).
  image_only : concat -> GELU -> Linear(d) -> residual MLP -> Linear(14).
  film       : image_only + FiLM conditioning on the 9 metadata features (zero-init, starts as image_only).
"""
import copy
import torch
import torch.nn as nn
from ..constants import D_IMG, D_TAB, N_LABELS

HEADS = ("film", "image_only", "linear")


class FusionHead(nn.Module):
    def __init__(self, n_enc=1, head="film", d=256, d_img=D_IMG, d_tab=D_TAB, n_labels=N_LABELS, p_drop=0.2, p_modality_drop=0.1):
        super().__init__()
        assert head in HEADS, head
        self.head, self.p_mod = head, p_modality_drop
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_img), nn.Linear(d_img, d)) for _ in range(n_enc)])
        if head == "linear":
            self.cls = nn.Linear(n_enc * d, n_labels)
            return
        self.mix = nn.Sequential(nn.GELU(), nn.Dropout(p_drop), nn.Linear(n_enc * d, d))
        if head == "film":
            self.tab = nn.Sequential(nn.Linear(d_tab, 64), nn.GELU(), nn.Linear(64, 2 * d))
            nn.init.zeros_(self.tab[-1].weight)
            nn.init.zeros_(self.tab[-1].bias)
        self.post = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p_drop))
        self.cls = nn.Linear(d, n_labels)

    def forward(self, f_img, f_tab=None):
        """f_img [..., E, 1024]; f_tab [..., 9]."""
        z = torch.cat([p(f_img[..., e, :]) for e, p in enumerate(self.proj)], dim=-1)
        if self.head == "linear":
            return self.cls(z)
        h = self.mix(z)
        if self.head == "film":
            if self.training and self.p_mod > 0:             # modality dropout
                keep = torch.rand(f_tab.shape[:-1] + (1,), device=f_tab.device) >= self.p_mod
                f_tab = f_tab * keep.to(f_tab.dtype)
            gamma, beta = self.tab(f_tab).chunk(2, dim=-1)
            h = h * (1 + gamma) + beta
        return self.cls(h + self.post(h))


def build_head(cfg, head=None):
    m = cfg["model"]
    return FusionHead(len(cfg["features"]["encoders"]), head or m["head"], d=m["d"], p_drop=m["p_drop"],
                      p_modality_drop=m["p_modality_drop"])


class EMA:
    """Exponential moving average of model weights (decay 0 disables). Evaluating EMA weights is post-processing."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval() if decay > 0 else None

    @torch.no_grad()
    def update(self, model):
        if self.model is None:
            return
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - self.decay)

    def target(self, model):
        return self.model if self.model is not None else model
