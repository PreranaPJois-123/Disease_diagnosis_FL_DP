# 04 — Model Architecture

## 4.1 Full inference model

```
X-ray (1×224×224) ─► Frozen DenseNet-121 (NIH)  ─► f_img ∈ R^1024   (pre-deployed, never transmitted)
metadata (9)      ───────────────────────────────► f_tab ∈ R^9
                                    │
                         Trainable FusionHead (federated, DP)
                                    │
                               14 logits → sigmoid
```

During training only `FusionHead` runs (on cached `f_img`). At inference / Grad-CAM, encoder + head are chained.

## 4.2 FusionHead (default: FiLM)

```python
class FusionHead(nn.Module):
    def __init__(self, d_img=1024, d_tab=9, d=256, n_labels=14, p_drop=0.2, use_tab=True):
        self.img = nn.Sequential(nn.LayerNorm(d_img), nn.Linear(d_img, d), nn.GELU(), nn.Dropout(p_drop))
        self.use_tab = use_tab
        if use_tab:
            self.tab = nn.Sequential(nn.Linear(d_tab, 64), nn.GELU(), nn.Linear(64, 2 * d))
            nn.init.zeros_(self.tab[-1].weight); nn.init.zeros_(self.tab[-1].bias)   # starts as identity
        self.post = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p_drop))
        self.cls = nn.Linear(d, n_labels)

    def forward(self, f_img, f_tab=None):
        h = self.img(f_img)
        if self.use_tab:
            gamma, beta = self.tab(f_tab).chunk(2, dim=-1)
            h = h * (1 + gamma) + beta            # FiLM
        h = h + self.post(h)                      # residual
        return self.cls(h)
```

Parameters ≈ 1024·256 + 256·256 + 9·64 + 64·512 + 256·14 ≈ **0.36M**.
Zero-init of the FiLM generator makes the multimodal model start exactly as the image-only model — a clean ablation.

Variants (config `fusion: film | concat | image_only`):
- `image_only`: `use_tab=False` (E1 baseline)
- `concat`: `cls(MLP([h_img ; MLP(f_tab)]))`

**Modality dropout** (training only, p = 0.1): zero `f_tab` occasionally so the model stays robust if metadata is missing at a site.

## 4.3 Loss

Masked BCE-with-logits with per-label `pos_weight`:
```
pos_weight_j = clip(neg_j / pos_j, 1, 10)      computed on the FL-pool train split
L = Σ_j mask_ij · BCE(z_ij, y_ij; pos_weight_j) / max(1, Σ_j mask_ij)
```
(Clients can compute pos_weight locally; for simplicity use global train statistics and state it. Under strict FL, use local ones.)

## 4.4 Thresholds and calibration

Probabilities are **not** thresholded at 0.5. For each label pick the threshold maximising Youden's J (or F1) on VAL, store in `thresholds.json`, apply to TEST. Report ECE before/after optional temperature scaling on VAL.

## 4.5 Optional extension (only if time remains)

Partial fine-tuning of `denseblock4` (non-private centralized only) as an upper-bound reference. Skip under DP.
