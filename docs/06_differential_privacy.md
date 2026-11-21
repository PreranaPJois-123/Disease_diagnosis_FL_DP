# 06 — Differential Privacy (patient-level)

## 6.1 Threat model and guarantee

- **Unit of privacy: one patient** (all of their studies/images). Adjacent datasets differ by adding/removing one patient's entire record.
- Guarantee: each client's sequence of released updates is (ε_k, δ)-DP w.r.t. its own patients. Since every patient lives at exactly one client, the overall guarantee for any patient is (max_k ε_k, δ). Report `max ε` as the headline.
- δ = 1e-5 (< 1 / n_patients for every client with the min-size constraint).
- Server is honest-but-curious; DP protects against reconstruction from updates (DLG) and membership inference on the released model. Secure aggregation is simulated, not relied upon for the DP claim.

## 6.2 Algorithm — patient-level DP-SGD (per client)

```
P_k = patients at client k, q = B / |P_k|         (B = expected patients per batch)
σ_k = calibrate(target_ε, δ, q, total_steps = R × S)     # once, before training
for each local step:
    sample batch: include each patient independently with prob q   (Poisson)
    for each sampled patient p:
        I_p = their images (cap at M = 8, random subset if more)
        g_p = ∇_w [ (1/|I_p|) Σ_{i∈I_p} L_i ]             # one gradient per patient
        ĝ_p = g_p · min(1, C / ||g_p||₂)                   # clip, C = 1.0
    g̃ = ( Σ_p ĝ_p + N(0, σ_k² C² I) ) / B                 # divide by EXPECTED B, not actual
    w ← w − η g̃
    accountant_k.step(noise_multiplier = σ_k, sample_rate = q)
```

Why the cap M: bounds compute and keeps padding small; privacy holds regardless because clipping bounds each patient's total contribution.

## 6.3 Implementation (no Opacus wrapper needed)

Use `torch.func` for per-patient gradients on the small head — fast and exact.

```python
from torch.func import functional_call, vmap, grad

params = {k: v.detach() for k, v in model.named_parameters()}
def patient_loss(params, f_img, f_tab, y, m, img_mask):     # shapes [M,1024],[M,9],[M,14],[M,14],[M]
    logits = functional_call(model, params, (f_img, f_tab))
    l = masked_bce(logits, y, m, reduction="none_per_image")  # [M]
    return (l * img_mask).sum() / img_mask.sum().clamp(min=1)

per_patient_grads = vmap(grad(patient_loss), in_dims=(None, 0, 0, 0, 0, 0))(params, F, T, Y, Mk, IM)
# per_patient_grads[name] : [B_actual, *param.shape]
norms = torch.sqrt(sum(g.flatten(1).pow(2).sum(1) for g in per_patient_grads.values()))
factor = (C / (norms + 1e-6)).clamp(max=1.0)
for name, p in model.named_parameters():
    g = (per_patient_grads[name] * factor.view(-1, *[1]*(p.dim()))).sum(0)
    g = g + torch.randn_like(g) * sigma * C
    p.grad = g / B_expected
optimizer.step()
```

- Batches are padded to `[B_actual, M, ·]` with `img_mask`. Group the client's rows by `patient_id` once and keep index lists.
- Dropout in DP: fine (it's per-sample randomness). Model must have **no BatchNorm** (it doesn't).
- Memory: B=256 × 0.36M params × 4 B ≈ 370 MB for per-patient grads — OK on GPU or CPU. Reduce B or chunk the vmap if needed.

## 6.4 Accounting and calibration

Use Opacus' accountant utilities only (no `PrivacyEngine`):
```python
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
sigma_k = get_noise_multiplier(target_epsilon=eps, target_delta=1e-5,
                               sample_rate=q_k, steps=R*S, accountant="rdp")
acc_k = RDPAccountant();  ...  acc_k.step(noise_multiplier=sigma_k, sample_rate=q_k)
eps_k = acc_k.get_epsilon(delta=1e-5)
```
Because all clients run the same R×S steps but have different q_k, calibrating σ_k per client gives **every client the same ε** (the v1 "privacy tax" fix). Log ε_k after every round for the budget-vs-round plot.

Privacy targets for the sweep: **ε ∈ {1, 3, 8}** plus non-private (ε = ∞).

Hyperparameter tuning on VAL technically consumes privacy; state it as a standard limitation (as in most DP-ML papers) and keep the sweep small.

## 6.5 Ablation: image-level vs patient-level DP (E5)

- Image-level DP: treat each image as the unit (per-image clipping, q = B / n_images). Same nominal ε.
- Its *patient-level* guarantee degrades with the number of images per patient (group privacy: roughly k·ε for a patient with k images, worse for δ). Report the distribution of images per patient and the implied worst-case patient ε for the image-level run.
- Show utility of both at the same nominal ε → the cost of honest patient-level protection.

## 6.6 Optional empirical audit (if time)

Simple loss-threshold membership-inference attack on the final head: AUC of distinguishing train patients vs VAL patients from per-patient loss. Expect ≈ 0.5 under DP, higher without. One plot, strong slide.
