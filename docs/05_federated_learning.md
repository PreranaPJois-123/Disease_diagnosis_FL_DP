# 05 — Federated Learning

## 5.1 Setup

- K = 5 simulated hospitals from `partition.json`; each client sees **only** its own rows of the feature cache.
- Global model = `FusionHead` only. The frozen encoder is identical everywhere and never sent.
- Clients run sequentially in one process (simulation). Every client gets a deep copy of the global state at round start.

## 5.2 Round protocol

```
for t in 1..R:
    for k in clients (all participate; optional fraction q_c):
        w_k ← w_t
        run S local steps (non-DP: Adam/SGD; DP: patient-level DP-SGD, see 06)
        Δ_k = w_k − w_t
    Δ = Σ_k p_k Δ_k ,   p_k = n_patients_k / Σ n_patients
    server step:  FedAvg       w_{t+1} = w_t + Δ
                  FedAvgM      m = β m + Δ ;  w_{t+1} = w_t + η_s m     (β = 0.9, η_s = 1.0)
    evaluate global model on VAL every round; log AUROC, loss, ε_k
keep the round with best VAL mean-5 AUROC → evaluate on TEST once
```

**Equal local work:** every client does the same number of local steps S per round (not one epoch each). This prevents the hub from doing 1,510 steps while the rural site does 10 — the root of v1's skewed ε and "privacy tax".

## 5.3 Default hyperparameters

| Param | Non-DP | DP |
|---|---|---|
| Rounds R | 30 | 30 |
| Local steps S | 50 | 50 |
| Batch | 256 images | expected 256 patients (Poisson) |
| Optimizer (client) | AdamW lr 1e-3, wd 1e-4 | SGD lr ∈ {0.5, 1, 2}, momentum 0 (reset per round) |
| Server | FedAvg | FedAvgM β = 0.9 |

Small sweep on VAL only. With cached features one full FL run takes a few minutes on CPU.

## 5.4 Optional (only if time)

- **FedProx** (μ = 0.01) for the α = 0.1 heterogeneity case.
- **Local fine-tuning / FedPer**: after FL, fine-tune `cls` on each client's local data and report per-client gains (non-DP only, or DP with its own budget).

## 5.5 Communication accounting (E8)

Per round per client: `n_params × 4 bytes` (fp32) up + down. With 0.36M params ≈ **1.4 MB fp32 / 0.7 MB fp16** vs v1's 43.6 MB. Report total = 2 × K × R × payload. The frozen encoder (~7M params) is a one-time pre-deployment, not per-round traffic.
