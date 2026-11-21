# 02 — System Architecture (v2)

## Overview

```mermaid
flowchart LR
  subgraph Offline["Stage A — one-time, per hospital (local)"]
    CSV[train.csv / valid.csv] --> P[Parse & clean\nlabels, metadata, patient_id]
    IMG[Frontal JPEGs] --> FE[Frozen CXR encoder\nDenseNet-121 (NIH-pretrained)\n→ 1024-d feature]
    P --> TAB[Tabular vector\nage, sex, view, prior studies]
    FE --> CACHE[(Feature cache\n.npy float16)]
    TAB --> CACHE
  end

  subgraph Client["Stage B — each client k, each round"]
    CACHE --> H[Trainable fusion head\nFiLM(image | tabular) → 14 logits]
    H --> DPSGD[Patient-level DP-SGD\ngroup → clip C → Gaussian noise σ_k]
    DPSGD --> UPD[Δw_k (head only, ~0.3M params)]
  end

  subgraph Server["Stage C — server"]
    UPD --> AGG[Weighted FedAvg\n(+ optional server momentum)]
    AGG --> ACC[RDP accountant per client\nreport ε_k, max ε]
    AGG --> G[Global head w_t+1]
    G --> H
  end

  G --> EVAL[Stage D — evaluation\nofficial valid = test\nAUROC + CI, fairness, Grad-CAM]
```

## Stages

| Stage | Where | Runs | Output |
|---|---|---|---|
| A. Preprocess + feature extraction | GPU (3050 / Kaggle) | **once**, ~30–60 min | `features/*.npy`, `meta/*.parquet` |
| B. Local training | CPU or GPU | every round, seconds | head deltas |
| C. Aggregation + accounting | CPU | every round | global head, ε log |
| D. Evaluation + explainability | GPU for Grad-CAM, else CPU | end | tables, plots, heatmaps |

## Key design decisions and rationale

| Decision | Rationale |
|---|---|
| Frozen, domain-pretrained encoder (TorchXRayVision DenseNet-121, NIH weights) | Strong CXR features without training a CNN; NIH weights were not trained on CheXpert → no test leakage. Pre-deployed to every hospital like v1's frozen MiniLM. |
| Cache features once | All experiments (≥ 10 runs × seeds) become minutes. Essential for 1.5 days. |
| Train only a small head | DP utility loss scales with dimension; communication drops from 43 MB to ~1 MB per round. |
| LayerNorm, no BatchNorm | BatchNorm leaks cross-sample info and is incompatible with per-sample DP gradients. |
| Tabular branch (age, sex, view, prior-study count) | Real second modality available at acquisition time; prospective, leakage-free. |
| FiLM fusion (tabular modulates image features) | Cheap, more expressive than concatenation, interpretable gate. |
| Patient = privacy unit | Bounds total influence of a patient with many studies — the v1 research gap. |
| Per-client σ calibrated to common target ε | Every hospital gets the same guarantee regardless of size. |
| Official `valid.csv` as test | Standard benchmark, radiologist consensus labels, comparable to literature. |

## What is simulated vs real

- **Real:** DP-SGD with per-patient clipping, noise, RDP accounting; FedAvg over isolated client data; non-IID partitions.
- **Simulated:** clients run sequentially in one process; secure aggregation and encryption are *not* implemented cryptographically (state as "assumed / out of scope", or show as a placeholder module). Do not claim encryption you did not implement.

## Pluggable modality (for the text story)

The head accepts `x_img`, `x_tab`, and optional `x_txt` (+ `txt_present` flag). With CheXpert-small, `x_txt` is absent. If CheXpert Plus reports become available, embed only the **Indication / History** section with frozen MiniLM (384-d) and add it as a second FiLM conditioner. Findings/Impression must never be used (label leakage).
