# 10 — Results Reporting

## 10.1 Tables

**T1 — Master benchmark (replaces slide 18)**

| Exp | Setup | Modality | ε (δ=1e-5) | TEST mean-5 AUROC [95% CI] | VAL mean-14 AUROC | Δ vs previous |
|---|---|---|---|---|---|---|
| E0 | LogReg on features | img | ∞ | | | — |
| E1 | Centralized | img | ∞ | | | |
| E2 | Centralized | img+tab | ∞ | | | |
| E3 | FedAvg K=5 | img+tab | ∞ | | | |
| E4 | DP-FedAvg (patient) | img+tab | 8 | | | |
| E4 | DP-FedAvg (patient) | img+tab | 3 | | | |
| E4 | DP-FedAvg (patient) | img+tab | 1 | | | |

Values as mean ± std over 3 seeds; CI from bootstrap on seed 0 (or pooled).

**T2 — Per-label TEST AUROC** (5 competition labels × E1–E4).

**T3 — Client partition** (patients, images, p_k, prevalence) — replaces slide 15 with no tiny clients.

**T4 — Per-client audit** (E3 vs E4@ε=8: AUROC, Δ, ε_k). With per-client calibration ε_k is equal — contrast with v1's 10.7 vs 989.8.

**T5 — Fairness** (AUROC by sex / age bin / AP-PA; gap; non-DP vs DP).

**T6 — Image-level vs patient-level DP** (nominal ε, implied worst-case patient ε, AUROC).

**T7 — Communication** (params synced, MB/round fp32/fp16, total over R rounds; v1 vs v2).

## 10.2 Figures (`scripts/aggregate.py`)

1. Privacy–utility curve: x = ε (log, plus ∞ marker), y = TEST mean-5 AUROC with error bars.
2. FL convergence: VAL AUROC vs round for E3 and E4 (ε 8/3/1), with cumulative ε on a secondary axis.
3. Per-label grouped bars (E1, E2, E3, E4@8).
4. ROC curves on TEST for the 5 labels (best DP model).
5. Per-client grouped bars (E3 vs E4).
6. Fairness bars by subgroup.
7. Grad-CAM 2×4 grid.
8. (Optional) MIA ROC: non-DP vs DP.

## 10.3 Mapping to the review deck

| Slide | Change |
|---|---|
| 6 Problem | Keep gaps; replace "clinical indication text" with "prospective clinical metadata (text pluggable)" |
| 7–11 Design | Replace with `02` diagram: frozen pre-deployed encoder + federated DP head |
| 13 Decisions | Frozen CXR encoder, LayerNorm head, FiLM fusion, per-client σ calibration, equal local steps |
| 14–15 Data | Official valid as TEST; partition with min client size |
| 17 Output | Same chart but with tuned thresholds and Grad-CAM next to it |
| 18–22 Results | T1, Fig 1–2, T4, Fig 5–6 |
| 23 Comms | T7 |
| 24 Conclusion | Claims backed by the new numbers; "privacy tax" now presented as identified in v1 and fixed in v2 |

## 10.4 Honest-claims checklist

- Don't call ε > ~10 "strong privacy"; ε ≤ 8 is the commonly cited range for meaningful DP-ML.
- Don't claim encryption/secure aggregation unless implemented.
- State tuning-on-VAL as a limitation.
- TEST has 234 images — CIs are wide (often ±0.03–0.05); avoid over-interpreting small Δs.
