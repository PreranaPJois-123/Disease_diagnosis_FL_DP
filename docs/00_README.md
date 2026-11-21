# DP-MMFL v2 — Design Docs

Differentially Private, Multimodal Federated Learning for chest X-ray diagnosis, rebuilt on **CheXpert-v1.0-small** for a 1.5-day turnaround.

## Read in this order

| # | File | What it settles |
|---|------|-----------------|
| 01 | `01_diagnosis_of_v1.md` | Why v1 scored AUROC ≈ 0.46–0.54 (it is a bug, not the architecture) |
| 02 | `02_system_architecture.md` | End-to-end pipeline and the key design decisions |
| 03 | `03_data_and_preprocessing.md` | CSV parsing, labels, uncertainty policy, splits, non-IID partition, feature caching |
| 04 | `04_model_architecture.md` | Frozen CXR encoder + tabular encoder + FiLM fusion head |
| 05 | `05_federated_learning.md` | Clients, FedAvg / server momentum, rounds, communication |
| 06 | `06_differential_privacy.md` | Patient-level DP-SGD, per-client noise calibration, RDP accounting |
| 07 | `07_training_and_evaluation.md` | Experiments E0–E8, metrics, fairness, explainability |
| 08 | `08_implementation_plan.md` | Repo layout, configs, commands, hardware, hour-by-hour timeline |
| 09 | `09_sanity_checks_and_debugging.md` | Checks that must pass before any result goes on a slide |
| 10 | `10_results_reporting.md` | Tables/figures to produce and how they map to the review slides |

## The five decisions that change everything

1. **Fix the bug first.** Macro AUROC below 0.5 with train loss → 0 means images and labels are misaligned or evaluation is wrong. A correct ImageNet ResNet-18 on CheXpert lands far above 0.54. See `01`.
2. **Freeze a chest-X-ray-pretrained encoder and cache its features once.** Every later experiment (centralized, FL, DP, ablations, seeds) then trains only a ~0.3M-parameter head in minutes. This is what makes 1.5 days feasible.
3. **Small trainable head → DP actually works.** DP noise scales with parameter count; a 0.3M head at ε = 8 (or even 1–3) keeps most utility. v1's ε ≈ 990 is not a meaningful guarantee.
4. **Second modality = structured clinical metadata available at scan time** (age, sex, view/projection, prior-study count). CheXpert-small has no report text. It is prospective and leakage-free, which keeps the "zero diagnostic leakage" story. Text plugs in later as a third branch.
5. **Calibrate noise per client to a common target ε.** This directly fixes the v1 "high-volume centre privacy tax" (it was the future-work item on slide 24).

## Target outcome (what "good" looks like)

- Official CheXpert valid set (234 radiologist-labelled images) used **only** as the final test set.
- Headline metric: mean AUROC over the 5 competition labels (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion) with bootstrap 95% CI.
- Privacy–utility curve at ε ∈ {1, 3, 8, ∞}, δ = 1e-5, patient-level.
- Per-client and per-subgroup (sex, age, AP/PA) audit.

Expected ballpark for a frozen CXR encoder + head is in the high 0.7s to mid 0.8s on the 5 competition labels; treat any number you get as the result, but anything under ~0.70 for the non-private centralized run means go back to `09`.
