# 07 — Training and Evaluation

## 7.1 Experiment matrix

| ID | Setup | Modality | Privacy | Purpose | Priority |
|---|---|---|---|---|---|
| E0 | Logistic regression on cached features (sklearn, per label) | image | none | **Sanity floor** — must already be well above 0.7 on test mean-5 | must |
| E1 | Centralized head | image only | none | Vision baseline | must |
| E2 | Centralized head | image + tab (FiLM) | none | Value of the 2nd modality | must |
| E3 | FedAvg, K=5, α=0.5 | image + tab | none | Cost of federation | must |
| E4 | DP-FedAvg, patient-level | image + tab | ε = 8, 3, 1 | Privacy–utility curve | must |
| E5 | DP-FedAvg, image-level vs patient-level @ ε=8 | image + tab | ε = 8 | Value/cost of patient-level DP | should |
| E6 | Heterogeneity α ∈ {0.1, 10} for E3 & E4(ε=8) | image + tab | none / 8 | Robustness to non-IID | nice |
| E7 | Per-client + fairness audit of E3 vs E4(ε=8) | — | — | Institutional & subgroup disparity | must |
| E8 | Communication cost | — | — | Bandwidth | must (calc only) |
| E9 | Membership-inference audit | — | — | Empirical privacy | nice |

Seeds: 3 seeds (0, 1, 2) for E1–E4; report mean ± std. Runs are minutes each, so this is affordable.

Concat-vs-FiLM fusion ablation can ride along with E2 (one extra run).

## 7.2 Metrics

**Primary:** mean AUROC over the competition 5 on TEST (official valid), with 95% CI from 1,000 patient-level bootstrap resamples.

**Secondary:**
- Per-label AUROC (TEST) for the 5; per-label AUROC on VAL for all 14 (labels with < 20 positives in TEST are not reported on TEST — Fracture, Lung Lesion, Pleural Other, etc. have 0–1 positives there).
- AUPRC per label.
- F1 / sensitivity / specificity at VAL-tuned thresholds.
- ECE (calibration).
- ε (max over clients), δ.

Rule: compute AUROC per label only when both classes exist; never feed -1/NaN.

## 7.3 Fairness audit (E7)

On VAL (larger than TEST) for the chosen models:
- AUROC by sex (M/F), age bin (<40, 40–60, 60–80, ≥80), projection (AP/PA).
- Report max–min gap per attribute and compare non-DP vs DP. DP-SGD is known to hit under-represented groups harder; whichever way your numbers fall, this is a legitimate finding.
- Per-client: global model AUROC on each client's local val; compare E3 vs E4.

## 7.4 Explainability

1. **Grad-CAM** on the chained model (frozen DenseNet → head) for the target label, using `features.denseblock4` (or `features.norm5`) activations. Produce a 2×4 grid: TEST images with the predicted label, probability and heatmap overlay, including one false positive.
2. **Modality contribution:** TEST AUROC with `f_tab` zeroed vs real (ablation), plus mean |γ|, |β| of the FiLM layer.
3. **Tabular attribution** (optional): Integrated Gradients via Captum over the 9 metadata features.

## 7.5 Outputs per run

`results/<exp_id>/<seed>/`:
- `config.yaml` (resolved), `metrics_val.json`, `metrics_test.json`, `per_label.csv`, `history.csv` (round/epoch, loss, val AUROC, ε_k), `head.pt`, `thresholds.json`.
- One aggregator script builds `results/summary.csv` and all figures (see `10`).
