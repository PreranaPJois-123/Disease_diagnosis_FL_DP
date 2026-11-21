# 01 — Diagnosis of v1 results

## Symptoms (from Review-2 slides)

| Evidence | Value | What it means |
|---|---|---|
| E1 centralized ResNet-18 macro AUROC | 0.540 (peak, epoch 7) | Barely above coin-flip |
| E2–E4 macro AUROC | 0.487 / 0.477 / 0.465 | **Below 0.5** — worse than random |
| E1 train loss | → ~0.01 by epoch 20 | Model memorises training data perfectly |
| E1 val loss | rises from epoch 3 | Nothing generalises from the very start |
| Client 3 AUROC | 0.44 | Systematically inverted predictions |
| Sample inference | 9/13 labels > 0.5 on one image | Outputs are not calibrated / not informative |
| E4 privacy | ε = 989.8 | No practical privacy guarantee |

A correctly wired ImageNet-pretrained ResNet-18 on CheXpert typically reaches macro AUROC in the 0.75–0.85 range within a couple of epochs. **Perfect train fit + chance-level val = the model is learning noise.** The architecture is not the problem.

## Most likely root causes (check in this order)

1. **Image ↔ label misalignment.** E.g. building the image list with `glob`/`os.listdir` (sorted differently) and zipping it with CSV rows; shuffling one list but not the other; resetting a DataFrame index after filtering only one side; mixing frontal/lateral rows after filtering. This alone explains every symptom.
2. **Uncertain labels (-1) or blanks (NaN) leaking into evaluation.** Passing `-1` to `roc_auc_score` as a class, or computing AUROC with NaNs filled as 1, distorts and can invert AUROC.
3. **Label column order mismatch** between the dataset and the metric/plot (e.g. the CSV order vs. a hard-coded list).
4. **Wrong normalisation / channel handling.** Grayscale fed as 1 channel into ImageNet weights without replication, or pixel range 0–255 without normalisation.
5. **Pretrained weights not actually loaded** (`weights=None`) + GroupNorm replacing BatchNorm discards the pretrained BN statistics — swapping BN→GN on a pretrained ResNet effectively re-initialises the normalisation layers.
6. **Masked loss with an inverted mask** (training only on the masked-out entries).
7. **AUROC averaged over labels with a single class present** in val (undefined → 0.5 or errors silently replaced).

## Secondary design issues

- **Text modality had no signal.** CheXpert-small has no indication text; if MiniLM received empty/constant strings, the "multimodal" branch was a constant vector. That matches E2 being *worse* than E1.
- **Training the full ResNet-18 (11M params) under DP-SGD** needs huge σ or huge ε. Hence ε ≈ 990.
- **Fixed σ with unequal local step counts** → the hub (1,510 steps) burns most of the budget. Calibrating σ per client to a target ε removes this.
- **Extreme partition** (clients with 34 and 62 patients) makes per-client AUROC on those clients statistically meaningless.
- **No proper test set.** The official `valid.csv` (radiologist consensus labels, no uncertainty) is the standard benchmark and should be the held-out test.
- **pos_weight up to 5 + threshold 0.5** inflates probabilities → most labels > 0.5 on the sample image.

## What to say in the next review

"v1 contained a data-pipeline defect that produced chance-level AUROC. v2 fixes it, adds sanity checks that would have caught it, and redesigns the system so DP is applied to a compact head with a meaningful ε."  Being upfront about this is stronger than defending 0.46.
