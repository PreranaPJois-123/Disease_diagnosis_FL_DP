# 09 — Sanity Checks (gate before any result is reported)

`scripts/03_sanity.py` runs all of these and exits non-zero on failure.

## 9.1 Data checks

| # | Check | Pass condition |
|---|---|---|
| D1 | Path ↔ label alignment | For 10 random rows, re-read the CSV row by `row_id` and compare labels with the cache meta: identical |
| D2 | Patient IDs parsed from `Path` only | `patient_id` regex matches 100% of rows |
| D3 | Split overlap | 0 shared patients across TEST/VAL/each client |
| D4 | Label values in targets | only {0, 1}; masks only {0, 1}; no NaN |
| D5 | Label column order | `LABELS` constant equals CSV header order used in parse |
| D6 | Prevalence | per-label positive rate printed for train/VAL/TEST; nothing at 0% or 100% unexpectedly |
| D7 | Feature cache | shape matches meta; no NaN/Inf; spot re-extraction diff < 1e-2 |

## 9.2 Learning checks

| # | Check | Pass condition |
|---|---|---|
| L1 | E0 logistic regression on features | TEST mean-5 AUROC well above 0.7 |
| L2 | **Shuffled-label control** — train E1 with labels permuted across patients | VAL AUROC ≈ 0.50 (±0.02). If real labels also give ~0.5, the pipeline is broken |
| L3 | Overfit a tiny batch (64 samples, 300 steps) | train loss → near 0 |
| L4 | Train AUROC vs VAL AUROC after 1 epoch | both > 0.7 and within ~0.05 |
| L5 | Any per-label AUROC < 0.45 on VAL | investigate — likely label flip or mask error |

## 9.3 DP checks (unit tests)

- With σ = 0 and C = ∞, DP-SGD gradient equals the ordinary mean-of-patient-means gradient (allclose).
- After clipping, every per-patient grad norm ≤ C (+1e-5).
- Accountant: ε increases monotonically with steps; calibrated σ yields ε within 1% of target at R×S steps.
- Per-client ε at end of training are equal (±1%) under per-client calibration.

## 9.4 Debugging playbook (if AUROC ≈ 0.5 again)

1. Run L2. If real ≈ shuffled → alignment bug: print `Path` + labels for 5 samples, open the images, eyeball (Support Devices = lines/wires visible).
2. Check `roc_auc_score(y_true, y_score)` argument order and that `y_true` is the TEST labels, not training targets.
3. Confirm the `features` used for TEST came from `feat_test.npy`, in `meta_test` order.
4. Confirm encoder weights loaded (print a weight checksum; features should differ across images with std > 0.1).
5. Confirm sigmoid is applied for scores (AUROC is invariant to monotone transforms, so this matters only for thresholds/ECE).
