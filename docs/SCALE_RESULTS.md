# Scale ladder: small vs medium vs large

Same TEST set (202 official CheXpert valid frontal images) and the same metric (mean AUROC over the 5 competition labels) at every scale. Mean ± sd over 3 seeds.

| scale | FL-pool / VAL patients | code | encoders | rounds × steps |
|---|---|---|---|---|
| small | 2,000 / 1,000 | v2.1 | nih | 15 × 50 |
| medium | 20,000 / 4,000 | v2.1 | nih | 15 × 50 |
| large (v2) | ~58,000 / ~6,400 (full data) | v2 (older code, in `results/v2-large/`) | nih | 30 × 50 |
| **large (v2.1)** | ~58,000 / ~6,400 (full data) | v2.1 (overnight run, 2026-11-03) | nih + mimic_ch + pc | frozen config (`results/v2.1-final/frozen.yaml`) |

large (v2.1) differs from the other scales in two ways: more data and three encoders. Its gain over medium therefore mixes both effects. The nih-only trials at full scale give VAL 0.749–0.752 against 0.763–0.770 with three encoders, so roughly 0.02 of VAL comes from the extra encoders.

| config | small VAL | medium VAL | large v2 VAL | large v2.1 VAL | small TEST | medium TEST | large v2 TEST | large v2.1 TEST |
|---|---|---|---|---|---|---|---|---|
| central, image-only | 0.708±0.002 | 0.738±0.001 | 0.746 | 0.768±0.000 | 0.796±0.007 | 0.823±0.008 | 0.818±0.001 | 0.842±0.003 |
| central, FiLM | 0.709±0.003 | 0.740±0.001 | 0.749 | 0.771±0.000 | 0.794±0.010 | 0.819±0.002 | 0.823±0.009 | 0.846±0.005 |
| FedAvg | 0.694±0.003 | 0.732±0.001 | 0.736 | 0.759±0.001 | 0.765±0.013 | 0.813±0.007 | 0.817±0.002 | 0.839±0.003 |
| DP ε=8 | 0.674±0.001 | 0.703±0.000 | 0.711 | 0.739±0.001 | 0.781±0.001 | 0.807±0.005 | 0.803±0.004 | 0.813±0.001 |
| DP ε=3 | 0.636±0.007 | 0.702±0.001 | 0.710 | 0.737±0.001 | 0.748±0.014 | 0.807±0.003 | 0.804±0.003 | 0.818±0.009 |
| DP ε=1 | 0.566±0.017 | 0.687±0.001 | 0.701 | 0.727±0.002 | 0.603±0.032 | 0.784±0.004 | 0.798±0.007 | 0.819±0.002 |
| image-level DP ε=8 | 0.691±0.002 | 0.696±0.000 | 0.711 | 0.737±0.001 | 0.795±0.004 | 0.787±0.003 | 0.804±0.004 | 0.815±0.003 |
| FedAvg int8 up / fp16 down | 0.694±0.002 | 0.733±0.001 | 0.736 | 0.759±0.001 | 0.765±0.014 | 0.812±0.007 | 0.818±0.002 | 0.842±0.002 |

DP(ε=3) VAL cost vs FedAvg: small 0.058, medium 0.031, large v2 0.025, large v2.1 0.022. VAL is the more reliable column: 1,000 to 6,400 patients, against 202 images for TEST.

Where things are stored: `results/scale_small/`, `results/scale_medium/` and `results/v2.1-final/` (runs, `master_table.md`, figures), `results/scale_comparison.{csv,md,png}`. The seed-0 E4 ε=3, E3, E2 and E1 models are saved in `models/v2.1-large/` for the dashboard. The dashboard (`streamlit run dashboard/app.py`) shows the same comparison.

Other measured results (large v2.1, VAL gaps, seeds averaged; the smaller scales agree):

- **Communication:** int8 up / fp16 down saves 62.4% of bytes, with AUROC unchanged (FedAvg 0.842 vs 0.840; DP ε=3 identical). Top-k with error feedback saves 74.2% but costs 0.016 VAL under DP.
- **Reweighting (F1):** E7 vs E4 at ε=3. VAL is identical (0.7373 vs 0.7374), and the age and projection gaps are unchanged (0.078 vs 0.078; 0.069 vs 0.069). The TEST difference (0.827 vs 0.818) is inside the CI.
- **Per-group thresholds (F2):** the TPR gap across AP/PA groups drops from 0.16–0.26 to 0.015–0.06, on held-out VAL patients.
- **Membership inference:** the loss-threshold attack gives AUC 0.50–0.52 everywhere. LiRA with reference models (`results/mia_lira/real/`) gives 0.68 on the central model (6.1% of training patients found at 1% FPR), 0.55 on FedAvg and 0.495–0.497 on the DP models.
- **Privacy accounting (PRV / RDP):** ε=1 runs spent 0.91 / 0.99 (early stopping), ε=3 spent 3.00 / 3.31, ε=8 spent 8.00 / 9.04. Image-level ε=8 spent 8.00 / 9.51 per image.

Caveats:

- TEST has 202 images (95% CI about ±0.03). Differences below that are not resolved, including the ordering of ε=1, 3 and 8.
- The model trials, the DP sweep and early stopping use VAL, and that budget is not accounted for.
