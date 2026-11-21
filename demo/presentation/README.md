# Demo presentation set

X-rays grouped by what they demonstrate. Each table gives the inputs to type on the dashboard's **Predict** tab and what the models output. Start the dashboard with `streamlit run dashboard/app.py`, choose **Trained on: saved (v2.1-large)** and the model, upload the image and enter age, sex and view exactly as listed.

- **Private** = `E4_eps3` (patient-level DP, eps = 3). **Central** = `E2` (no privacy). **FedAvg** = `E3` (federated, no DP).
- The model columns give the calibrated probability for each finding in the radiology report: ✓ = at or above the model's decision threshold (fitted on validation patients), ✗ = below it. *False alarm* = the strongest finding not in the report with probability of at least 0.5.
- Rare findings (fracture, lung lesion, pleural other, enlarged cardiomediastinum) have very low thresholds, so the dashboard may flag them at small probabilities. Focus the demo on the report findings and their probabilities.
- Groups 1-5 contain **selected** cases that work; they overstate accuracy. Group 6 shows typical misses. All patients except the `member_*` images in group 5 are held-out validation patients the models never trained on. No TEST image is used.
- Research prototype, not for clinical use.

## 1. Normal vs abnormal

Start here. Show a normal chest (high *No Finding*), then abnormal chests where the private model flags real findings.

| image | age | sex | view | report findings | private model | private false alarm | central model | central false alarm |
|---|---|---|---|---|---|---|---|---|
| `1_normal_vs_abnormal/01_normal_patient22025.jpg` | 45 | Male | PA | No Finding | No Finding 0.98 ✓ | none | No Finding 0.93 ✓ | none |
| `1_normal_vs_abnormal/02_normal_patient50868.jpg` | 40 | Male | AP | No Finding | No Finding 0.66 ✓ | none | No Finding 0.89 ✓ | none |
| `1_normal_vs_abnormal/03_abnormal_patient44148.jpg` | 59 | Male | AP | Lung Opacity, Atelectasis (uncertain: Consolidation) | Lung Opacity 0.51 ✗, Atelectasis 0.65 ✓ | none | Lung Opacity 0.59 ✓, Atelectasis 0.75 ✓ | Pleural Effusion 0.55 |
| `1_normal_vs_abnormal/04_abnormal_patient20739.jpg` | 86 | Male | PA | Cardiomegaly, Lung Opacity, Atelectasis (uncertain: Pneumonia) | Cardiomegaly 0.59 ✓, Lung Opacity 0.45 ✗, Atelectasis 0.50 ✗ | none | Cardiomegaly 0.95 ✓, Lung Opacity 0.45 ✗, Atelectasis 0.40 ✗ | Edema 0.65 (+1) |

## 2. One finding at a time

One image per finding that the private model detects reliably (VAL AUROC at least 0.70). Pick the finding, open the Grad-CAM for it and point at the highlighted region.

| image | age | sex | view | report findings | private model | private false alarm | central model | central false alarm |
|---|---|---|---|---|---|---|---|---|
| `2_one_finding_each/01_Cardiomegaly_patient22079.jpg` | 44 | Male | AP | Cardiomegaly | Cardiomegaly 0.60 ✓ | none | Cardiomegaly 0.84 ✓ | Edema 0.60 (+1) |
| `2_one_finding_each/02_Edema_patient10287.jpg` | 81 | Female | AP | Edema, Atelectasis (uncertain: Cardiomegaly) | Edema 0.63 ✓, Atelectasis 0.54 ✓ | none | Edema 0.68 ✓, Atelectasis 0.55 ✓ | Enlarged Cardiomediastinum 0.54 |
| `2_one_finding_each/03_Pneumothorax_patient15702.jpg` | 46 | Female | AP | Pneumothorax | Pneumothorax 0.37 ✓ | none | Pneumothorax 0.87 ✓ | Pleural Effusion 0.65 (+2) |
| `2_one_finding_each/04_PlEffusion_patient45460.jpg` | 68 | Male | AP | Lung Opacity, Pleural Effusion (uncertain: Pneumonia, Atelectasis) | Lung Opacity 0.57 ✓, Pleural Effusion 0.54 ✓ | none | Lung Opacity 0.56 ✓, Pleural Effusion 0.78 ✓ | Pneumothorax 0.54 |

## 3. Several findings in one X-ray

Multi-label classification: the model scores all 14 findings at once, and several are detected together.

| image | age | sex | view | report findings | private model | private false alarm | central model | central false alarm |
|---|---|---|---|---|---|---|---|---|
| `3_multiple_findings/01_2findings_patient13627.jpg` | 80 | Female | PA | Lung Opacity, Pleural Effusion, Pleural Other (uncertain: Consolidation, Atelectasis) | Lung Opacity 0.62 ✓, Pleural Effusion 0.82 ✓, Pleural Other 0.01 ✗ | none | Lung Opacity 0.53 ✓, Pleural Effusion 0.91 ✓, Pleural Other 0.04 ✓ | none |
| `3_multiple_findings/02_2findings_patient07161.jpg` | 64 | Male | PA | Lung Opacity, Lung Lesion (uncertain: Atelectasis) | Lung Opacity 0.51 ✓, Lung Lesion 0.41 ✓ | none | Lung Opacity 0.52 ✓, Lung Lesion 0.73 ✓ | Pneumonia 0.59 (+1) |

## 4. Private vs non-private model

Switch the model between *Central* and *E4 eps=3* on the same image. `both_detect`: privacy costs nothing here. `only_central`: the cost of privacy, which is largest for subtle and rare findings.

| image | age | sex | view | report findings | private model | private false alarm | central model | central false alarm |
|---|---|---|---|---|---|---|---|---|
| `4_private_vs_nonprivate/01_both_detect_patient17137.jpg` | 76 | Male | AP | Cardiomegaly, Support Devices | Cardiomegaly 0.77 ✓, Support Devices 0.38 ✗ | Atelectasis 0.53 | Cardiomegaly 0.79 ✓, Support Devices 0.23 ✗ | Enlarged Cardiomediastinum 0.56 |
| `4_private_vs_nonprivate/02_both_detect_patient37486.jpg` | 22 | Male | AP | Atelectasis, Support Devices | Atelectasis 0.63 ✓, Support Devices 0.49 ✓ | No Finding 0.51 | Atelectasis 0.68 ✓, Support Devices 0.83 ✓ | No Finding 0.77 |
| `4_private_vs_nonprivate/03_only_central_patient01685.jpg` | 56 | Female | PA | Atelectasis | Atelectasis 0.38 ✗ | No Finding 0.58 | Atelectasis 0.61 ✓ | No Finding 0.63 |
| `4_private_vs_nonprivate/04_only_central_patient34627.jpg` | 61 | Male | AP | Cardiomegaly | Cardiomegaly 0.12 ✗ | Support Devices 0.59 (+1) | Cardiomegaly 0.83 ✓ | Support Devices 0.70 (+1) |

## 5. Membership privacy

Pairs of a training-population patient and a held-out patient with the same finding. Use them to explain what membership inference asks (was this patient in training?), then run the live attack on the Privacy tab: the central model leaks measurably, the private model stays at chance. A single image pair does not show leakage by itself.

| image | age | sex | view | report findings | private model | private false alarm | central model | central false alarm |
|---|---|---|---|---|---|---|---|---|
| `5_membership_privacy/01_member_NoFinding_patient11086.jpg` | 24 | Female | AP | No Finding | No Finding 0.98 ✓ | none | No Finding 0.93 ✓ | Pneumothorax 0.59 |
| `5_membership_privacy/02_heldout_NoFinding_patient05041.jpg` | 84 | Female | AP | No Finding | No Finding 0.19 ✗ | Atelectasis 0.51 | No Finding 0.44 ✗ | Fracture 0.65 (+1) |
| `5_membership_privacy/03_member_EnlCardiomed_patient13396.jpg` | 54 | Female | AP | Enlarged Cardiomediastinum, Lung Opacity, Edema (uncertain: Pneumonia) | Enlarged Cardiomediastinum 0.10 ✓, Lung Opacity 0.64 ✓, Edema 0.83 ✓ | Support Devices 0.61 (+1) | Enlarged Cardiomediastinum 0.31 ✓, Lung Opacity 0.67 ✓, Edema 0.91 ✓ | Support Devices 0.61 (+1) |
| `5_membership_privacy/04_heldout_EnlCardiomed_patient34448.jpg` | 61 | Male | AP | Enlarged Cardiomediastinum, Fracture (uncertain: Pneumothorax) | Enlarged Cardiomediastinum 0.12 ✓, Fracture 0.03 ✗ | Support Devices 0.54 (+2) | Enlarged Cardiomediastinum 0.14 ✗, Fracture 0.15 ✓ | Atelectasis 0.60 (+3) |
| `5_membership_privacy/05_member_Cardiomegaly_patient14360.jpg` | 88 | Female | PA | Cardiomegaly, Pleural Effusion | Cardiomegaly 0.45 ✓, Pleural Effusion 0.67 ✓ | Atelectasis 0.63 (+2) | Cardiomegaly 0.74 ✓, Pleural Effusion 0.43 ✗ | Atelectasis 0.64 (+4) |
| `5_membership_privacy/06_heldout_Cardiomegaly_patient57844.jpg` | 68 | Male | AP | Cardiomegaly, Lung Opacity, Pleural Effusion (uncertain: Consolidation, Pneumonia, Atelectasis) | Cardiomegaly 0.29 ✓, Lung Opacity 0.64 ✓, Pleural Effusion 0.60 ✓ | Edema 0.56 | Cardiomegaly 0.72 ✓, Lung Opacity 0.67 ✓, Pleural Effusion 0.67 ✓ | Edema 0.53 |

## 6. Honest misses

Randomly chosen patients whose finding the private model misses. Show these to make clear the model is a research prototype and not a diagnostic tool.

| image | age | sex | view | report findings | private model | private false alarm | central model | central false alarm |
|---|---|---|---|---|---|---|---|---|
| `6_honest_misses/01_missed_Pneumothorax_patient04672.jpg` | 48 | Female | AP | Pneumothorax, Fracture | Pneumothorax 0.00 ✗, Fracture 0.02 ✗ | No Finding 0.91 | Pneumothorax 0.40 ✗, Fracture 0.45 ✓ | No Finding 0.91 |
| `6_honest_misses/02_missed_Pneumothorax_patient59888.jpg` | 56 | Male | AP | Pneumothorax | Pneumothorax 0.00 ✗ | No Finding 0.98 | Pneumothorax 0.26 ✗ | No Finding 0.93 |
| `6_honest_misses/03_missed_Pneumonia_patient39895.jpg` | 30 | Male | AP | Edema, Pneumonia (uncertain: Enlarged Cardiomediastinum, Lung Opacity) | Edema 0.06 ✗, Pneumonia 0.00 ✗ | No Finding 0.94 | Edema 0.21 ✗, Pneumonia 0.02 ✗ | No Finding 0.92 |

## Model readiness check

All 92 demo images were scored by the three exported models with the dashboard's pipeline. Findings with private-model VAL AUROC of at least 0.70 (used for group 2): Cardiomegaly, Edema, Pneumothorax, Pleural Effusion, Support Devices.
