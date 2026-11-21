# Implementation: what was built, how, and what the results mean

Scope: DP-MMFL v2 as run on **full CheXpert-v1.0-small** (191,027 frontal images, 64,534 patients), 3 seeds, plus a small 4k-patient dry run used for development.
Deviations from the design docs are in `DECISIONS.md`. Reproduction commands are in the root `README.md`.

> **v2.1 code (current repo).** The results below are from v2 (single `nih` encoder). The code has since moved to v2.1: three frozen encoders
> (`nih`, `mimic_ch`, `pc`), linear/FiLM heads with BCE or ASL, PRV accounting, early stopping, F1/F2 fairness, top-k compression, a stricter
> results/TEST policy (DECISIONS 18–30). Some file and function names below (e.g. `data/cached.py`) refer to v2 and were renamed.
> The v2.1 full-data results (overnight run of 2026-11-03) are in **section 6** and `results/v2.1-final/master_table.md`; they supersede
> section 3 as the headline numbers. Section 5 covers the scale ladder, the membership-inference stress test and the dashboard.
> The v2 full-data results in section 3 are in `results/v2-large/`.

---

## 1. Data processing techniques

### 1.1 Parsing and identity (`data/parse.py`)
- `patient_id`, `study_num` and view are extracted **only from the `Path` string** with one regex. Image lists are never zipped with CSV rows; every artifact carries the original `row_id`. This directly removes the most likely cause of v1's chance-level AUROC (image/label misalignment).
- Path-derived view is cross-checked against the CSV `Frontal/Lateral` column; the 14 label columns are asserted equal to `constants.LABELS` (single source of truth for names and order).
- Only **frontal** images are kept (191,027 of 223,414 train rows; 202 of 234 test rows). 1,000 random image paths are checked to exist.

### 1.2 Label handling (uncertainty policy)
| Raw value | Target | Mask |
|---|---|---|
| 1 | 1 | 1 |
| 0 | 0 | 1 |
| blank | 0 ("not mentioned = negative") | 1 |
| -1 (uncertain) | U-Ones for Atelectasis, Edema; U-Zeros for Cardiomegaly, Consolidation, Pleural Effusion; **ignored (mask 0)** for the rest | 1 or 0 |

Targets and masks are separate arrays; the loss uses the mask, and metrics never receive -1 or NaN.

### 1.3 Metadata (second modality), 9 features
age (clipped 18–90, z-scored with training statistics, plus an "age missing" flag), sex one-hot (M/F/unknown), projection one-hot (AP/PA/other), and `log1p(study_num − 1)` (number of earlier studies). All are known **at scan time**, so there is no diagnostic leakage. CheXpert-small contains no report text, which is why text could not be the second modality.

### 1.4 Splits (patient-disjoint, `data/splits.py`)
- Official `valid.csv` (200 patients, 202 frontal images) = **TEST**, used once per run after training.
- Of train patients: 10% (6,453 patients, 19,206 images) = **VAL** for model selection, thresholds, early stopping; 90% (58,081 patients, 171,821 images) = **federated pool**.
- Inside each client, 10% of its patients form a **local-val** set used only for the per-client audit and excluded from training.
- Pairwise patient overlap across TEST / VAL / every client is asserted to be 0.

### 1.5 Non-IID federated partition (`data/partition.py`)
Patients (not images) are partitioned over K = 5 clients. Each patient gets a proxy class (their most frequent positive among the 5 competition labels, or "none"); client proportions per proxy class are drawn from Dirichlet(α = 0.5); a minimum of 1,500 patients per client is then enforced by moving random patients from the largest client (v1 had clients with 34 and 62 patients).

Resulting clients (patients): 10,703 / 14,745 / 6,163 / 5,353 / 21,117. Label prevalence differs strongly across clients (e.g. Edema 14% to 48%, Cardiomegaly 4% to 25%).

### 1.6 Image preprocessing and feature caching (`features/extract.py`)
- Images are normalised to the TorchXRayVision convention ([-1024, 1024]), centre-cropped and resized to 224×224, one grey channel.
- A **frozen** TorchXRayVision DenseNet-121 with `densenet121-res224-nih` weights (never the `-all` / `-chex` weights, which were trained on CheXpert and would leak test data) turns every image into a 1,024-d vector (ReLU → global average pool of the last feature map).
- Features are cached once as float16 (about 390 MB), in resumable chunks, with integrity checks: shape equals metadata length, no NaN/Inf, nonzero variance, and 20 random images re-extracted on the fly (max difference 1.95e-3, threshold 1e-2).
- Consequence: every later experiment (central, FL, DP, 3 seeds) trains only a 0.36M-parameter head on cached vectors, in minutes.

### 1.7 Class imbalance
Per-label `pos_weight = clip(neg/pos, 1, 10)` from the training pool, applied inside a masked BCE loss.

---

## 2. How the deep-learning algorithms were used

### 2.1 Model (`models/head.py`)
```
X-ray -> frozen DenseNet-121 (NIH) -> f_img (1024)  ┐
metadata (9) ---------------------------------------┴-> FusionHead -> 14 logits (sigmoid)
```
FusionHead (0.37M parameters, **no BatchNorm** so it is DP-compatible): `LayerNorm → Linear(1024,256) → GELU → Dropout`, then **FiLM** conditioning from the metadata (`h·(1+γ)+β`, generator zero-initialised so the model starts exactly as the image-only model), then a residual block and a linear classifier. Variants: `image_only` (E1), `film` (E2 and all FL/DP runs), `concat` (ablation). **Modality dropout** (p = 0.1) randomly zeros the metadata during training.

### 2.2 Loss
Masked BCE-with-logits with `pos_weight`: per image, sum over labels of mask × BCE divided by max(1, sum of masks).

### 2.3 Centralized training (E1, E2)
AdamW (lr 1e-3, wd 1e-4), batch 256, 20 epochs; the epoch with best VAL mean-5 AUROC is kept.

### 2.4 Federated learning (E3, `train/server.py`, `train/client.py`)
Five simulated hospitals; the global model is the head only. Per round every client starts from the global weights and runs the **same number of local steps** (50), so the biggest client no longer does far more work than the smallest. The server computes the patient-count-weighted mean of the weight deltas (FedAvg), or FedAvg with server momentum (β = 0.9) for DP runs. 30 rounds; the round with best VAL mean-5 AUROC is selected.

### 2.5 Patient-level differential privacy (E4, `train/privacy.py`)
DP-SGD with the **patient** as the privacy unit:
1. Poisson-sample patients (probability q = 256 / patients in the client).
2. Cap each patient at 8 images, compute one gradient per patient (mean over their images) with `torch.func` (`functional_call` + `vmap(grad)`), no Opacus wrapper.
3. Clip each patient's gradient to L2 norm C = 1.
4. Sum, add Gaussian noise N(0, σ²C²), divide by the **expected** batch size, take an SGD step (lr 0.03).
5. σ is **calibrated per client** with Opacus' RDP utilities so that every client reaches the same target ε (8, 3 or 1; δ = 1e-5) after R×S steps. A per-client `RDPAccountant` logs ε every round. Because every patient lives at one client, the reported guarantee is the max over clients.

E5 re-runs this with the **image** as the unit (q = batch / images). Its worst-case patient-level guarantee degrades with images per patient (group privacy), so equal nominal ε is not equal protection.

### 2.6 Evaluation and audits
Mean AUROC over the 5 competition labels on TEST with a 1,000-resample **patient-level bootstrap** 95% CI; AUROC only where both classes exist; AUPRC, F1/sensitivity/specificity at VAL-tuned (Youden) thresholds, ECE. Fairness on VAL by sex, age bin and projection; per-client AUROC on each client's local-val; communication cost; a loss-threshold **membership-inference** audit; Grad-CAM through the chained model plus a metadata on/off ablation.

### 2.7 Safeguards (`scripts/03_sanity.py`, `tests/`)
Sanity gate (exits non-zero on failure): label/path alignment by `row_id`, ID regex, zero patient overlap, target/mask domains, label order, prevalence, feature-cache checks, **logistic-regression floor** (TEST mean-5 AUROC 0.833; must exceed 0.70), a **shuffled-label control**, overfit-a-tiny-batch, train-vs-VAL gap, and per-label AUROC > 0.45. Unit tests cover label mapping, split overlap, per-patient clipping bound, DP gradient equals the plain gradient when σ = 0 and C = ∞, noise scale, and accountant calibration (ε within 1% of target, equal across clients).

---

## 3. Results and what they mean

All numbers are TEST mean-5 AUROC on the official 202-image CheXpert valid set, mean ± sd over 3 seeds (full scale). VAL is the larger, more reliable set and is given for comparison.

> **Correction notice.** An earlier version of this file reported FedAvg 0.829 and DP ε=8 0.813. Those federated numbers were invalid: `vector_to_parameters` made each client's parameters views of the server vector, so clients trained sequentially on the global weights and updates were double counted (see DECISIONS 14). The bug is fixed, covered by a regression test, and every federated run below was regenerated. Central runs (E0–E2) were never affected. Invalid runs are kept in `archive/results_aliasing_bug/`.

| Run | VAL | TEST | Max ε |
|---|---|---|---|
| E0 logistic regression (sanity floor) | – | 0.833 | – |
| E1 central, image only | 0.746 | 0.818 ± 0.001 | – |
| E2 central, FiLM (image + metadata) | 0.749 | 0.823 ± 0.009 | – |
| E2 central, concat | 0.749 | 0.816 ± 0.003 | – |
| E3 FedAvg | 0.736 | 0.817 ± 0.002 | – |
| E4 patient-level DP, ε = 8 | 0.711 | 0.803 ± 0.005 | 8.0 |
| E4 patient-level DP, ε = 3 | 0.710 | 0.804 ± 0.003 | 2.95 |
| E4 patient-level DP, ε = 1 | 0.701 | 0.798 ± 0.007 | 0.88 at selected round |
| E5 image-level DP, ε = 8 | 0.711 | 0.804 ± 0.004 | 8.0 (patient-level weaker) |

**The pipeline works.** v1 scored 0.46–0.54. v2's non-private models are 0.82–0.83 and the sanity gate passes. Even plain logistic regression on the cached features reaches 0.833.

**Frozen features limit the headline.** E0 is as good as or better than every trained head (0.833 vs 0.816–0.823). The encoder carries the signal; the head adds nothing over a linear probe.

**Federation is nearly free; DP costs about 0.013.** FedAvg is 0.817 vs central 0.818 (equal within noise). DP at ε = 8 costs about 0.013 TEST and 0.025 VAL. The curve is flat: ε = 8, 3 and 1 are within noise on TEST, and ε = 1 costs only about 0.01 more on VAL. A 0.37M-parameter head with 5k–21k patients per client gives DP a favourable signal-to-noise ratio; at the 4k-patient dry run the cost was far larger, so this depends on data scale. (Small-scale FL numbers were produced before the bug fix and should not be quoted.) ε values at the selected round (2.95, 0.88) are what had been spent when VAL peaked.

**Image-level DP matches patient-level DP** at ε = 8 (0.804 vs 0.803), but its patient-level guarantee is weaker via group privacy, so equal nominal ε is not equal protection.

**The second modality adds nothing measurable.** FiLM vs image-only is +0.005 (noise); zeroing the metadata at test time changes AUROC by about 0.003 (E3: 0.8155 → 0.8181; E4 ε=8: 0.8028 → 0.8033); FiLM γ/β magnitudes are small. The multimodal claim is not supported by this data.

**Communication reduction works and is measured.** Updates are quantized to fp16 or int8 (per-block scales), and bytes are counted from the actual payloads over 30 rounds × 5 clients.

| Setting | MB total | Saving | TEST |
|---|---|---|---|
| fp32 up/down (E3) | 441.9 | 0% | 0.817 |
| fp16 up and down | 221.0 | 50.0% | 0.815 |
| int8 up, fp16 down | 166.0 | 62.4% | 0.818 |
| DP ε=8, int8 up, fp16 down | 166.0 | 62.4% | 0.8035 (vs 0.8034 fp32) |

Accuracy is unchanged within noise (VAL identical to 3–4 decimals). Quantizing a DP update is post-processing, so the ε guarantee is unchanged. The head is only 0.37M parameters (~1.5 MB per client per round in fp32, versus v1's 43.6 MB, because the frozen encoder is deployed once and never sent). This is quantization only; no sparsification or error feedback was tried.

**Fairness: measured, mitigation did not work.** VAL AUROC gaps (max − min over groups):

| Run | age | projection | sex | client spread |
|---|---|---|---|---|
| E1 central | 0.073 | 0.056 | 0.003 | 0.093 |
| E3 FedAvg | 0.078 | 0.057 | 0.007 | 0.098 |
| E4 ε = 8 | 0.080 | 0.057 | 0.008 | 0.110 |
| E4 ε = 1 | 0.078 | 0.058 | 0.009 | 0.114 |
| E3 + group reweighting | 0.081 | 0.063 | 0.007 | 0.099 |
| E4 ε=8 + group reweighting | 0.080 | 0.058 | 0.008 | 0.110 |

Gaps are about the same centrally, federated, and under DP, so federation and DP add little subgroup harm (the client spread grows slightly, 0.098 → 0.110). Age (under-40 best, over-80 worst) and AP-vs-PA gaps are present from the start. The group-reweighted loss (weights from age and projection group frequencies, clipped to [0.5, 3]) did **not** shrink them: without DP it widened slightly and cost 0.009 TEST; under DP it changed nothing. A plausible but **untested** reason for the DP case is that per-patient clipping to C = 1 removes the weight signal. So the framework supports fairness measurement and one tried mitigation, but fairness is not achieved. Fairness weights also use global (non-federated) statistics.

**Membership inference.** Loss-threshold attack AUC is ≈ 0.50 for all federated runs and about 0.52 for central. That is consistent with no leakage but is a weak attack and does not prove privacy beyond ε.

**Grad-CAM** (`results/v2-large/E4_eps8/0/gradcam_Pleural_Effusion.png`, regenerated from the corrected model): qualitative only; heatmaps are diffuse.

### Limitations to state with these numbers
- TEST is 202 images from 200 patients; CI ≈ ±0.03, so differences below ~0.03 are not resolved (DP cost, compression effects and reweighting effects are all near that scale; VAL agrees in direction).
- Hyperparameters (DP lr) were tuned on VAL, which consumes privacy budget that is not accounted for; the DP lr sweep was run before the bug fix and not repeated.
- Only 3 seeds, one partition seed, α = 0.5; the α sweep (E6) has not been rerun at full scale after the bug fix.
- Frozen encoder is public-data pretrained (NIH), outside the DP guarantee; the guarantee covers the head only.
- Simulated FL in one process; no secure aggregation, no network adversary, only a loss-threshold MIA.
- Fairness mitigation is ineffective; the clipping explanation is untested.
- Frontal images only.

---

## 4. Is this paper-worthy?

**Short answer: not yet for a strong venue; it is a reasonable workshop / student-conference paper if the gaps below are closed. It is already solid as a course project and as a corrective follow-up to the review.**

What helps:
- A rigorous, honest pipeline: leakage-free splits, official test set, bootstrap CIs, a sanity gate with controls, and unit-tested DP code.
- Patient-level (not image-level) DP with per-client noise calibration to equal ε, implemented and audited with an RDP accountant at 191k-image scale.
- A useful and reportable finding: at realistic scale, patient-level DP on a small head over frozen chest-X-ray features costs little (about 0.013 TEST at ε = 8, with a flat curve down to ε = 1) and the metadata modality contributes nothing.

What limits it:
- **Novelty is thin.** DP-SGD, FedAvg, FiLM and frozen-encoder transfer are all standard; per-client σ calibration is a modest idea and per-patient clipping exists in the literature. The combination is a careful engineering contribution, not a new method.
- **The headline claims need ablations that were not run**: fixed-σ vs calibrated-σ (to actually demonstrate that the "privacy tax" is fixed rather than asserting it), image-level vs patient-level comparison under a common *patient-level* guarantee, α sweep at full scale, and a linear-probe-under-DP baseline (since E0 ≈ trained head, a DP logistic regression is the natural competitor).
- **Fairness mitigation is negative.** The one mitigation tried (group-reweighted loss) did not reduce gaps, so fairness is audited, not delivered.
- **The multimodal story is negative.** A paper titled "multimodal" cannot rest on a modality that does nothing. Either add a real second modality (e.g. CheXpert+ report text with a careful leakage argument) or reframe as "DP federated learning on chest X-rays" with the null metadata result reported.
- **Evidence strength.** 202-image TEST, 3 seeds, and differences among FedAvg, ε = 8, 3 and 1 are all inside or near the CI. Claims should be phrased as "no detectable loss", not "no loss".
- **Privacy scope.** Tuning on VAL is unaccounted; the frozen encoder is outside the guarantee. (The weak-audit point is addressed by LiRA in 5.2b.)

Suggested path to a workshop-level submission (roughly 1–2 weeks of work):
1. Run the three missing ablations above plus E6 at full scale, with 5 seeds and CIs over seeds.
2. Add a stronger external check (another dataset or the larger CheXpert/MIMIC hold-out) so conclusions do not rest on 202 images.
3. Add a stronger privacy audit (e.g. a trained-shadow or likelihood-ratio attack) and a DP-tuned-hyperparameter statement.
4. Decide the framing: either add real text, or drop "multimodal" from the title and present the null metadata result.
5. Compare against at least one published DP-FL medical baseline.

Realistic venues with that work: MICCAI/NeurIPS-type workshops on federated or trustworthy medical imaging, or a regional/student conference. A full journal or main-track paper would need a genuinely new method or a much larger empirical study.

---

## 5. v2.1 additions: scale ladder, privacy attack, dashboard

### 5.1 Scale ladder (`docs/SCALE_RESULTS.md`)
The v2.1 code was run at two subset sizes (small: 2k FL patients, medium: 20k), nih encoder, 15 rounds × 50 steps, 3 seeds, and compared with the v2
full-data run. DP cost at ε=3 (VAL, vs FedAvg) falls with data: 0.058 small, 0.031 medium, 0.025 large. ε=1 is unusable at small scale
(TEST 0.60) and usable from medium up (0.78). int8 up / fp16 down saves 62.4% of traffic at every scale with no AUROC change.
Group reweighting (F1) still does not reduce gaps. Per-group thresholds (F2) reduce the AP/PA sensitivity gap from 0.18–0.30 to 0.01–0.05
on held-out VAL patients, so F2 is the one working mitigation.

### 5.2 Membership-inference stress test (`scripts/mia_demo.py`, `results/mia_demo/`)
On the real runs the loss-threshold attack gives AUC ≈ 0.50–0.53 for every model, including those without DP, so the audit cannot show what DP adds.
The stress test makes the difference visible: the same 300 FL-pool patients train an image-only head for 400 steps, once without DP and with no
early stopping (deliberate overfitting), and once with patient-level DP-SGD at ε = 8, 3, 1. Non-members are 300 held-out VAL patients. 5 seeds.

| Arm | Attack AUC | Member / non-member loss | VAL AUROC |
|---|---|---|---|
| no DP | 0.857 ± 0.017 | 0.32 / 1.64 | 0.651 |
| DP ε=8 | 0.525 ± 0.012 | 0.89 / 0.95 | 0.666 |
| DP ε=3 | 0.530 ± 0.015 | 1.00 / 1.08 | 0.623 |
| DP ε=1 | 0.521 ± 0.019 | 6.03 / 6.41 | 0.548 |

Without DP the attacker identifies training patients well; with DP the attack is near chance. At 300 patients DP costs utility (ε=1 most), which is
why the main experiments use thousands of patients. TEST is not used.

### 5.2b Likelihood-ratio attack and DP audit (`scripts/mia_lira.py`, `eval/lira.py`, `results/mia_lira/`)
The loss threshold is the weakest standard attack. LiRA (Carlini et al. 2022) is the reference attack: the attacker trains shadow models
with and without each patient, fits the patient's IN and OUT distributions of the statistic −log(patient loss), and scores the target model
by the likelihood ratio (online) or by the z-score against OUT only (offline). Results are reported as AUC and, more importantly, as the
true-positive rate at 1% and 0.1% false-positive rate: how many training patients can be named while almost never accusing an unseen one.

**Stress test** (600 candidate pool patients, 16 shadow models per arm, each trained on a random half; each model is the target once):

| Training | Loss threshold AUC | LiRA online AUC | Found at 1% FPR (best attack) | Found at 0.1% FPR | VAL AUROC | ε shown by attack |
|---|---|---|---|---|---|---|
| no DP, overfit | 0.874 | 0.944 | 42.6% | 14.9% | 0.651 | – |
| no DP, early stopped | 0.595 | 0.689 | 7.9% | 2.0% | 0.688 | – |
| DP ε = 8 | 0.537 | 0.604 | 2.5% | 0.5% | 0.668 | 0.39 |
| DP ε = 3 | 0.539 | 0.539 | 1.7% | 0.15% | 0.611 | 0.22 |
| DP ε = 1 | 0.531 | 0.502 | 1.0% | 0.04% | 0.529 | 0.03 |

- Early stopping, the usual "regularisation is enough" argument, still lets LiRA name 8% of training patients at 1% FPR (8× chance).
- DP ε = 8 has slightly lower accuracy (0.668 vs 0.688) and cuts that to 2.5%. ε = 3 gives 1.7% (close to chance); ε = 1 is at chance.
- **DP audit.** The empirical ε lower bound uses Clopper–Pearson 95%; the threshold is chosen on one half of the trials and the bound computed on the other. It stays well below the accountant's ε in every arm (0.39 < 8, 0.22 < 3, 0.03 < 1), so the attack finds no violation of the guarantee. These bounds are far from tight: natural patients are not worst-case canaries.

**The full-scale models** (seed 0 finals; 2,000 training patients vs 2,000 VAL patients; 8 reference models trained centrally with the frozen config, without any candidate; offline LiRA):

| Target | Loss threshold AUC | LiRA AUC | Found at 1% FPR | Found at 0.1% FPR |
|---|---|---|---|---|
| E2 central (no DP) | 0.520 | **0.684** | **6.1%** | 1.6% |
| E3 FedAvg (no DP) | 0.507 | 0.551 | 1.7% | 0.4% |
| E4 DP ε = 8 | 0.496 | 0.495 | 0.8% | 0.2% |
| E4 DP ε = 3 | 0.496 | 0.497 | 0.8% | 0.2% |
| E4 DP ε = 1 | 0.496 | 0.495 | 0.8% | 0.3% |

This corrects the earlier reading. The loss audit said even the central model leaks nothing (0.52). LiRA shows the central model does leak, naming 6% of its training patients at 1% FPR. FedAvg leaks less, and the DP models are at chance under the same attack. The references are central non-DP models, which is the matched attack for E2 but mismatched for E3 and E4. A matched attack on those (shadow FL or DP runs) could be stronger, but it costs hours per target.

Design choices (DECISIONS 40):
- The statistic −log(loss) was chosen over the mean signed logit on the overfit shadow models (TPR at 1% FPR 0.43 vs 0.09). An attacker can make the same choice with their own shadows.
- Trials across shadow models are not independent, so the ε lower bound is an estimate.
- `python scripts/mia_lira.py rescore` recomputes every number from the saved statistics.

### 5.3 Dashboard (`dashboard/app.py`, `src/dpmmfl/serve.py`)
`streamlit run dashboard/app.py`. Four tabs: **Results** (scale comparison, privacy vs accuracy, traffic, fairness gaps, per-finding AUROC),
**Predict** (example X-rays from `demo/` or an upload, patient details, 14 probabilities, Grad-CAM, saved records in `demo/patient_records.jsonl`),
**Privacy** (the stress test above and a live attack on any finished run) and **About**. Uploaded images go through the same preprocessing as the
cached features (`features.extract.preprocess`) and the same per-label temperature calibration as the run. Example images come from
`scripts/make_demo_set.py`: per finding, 3 images of VAL patients (never trained on) and 1 of an FL-pool patient, read from `train.csv`.
The archived v2 heads cannot be loaded into the v2.1 head class, so v2 runs appear in the charts but not in Predict.

---

## 6. v2.1 full-data results (headline)

Overnight run, 2026-11-03 (`scripts/run_night.sh`, 4 h 25 min, 0 failures). Full CheXpert-v1.0-small, three frozen encoders
(nih + mimic_ch + pc), configuration frozen on VAL before any TEST read (`results/v2.1-final/frozen.yaml`: FiLM head, BCE with pos_weight,
EMA 0.999; DP lr 0.03, C = 1.0, B = 256). 36 final runs (12 configs × 3 seeds), each read TEST exactly once (`test_access.jsonl` has 36 lines).
Mean ± sd over 3 seeds.

| Run | VAL | TEST | ε spent (PRV / RDP) |
|---|---|---|---|
| E0 linear probe (logistic regression) | 0.768 | 0.856 ± 0.001 | – |
| E1 central, image only | 0.768 | 0.842 ± 0.003 | – |
| E2 central, FiLM (image + metadata) | 0.771 | 0.846 ± 0.005 | – |
| E3 FedAvg, 5 non-IID clients | 0.759 | 0.840 ± 0.003 | – |
| E4 patient-level DP, ε = 8 | 0.739 | 0.813 ± 0.001 | 8.00 / 9.04 |
| E4 patient-level DP, ε = 3 | 0.737 | 0.818 ± 0.009 | 3.00 / 3.31 |
| E4 patient-level DP, ε = 1 | 0.727 | 0.819 ± 0.002 | 0.91 / 0.99 |
| E5 image-level DP, ε = 8 | 0.737 | 0.815 ± 0.003 | 8.00 / 9.51 (per image) |
| E7 DP ε = 3 + group reweighting (F1) | 0.737 | 0.827 ± 0.005 | 3.00 / 3.31 |
| E8 FedAvg, int8 up / fp16 down | 0.759 | 0.842 ± 0.002 | – |
| E8 DP ε = 3, int8 up / fp16 down | 0.737 | 0.818 ± 0.009 | 3.00 / 3.31 |
| E8 DP ε = 3, top-k up / fp16 down | 0.721 | 0.805 ± 0.011 | 3.00 / 3.31 |

All ε values meet their targets under PRV, and RDP is always the looser bound, as expected.

What the numbers say:

- **Better than v2 everywhere**, by +0.02 to +0.03 TEST (for example FedAvg 0.817 → 0.840, DP ε = 3 0.804 → 0.818) and by about +0.02 VAL. Most of that gain comes from the two extra encoders (nih-only trials reach VAL 0.749–0.752).
- **Federation costs about 0.01 VAL** and is within noise on TEST.
- **Patient-level DP costs about 0.02 VAL and 0.02–0.03 TEST** against FedAvg, and the cost barely depends on ε: ε = 1 loses only 0.01 VAL more than ε = 8. On TEST, the ordering of ε = 1, 3 and 8 is inside the CI.
- **Image-level DP** reaches the same accuracy as patient-level DP at nominal ε = 8. Its guarantee per patient is weaker.
- **Communication.** int8 up / fp16 down saves 62.4% of bytes, with no AUROC change, with or without DP. Top-k with error feedback saves 74.2% but costs 0.016 VAL under DP.
- **Fairness.**
  - Age gaps (0.07–0.08) and projection gaps (0.06–0.07) are the same centrally, federated and under DP.
  - F1 reweighting leaves VAL and the gaps unchanged. Its +0.009 TEST is inside the CI.
  - F2 per-group thresholds cut the AP/PA sensitivity gap from 0.16–0.26 to 0.015–0.06, so F2 is the mitigation that works.
- **Membership inference.** The loss-threshold audit gives AUC 0.50–0.52 for every run, but it is too weak to separate the models. With reference models, LiRA (5.2b) does separate them:
  - the central model names 6.1% of its training patients at 1% FPR (AUC 0.68);
  - FedAvg names 1.7% (AUC 0.55);
  - the DP models are at chance (AUC 0.495–0.497).

  In the shadow-model stress test, the empirical ε lower bounds stay below the guaranteed ε.
- **The head does not beat the linear probe.** E0 is the best TEST number (0.856) and equal on VAL (0.768 vs 0.771). The metadata modality again adds little: FiLM vs image-only is +0.003 VAL.

### 6.1 Paper-worthiness after this run
This run fixes several earlier weaknesses:

- Results now come from a frozen, VAL-only selection protocol, with a single TEST read per run.
- ε is reported from two accountants.
- There is a scale ladder showing how the cost of DP shrinks with data.
- There is a mitigation that works (F2) and an attack demonstration that shows what DP buys.

That is a solid basis for a workshop or student-conference paper on the evidence. The claims to make are:

- At full CheXpert scale, patient-level DP-FL on frozen public encoders costs about 0.02 AUROC.
- That cost is nearly flat from ε = 8 down to ε = 1.
- int8 compression is free.
- Post-hoc group thresholds, not loss reweighting, close the sensitivity gaps.

Gaps that remain, and what a reviewer will ask:

1. **TEST is 202 images.** Report VAL (6,400 patients) next to it. An external set (for example the MIMIC-CXR or NIH test split) would be needed for a stronger claim.
2. **The head does not beat the linear probe.** Frame the head as the trainable part under DP rather than as a modelling contribution, or add a DP linear-probe baseline.
3. **The multimodal claim is weak.** Metadata adds about 0.003. Either report this as a null result or drop "multimodal" from the title.
4. **Selection on VAL is not privacy-accounted.** This covers the 10 trials, the 9-point DP sweep and early stopping. State it, or cite a private-selection bound.
5. **The ablations are still missing.** These are fixed-σ vs calibrated-σ, an α sweep (E6) and more seeds (5).

