# 03 — Data and Preprocessing

## 3.1 Source

`CheXpert-v1.0-small/`
- `train.csv` — ~223k rows (frontal + lateral), labels auto-extracted from reports (values 1, 0, -1, blank).
- `valid.csv` — 234 rows, ~200 patients, radiologist-consensus labels (only 0/1).
- Images: downsampled JPEGs, path in the `Path` column, relative to the parent of `CheXpert-v1.0-small/`.

Columns: `Path, Sex, Age, Frontal/Lateral, AP/PA, <14 labels>`.

The 14 labels (fixed order, used everywhere — define once in `constants.py`):
```
No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion,
Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion,
Pleural Other, Fracture, Support Devices
```
Competition 5 (headline metric): `Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion`.

## 3.2 Parsing (`data/parse.py`)

Per row derive, **from the `Path` string only** (never from file listing order):
- `patient_id` = `patientXXXXX` segment
- `study_id` = `patientXXXXX/studyN`
- `study_num` = N (int)
- `view` = `frontal` / `lateral`
- `abs_path` = root / Path

Filter: keep **frontal** only (`Frontal/Lateral == "Frontal"`). Keep the original row index as `row_id` and carry it through every later file. Assert `abs_path` exists for a random 1,000 rows.

## 3.3 Labels and uncertainty policy

| Raw | Train target | Train mask |
|---|---|---|
| 1 | 1 | 1 |
| 0 | 0 | 1 |
| blank | 0 | 1 (standard "not mentioned = negative"); configurable to 0 |
| -1 | per-label policy below | 1, or 0 for `ignore` |

Default per-label policy (configurable in YAML):
- **U-Ones** (map to 1): Atelectasis, Edema
- **U-Zeros** (map to 0): Cardiomegaly, Consolidation, Pleural Effusion
- **Ignore** (mask = 0): all other labels

`valid.csv` has no -1: use as is. Never pass -1 or NaN into a metric.

## 3.4 Tabular (metadata) features — 9 dims

| Feature | Encoding |
|---|---|
| age | `clip(age, 18, 90)`, z-scored with **train** mean/std |
| age_missing | 1 if age ≤ 0 or NaN (then age = 0 after z-score) |
| sex | one-hot Male / Female / Unknown (3) |
| projection | one-hot AP / PA / Other (3) |
| prior_studies | `log1p(study_num - 1)` — number of earlier studies for this patient (known at scan time) |

These are all available **before** the radiologist reads the image → zero diagnostic leakage.

## 3.5 Splits (patient-disjoint)

```
official valid.csv (234 imgs)            → TEST  (touch once, at the end)
train.csv frontal, by patient (seed 42):
   90% patients → FL pool (partitioned across clients)
   10% patients → VAL (model selection, threshold tuning, early stopping)
Inside each client: 10% of its patients → local val (per-client audit)
```
Assert pairwise patient overlap = 0 between every split (keep v1's check — it was good).

Optional speed knob: `subset_patients: null | int` to train on a random patient subset if time is short.

## 3.6 Non-IID partition (`data/partition.py`)

K = 5 clients, partition **patients** (not images):
1. Give each patient a proxy class = index of the most frequent positive among the 5 competition labels (or "none").
2. For each proxy class, draw client proportions from Dirichlet(α), α = 0.5 default.
3. Assign patients; then enforce `min_patients_per_client = 1500` by moving random patients from the largest client. (v1 had clients with 34 and 62 patients — per-client metrics on those are noise.)
4. Save `partition.json` = {client_id: [patient_ids]} and a summary table (patients, images, label prevalence, weight p_k).

Also support `alpha ∈ {0.1, 0.5, 10}` for the optional heterogeneity ablation.

## 3.7 Feature extraction (`features/extract.py`) — run once

Encoder: `torchxrayvision.models.DenseNet(weights="densenet121-res224-nih")`, eval mode, no grad, AMP fp16.
**Do not use** the `-all` or `-chex` weights — they were trained on CheXpert (test leakage).

Per image:
```python
img = skimage.io.imread(path)                 # uint8 grayscale
img = xrv.datasets.normalize(img, 255)        # → [-1024, 1024]
img = img[None, ...] if img.ndim == 2 else img.mean(2)[None, ...]
img = XRayCenterCrop()(img); img = XRayResizer(224)(img)
x = torch.from_numpy(img).float()             # [1, 224, 224]
```
Feature = `relu(model.features(x)) → adaptive_avg_pool2d(1) → flatten` → 1024-d (equivalently `model.features2(x)` if present).

Outputs (aligned by `row_id`, same order as `meta.parquet`):
- `cache/feat_train.npy` float16 [N_train, 1024], `cache/feat_test.npy` [234, 1024]
- `cache/meta_train.parquet`: row_id, patient_id, study_num, labels(14), mask(14), tab(9), split, client_id
- `cache/meta_test.parquet`

Performance: DataLoader with `num_workers=4–8` (on Windows wrap in `if __name__ == "__main__":`), batch 64–128. ~190k images ≈ 30–60 min on an RTX 3050 or Kaggle T4; ~400 MB on disk. Save in chunks so a crash can resume.

Fallback encoder if xrv fails to install: torchvision DenseNet-121 ImageNet weights, gray → 3 channels, ImageNet normalisation, resize 224. Same 1024-d output.

## 3.8 Integrity checks at the end of this stage

- `len(feat) == len(meta)`; no NaN/Inf in features.
- Re-extract features for 20 random rows on the fly and compare (max abs diff < 1e-2).
- Label prevalence per split printed and sane (e.g. Support Devices ~50%, Pleural Effusion ~40% in train).
