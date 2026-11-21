# 08 — Implementation Plan

## 8.1 Repository layout

```
dp-mmfl-v2/
├── configs/
│   ├── base.yaml               # paths, labels, uncertainty policy, seeds
│   ├── central.yaml            # E1/E2
│   ├── fedavg.yaml             # E3
│   └── dpfed.yaml              # E4/E5 (target_eps, C, M, B, privacy_unit)
├── src/dpmmfl/
│   ├── constants.py            # LABELS (14), COMP5, policy defaults
│   ├── data/
│   │   ├── parse.py            # CSV → meta (patient_id, study_num, labels, masks, tab)
│   │   ├── splits.py           # patient-disjoint TEST/VAL/POOL + overlap asserts
│   │   ├── partition.py        # Dirichlet non-IID by patient, min size
│   │   └── cached.py           # Dataset over feature cache; patient-grouped sampler
│   ├── features/extract.py     # frozen xrv DenseNet → cache/*.npy (resumable)
│   ├── models/head.py          # FusionHead (film | concat | image_only)
│   ├── train/
│   │   ├── losses.py           # masked BCE + pos_weight
│   │   ├── central.py          # centralized training loop
│   │   ├── client.py           # local non-DP / DP-SGD (torch.func)
│   │   ├── server.py           # FedAvg / FedAvgM
│   │   └── privacy.py          # σ calibration, RDP accountants
│   ├── eval/
│   │   ├── metrics.py          # AUROC/AUPRC/F1/ECE, bootstrap CI, thresholds
│   │   ├── fairness.py
│   │   ├── gradcam.py
│   │   └── mia.py              # optional
│   └── utils.py                # seeding, logging, config loading
├── scripts/
│   ├── 01_prepare.py           # parse + splits + partition
│   ├── 02_extract.py           # feature cache (GPU)
│   ├── 03_sanity.py            # E0 + checks from doc 09
│   ├── run.py                  # --exp E1|E2|E3|E4|E5|E6 --seed --eps
│   ├── run_all.sh / run_all.ps1
│   ├── aggregate.py            # summary.csv + all figures
│   └── explain.py              # Grad-CAM grid + modality ablation
├── tests/                      # tiny unit tests: label mapping, overlap, DP clip/noise shapes
├── docs/                       # this folder
└── requirements.txt
```

## 8.2 Dependencies

```
torch, torchvision          (CUDA build matching the GPU)
torchxrayvision
opacus                      (accountant utilities only)
numpy, pandas, pyarrow, scikit-learn, scikit-image
pyyaml, tqdm, matplotlib
captum                      (optional)
```

## 8.3 Commands

```bash
python scripts/01_prepare.py  --root /data --config configs/base.yaml
python scripts/02_extract.py  --root /data --config configs/base.yaml      # GPU, once
python scripts/03_sanity.py   --config configs/base.yaml                   # STOP if any check fails
python scripts/run.py --exp E1 --seed 0
python scripts/run.py --exp E2 --seed 0
python scripts/run.py --exp E3 --seed 0
python scripts/run.py --exp E4 --eps 8 --seed 0     # then 3, 1
python scripts/run.py --exp E5 --eps 8 --seed 0
bash   scripts/run_all.sh                            # seeds 0–2 for everything
python scripts/aggregate.py
python scripts/explain.py --exp E4 --eps 8
```

## 8.4 Hardware: RTX 3050 vs Colab vs Kaggle

Only Stage A (feature extraction, ~190k images, one pass, no backprop) needs a GPU. Everything after works on cached features and is light (CPU is fine, GPU faster).

| Option | Verdict |
|---|---|
| **RTX 3050 (local)** | ✅ Recommended if the dataset is already on that machine. 4–8 GB VRAM is plenty for frozen DenseNet-121 inference at batch 64 fp16. Bottleneck is JPEG decoding → use `num_workers` 4–8 and an SSD. No session timeouts. |
| **Kaggle Notebook** | ✅ Best if the data is *not* local: CheXpert-small mirrors exist as Kaggle datasets (attach, no download), free T4/P100, ~12 h sessions. Extract features, save them as notebook output (~400 MB), download, continue locally. Respect the Stanford dataset terms. |
| **Colab (free)** | ⚠️ Works but painful: uploading/unzipping ~11 GB each session, disconnects, Drive I/O is slow (copy to `/content` first). Use only if neither of the above is available. |

**Recommendation:** extract features on the 3050 (or Kaggle if data isn't local), then run all experiments locally. Don't fine-tune the CNN at all — that's what would make the 3050 a problem.

## 8.5 Timeline (≈ 36 h wall clock, ~22 h of work)

| Hours | Work | Done when |
|---|---|---|
| 0–2 | Env setup, `01_prepare.py`, splits + partition | overlap = 0, partition table printed |
| 2–4 | `02_extract.py` running (write models/eval code meanwhile) | cache files + integrity checks pass |
| 4–6 | `03_sanity.py` (E0 logistic regression) | test mean-5 AUROC clearly > 0.7 |
| 6–9 | E1, E2 centralized | E2 ≥ E1 or honest "no gain" |
| 9–14 | E3 FedAvg, E4 DP at ε 8/3/1, E5 | privacy-utility curve |
| 14–17 | Seeds, E7 fairness, E8 comms, aggregate figures | `summary.csv` + plots |
| 17–20 | Grad-CAM, (E6/E9 if time) | figure grid |
| 20–24 | Update slides/paper, buffer for re-runs | — |
| 24–36 | Sleep + slack. Don't skip sleep. | — |

Split work across 4 people: (1) data+extraction, (2) model+central+metrics, (3) FL+DP, (4) eval/fairness/Grad-CAM/plots/slides.
