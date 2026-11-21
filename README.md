# DP-MMFL: differentially private multimodal federated learning on CheXpert

Multi-label chest X-ray classification (14 CheXpert labels, headline metric = mean AUROC over the 5 competition labels) with

- **frozen TorchXRayVision DenseNet-121 encoders** (`nih`, `mimic_ch`, `pc`, never CheXpert-trained weights) whose 1024-d features are cached once;
- a small **multi-encoder head** (per-encoder LayerNorm → Linear(1024, 256), concat; FiLM conditioning on 9 metadata features, or a linear head);
- **federated learning** over 5 simulated hospitals (Dirichlet α = 0.5, patient-disjoint), FedAvg / FedAvgM + server EMA;
- **patient-level DP-SGD** (Poisson sampling, per-patient clipping, per-client σ calibrated with PRV, ε reported from PRV and RDP);
- **fairness** audit (sex / age / AP-PA / client; bootstrap CIs, worst group, EO gaps, per-group ECE) with mitigations F1 (reweighting) and F2 (group thresholds);
- **communication compression** (fp16 / int8 / top-k with error feedback), bytes measured on the serialised payload;
- a **membership-inference** audit for every run, plus a stress test showing what DP prevents;
- a **dashboard** for results, predictions on chest X-rays and the privacy attack.

Docs: [docs/](docs/) (design spec `00–10`), [DECISIONS](docs/DECISIONS.md) (every deviation), [IMPLEMENTATION](docs/IMPLEMENTATION.md) (methods and results),
[SCALE_RESULTS](docs/SCALE_RESULTS.md) (small / medium / large), [NIGHT_RUN](docs/NIGHT_RUN.md) (full run).

## Layout

```
configs/         base.yaml (full scale), smoke.yaml (--smoke overrides), scale_{small,medium}.yaml (scale ladder)
src/dpmmfl/
  data/          parse (CSV → meta), splits, partition, store (GPU feature store, per-client patient index)
  features/      extract (one-pass multi-encoder extraction, integrity checks)
  models/        head (FusionHead, EMA)
  train/         losses, privacy (DP-SGD, PRV/RDP), comm (compression + wire format), client, server, central, infer
  eval/          metrics, fairness, mia, lira, gradcam
  results.py     run writer (run_id, manifest, refuse-overwrite, TEST guard)
  serve.py       run discovery, single-image prediction, live membership attack (used by the dashboard)
scripts/         pipeline: 01_prepare, 02_extract, 03_sanity, run, freeze, aggregate, explain
                 drivers: run_smoke.sh, run_night.{sh,ps1}, run_scale.sh; checks: smoke_checks
                 extras: compare_scales, export_model, make_demo_folders, make_demo_set, mia_demo, mia_lira, paper_figures
dashboard/       app.py (Streamlit)
tests/           fast unit tests (pytest, < 1 min)
docs/            design spec 00–10, DECISIONS, IMPLEMENTATION, SCALE_RESULTS, NIGHT_RUN; reference/ (Phase-1 report, survey paper, review)
report/          Phase-2 project report: main.tex (single file), images/, make_figures.py, REVIEW_AND_CHANGELOG.md
paper/           MBEC manuscript: main.tex, ESM_1.tex, cover_letter.tex, references.bib, figures/, analysis/ (derive_from_stored_results.py)

generated, not in git (see .gitignore):
CheXpert-v1.0-small/   dataset
cache/           meta_*.parquet, feat_<split>[_<enc>].npy features; scale_small/, scale_medium/ for the scale ladder
splits/          partition.json, tab_stats.json
results/         v2.1-final/ (final full-scale study), scale_small/, scale_medium/, v2-large/ (v2 full-scale runs),
                 mia_demo/, mia_lira/ (attacks), scale_comparison.*
models/          final models saved for demos by scripts/export_model.py (v2.1-large/)
demo/            images/, by_label/ and presentation/ (example X-rays), patient_records.jsonl (saved from the dashboard); demo_samples.csv is tracked
```

## Setup

```bash
pip install -e .[dev,dashboard]    # or: pip install -r requirements.txt
```
Set `data_root` in `configs/base.yaml` to the folder containing `CheXpert-v1.0-small/`. On Windows set `PYTHONUTF8=1` (the drivers do).

## Quick check (≈ 6 min, no TEST access)

```bash
bash scripts/run_smoke.sh      # pytest, smoke prepare/extract, sanity, E0–E8 on 1,500 patients, aggregate, assertions
```
Everything goes to `cache/smoke/` and `results/smoke/`, which are never used for reporting.

## Full pipeline

```bash
python scripts/01_prepare.py                # meta + patient-disjoint splits + partition (refuses to overwrite)
python scripts/02_extract.py                # mimic_ch + pc features (~1 h on an RTX 3050), resumable
python scripts/03_sanity.py --set "features.encoders=[nih,mimic_ch,pc]"
bash scripts/run_night.sh                   # trials → DP sweep → frozen config → 3-seed final runs → aggregate → models/v2.1-large/
```
See [docs/NIGHT_RUN.md](docs/NIGHT_RUN.md) for runtime, morning checks and resuming.

## Single runs

```bash
python scripts/run.py --exp E4 --eps 3 --seed 0 [--final] [--resume] [--smoke] [--config f.yaml] [--set dp.lr=0.1] [--fair F1] [--tag name]
```

| exp | what |
|---|---|
| E0 | linear probe on frozen features (central) |
| E1 | central, image-only head |
| E2 | central, configured head (FiLM by default) |
| E3 | FedAvg, no DP |
| E4 | patient-level DP-FedAvg (`--eps`) |
| E5 | image-level DP (patient guarantee weaker; group-privacy bound reported) |
| E7 | E4 + F1 group reweighting (weights ≤ 1) |
| E8 | FL with compressed traffic (int8 up / fp16 down unless `--set comm.up=topk` etc.; DP if `--eps`) |
| E9 | membership-inference audit, written by every run (`mia.json`) |

Every run also writes the F2 per-group-threshold report. TEST is evaluated only with `--final`, once per run, and logged.

## Outputs

`<results>/runs/<tag>__<hash>__s<seed>/`: `manifest.json`, `config.yaml`, `metrics_val.json`, `metrics_test.json` (final only),
`preds_val.parquet` / `preds_test.parquet`, `history.csv`, `fairness.json`, `privacy.json`, `comms.json`, `mia.json`, `calibration.json`, `head.pt`, `DONE`.
`scripts/aggregate.py` → `master_runs.csv`, `master_table.csv/.md`, `figs/*.png`.

## Scale ladder

```bash
bash scripts/run_scale.sh small && bash scripts/run_scale.sh medium     # 2k / 20k patients, 3 seeds each (~3 h together)
python scripts/compare_scales.py                                         # results/scale_comparison.{csv,md,png}
```

## Dashboard

```bash
python scripts/make_demo_set.py      # once: example X-rays per finding from train.csv -> demo/
python scripts/make_demo_folders.py  # once: demo/by_label/<finding>/ with patient info and model predictions
python scripts/make_demo_groups.py   # once: demo/presentation/ (normal vs abnormal, one finding each, multi-finding,
                                     #       private vs non-private, membership pairs, honest misses) + README demo script
python scripts/mia_demo.py           # once: membership-inference stress test -> results/mia_demo/ (~5 min)
python scripts/mia_lira.py stress    # once: LiRA shadow-model attack + DP audit -> results/mia_lira/stress/ (~1 h)
python scripts/mia_lira.py real      # once: LiRA on the full-scale models -> results/mia_lira/real/ (~15 min)
streamlit run dashboard/app.py
```

- **Results**: accuracy by data scale, privacy vs accuracy, traffic saved, fairness gaps, per-finding AUROC.
- **Predict**: pick an example X-ray or upload one, enter patient details, get 14 probabilities and a Grad-CAM map; save the record to
  `demo/patient_records.jsonl` (local only; use de-identified data).
- **Privacy**: membership-inference attack on an overfit model with and without DP, and a live attack on any trained model.

The dashboard opens on the final full-scale study (`results/v2.1-final`) and also shows the scale ladder and the v2 run.
New runs under `results/` appear automatically. The v2 runs (`results/v2-large/`) are shown in the charts but cannot be used for
prediction (their head layout predates the current model).

## Tests

```bash
python -m pytest               # label mapping, split overlap, DP equivalence/clipping/noise, PRV ≤ RDP, equal ε across clients,
                               # compression round trip, results writer refuses overwrite, no weight aliasing, F1 weights ≤ 1, ASL
```

## Notes

- v2 full-data results (single nih encoder) are in `results/v2-large/`; v2.1 writes to `results/v2.1-final/`.
- Earlier federated numbers produced before the weight-aliasing fix are invalid (DECISIONS 14).
