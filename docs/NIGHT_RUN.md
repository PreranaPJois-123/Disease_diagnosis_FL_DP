# v2.1-final overnight run

## Before starting (2 minutes)

```bash
bash scripts/run_smoke.sh            # must end with "SMOKE CHECKS PASSED"
```
Check three lines in `results/smoke/logs/`: `pytest.log` says passed, `smoke_checks` shows
"E8 int8 VAL predictions differ from fp32", and `sanity.log` shows the shuffled-label AUROC (expected ≈ 0.5–0.6, well below real labels).

Close the dashboard (`streamlit`) first if it is running: it holds models on the GPU.

## Start

```bash
bash scripts/run_night.sh                                          # Git Bash
# or
powershell -ExecutionPolicy Bypass -File scripts\run_night.ps1     # PowerShell
```
Run it from the repo root, with the laptop on mains power and sleep disabled. `SEEDS="0"` (bash) or `-Seeds 0` (ps1) gives a one-seed night.

## What it does (in order) and estimated time

Timings come from micro-benchmarks on the RTX 3050 4 GB after the v2.1 optimisations (DECISIONS 38): extraction 70-150 img/s for
2 encoders (disk-bound; the GPU does 287 img/s); DP step 0.033-0.045 s (3 encoders, FiLM, B=256, chunk 256; was 0.081 s at chunk 128);
post-training evaluation ~10 s per run (was 40-95 s); σ calibration ~5 s per client, cached afterwards.

| step | runs | est. time |
|---|---|---|
| 1. extract mimic_ch + pc (191k train + 202 test images), resumable; falls back to nih-only on failure | – | 25–50 min |
| 2. sanity gate (VAL only, shuffled-label tolerance 0.10, DECISIONS 37); the night **stops** if it fails | – | ~10 min |
| 3. model trials, central, seed 0, VAL: {nih, all 3} × {film, linear} × {bce_pw, asl} + EMA variants → `frozen.yaml` | 10 | ~40 min |
| 4. DP sweep at ε=3, seed 0, VAL: lr {0.03, 0.1, 0.3} × (C, B) {(1, 256), (0.5, 256), (1, 512)} → `frozen.yaml` | 9 | ~1.3 h |
| 5. final runs, seeds 0–2, TEST once each: E0, E1, E2, E3, E4 ε∈{1,3,8}, E5 ε=8, E7 ε=3 (F1), E8 (E3 int8, ε=3 int8, ε=3 top-k) | 36 | ~3.3 h |
| 6. aggregate → `master_table.md`, figures; `compare_scales`; `export_model` → `models/v2.1-large/` (E4 ε=3, E3, E2, E1, seed 0) | – | 2 min |
| **total** | | **~6–7 h** (range 5–8.5 h; early stopping usually trims the DP runs) |

## In the morning

1. `results/v2.1-final/logs/night.log`: the last line should be `=== night run end. failures: 0 ===`.
2. `results/v2.1-final/logs/failures.txt` should not exist (or should be empty). Each failed run has its log in `logs/runs/<name>.log`.
3. `results/v2.1-final/master_table.md`: the master table (VAL and TEST mean ± sd over seeds, ε PRV/RDP, MB, fairness gaps, F2 gaps, MIA).
4. `results/v2.1-final/frozen.yaml`: which model and DP settings were selected, and their VAL scores.
5. Sanity: in `master_table.md`, the `eps_prv` column must be ≤ target for every E4/E7/E8 row, with `eps_rdp` ≥ `eps_prv`. `saving_pct` should be about 62
   for int8/fp16. `test_access.jsonl` should have exactly one line per final run.
6. Figures: `results/v2.1-final/figs/` (privacy_utility, communication, fairness_gaps, eps_vs_round).
7. `python scripts/compare_scales.py` adds the v2.1 large column to `results/scale_comparison.*`. The dashboard picks up the new runs
   automatically (`streamlit run dashboard/app.py`, scale "large (v2.1)").

## Resume after a crash or reboot

Run the same command again. Finished steps (`logs/*.ok`) and completed runs (`runs/*/DONE`) are skipped. A half-written run is deleted
and redone (`--resume`). Extraction resumes at the first missing chunk. A final run that crashed **after** reading TEST cannot re-read it.
It will fail with `TestAccessDenied`, and you decide what to do with it. This needs a crash in the few seconds between the TEST read and the end of the run.

## Notes

- The night writes only `cache/feat_*_{mimic_ch,pc}.*`, `cache/sigma_cache.json` and `results/v2.1-final/`. Existing v2 caches and results are not touched.
- Out-of-memory failures are retried with smaller `dp.vmap_chunk` (128 → 64 → 32). The run_id does not change, because performance keys are not hashed.
