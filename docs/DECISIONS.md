# DECISIONS (where the docs were silent or were changed)

1. **DP learning rate / server**: docs default (SGD lr 1.0, FedAvgM) diverged on the small setup (TEST mean-5 ~0.60). A VAL-only sweep
   over lr {0.1, 0.3} x {fedavg, fedavgm} chose `dp.lr=0.1`, FedAvgM. Tuning on VAL consumes privacy budget (standard limitation, not accounted).
2. **Shuffled-label control (L2)**: one epoch, mean of 3 shuffles, no VAL-based epoch selection (selection on a noisy VAL biases it above 0.5).
   Tolerance is config `sanity.shuffle_tol` (0.05 small / 0.02 full).
3. **Small experiment**: `configs/small.yaml` = 4,000 random train patients, K=5, min 150 patients/client, 8 rounds x 20 steps, batch 128.
4. **Best-round epsilon**: reported epsilon is the max over clients at the VAL-selected round (<= target).
5. **Central/FL training set** = POOL minus each client's local-val patients, so E1-E4 see identical data.
6. **pos_weight** from global train statistics (docs/04 allows; not strict FL).
7. **Image-level DP (E5)**: patient-level guarantee reported as a loose group-privacy bound k_max * eps (`info.image_level_worst_patient_eps_bound`).
8. **E6** re-partitions in memory for the requested alpha (same seed); split/val/test unchanged.
9. **Fairness** is measured on VAL (mean comp-5 AUROC per group, min 5 positives per label); per-client audit on each client's local-val.
10. **MIA (E9)**: per-patient mean loss, members = training patients (subsampled to |VAL patients|) vs VAL patients; AUC ~0.5 = no leakage signal.
11. Windows: set `PYTHONUTF8=1` for the first xrv weight download (progress bar crashes under cp1252).
12. **L2 gate revised at full scale**: shuffled-label VAL AUROC was 0.55 (not 0.50 +- 0.02). Diagnosis: untrained random heads already score 0.44-0.50 and
    a single label-agnostic score (mean/L2-norm of the 1024-d feature) gives AUROC 0.36-0.42 on comp-5 (i.e. 0.58-0.64 flipped), so any head trained a little
    on noise drifts along that generic direction. Not an alignment bug (E0 = 0.83, real-label 1-epoch = 0.73). Gate is now |shuffled-0.5| <= 0.07
    AND real - shuffled >= 0.10 under the same one-epoch protocol.
13. **Full-scale DP lr**: VAL sweep (seed 0, eps 8) lr 0.03 / 0.1 / 0.3 -> VAL mean-5 0.7165 / 0.7128 / 0.6991; chose `dp.lr=0.03` in base.yaml (raw runs in results_sweep/).
14. **Weight-aliasing bug (fixed)**: `vector_to_parameters(vec, params)` makes parameters views of `vec`, so local optimizer steps mutated the server's
    global vector. Clients trained sequentially and updates were double counted in every FL run (E3-E6, extensions). Found when fp16 downlink (a copy)
    scored lower than fp32 (aliased). Fix: `load_vec` in `train/server.py` clones first; regression test `test_load_vec_does_not_alias`.
    All FL runs regenerated; invalid ones kept in `results_aliasing_bug/`. Central runs unaffected. DP lr (0.03) was chosen with the buggy code and not re-swept.
15. **Fairness mitigation**: `--fair` / `fair.reweight` weights each image by the inverse frequency of its age-bin and projection group (global statistics, mean 1,
    clipped to [wmin, wmax] = [0.5, 3]); applied in non-DP AdamW and as a per-image weight inside each patient's DP gradient. Result: no gap reduction (IMPLEMENTATION s3).
16. **Communication compression**: `comm.up/down` in {fp32, fp16, int8}; int8 is symmetric per-block (1024) with fp32 scales; bytes are counted from the encoded payloads.
    The server aggregates decoded client updates; compression after DP noise is post-processing and does not change epsilon.
17. **Resumable runs**: `scripts/run_fl_all.sh` skips finished runs and retries on CUDA OOM (4 GB GPU), with `dp.vmap_chunk=128` for DP runs. Run tags auto-suffix `_fair`, `_up<mode>`, `_dn<mode>`.

## v2.1-final (code freeze for the overnight run)

18. **Missing spec files**: the v2.1 brief cites `docs/11` (s11.6 results layout, s11.8) and `docs/12`, which are not in the repo. The results
    layout is therefore defined in `src/dpmmfl/results.py` (module docstring): `<results>/runs/<tag>__<config-hash>__s<seed>/` + manifest + DONE marker.
19. **Encoders**: TorchXRayVision DenseNet-121 `nih`, `mimic_ch`, `pc` only (`constants.ENCODERS`; extraction refuses any weight name containing
    chex/all/rsna/mimic_nb). nih features are reused from the v2 cache (`cache/feat_<split>.npy`, legacy name); mimic_ch + pc are extracted in one
    pass that decodes each JPEG once. Row order is checked via `feat_<split>_<enc>.rowid.npy`.
20. **Head**: per-encoder LayerNorm -> Linear(1024, 256), concatenated; heads `film` / `image_only` / `linear`; losses `bce_pw` / `asl`;
    optional weight EMA; per-label temperature scaling fit on VAL (monotone, AUROC unchanged; reported ECE uses calibrated probabilities).
21. **DP accounting**: σ is calibrated with PRV (bisection inside the RDP bracket, cached in `<cache>/sigma_cache.json`); ε is reported from both PRV
    and RDP. The reported ε is the PRV ε after the steps **actually taken** (conservative under early stopping), not at the selected round.
    Planned vs actual steps are logged. Per-round history logs RDP only (PRV is ~1 s per call).
22. **DP memory**: per-unit gradients are clipped and summed chunk by chunk (peak memory ~0.7 GB at chunk 128 with 3 encoders, was 2.3 GB).
23. **Early stopping**: VAL every `fl.eval_every`=2 rounds, patience 5 evaluations (central: every epoch, patience 5). VAL use is not privacy-accounted (as before).
24. **Compression**: client -> `encode` -> `serialize` (bytes counted from the real blob) -> `deserialize` -> `decode`; the server aggregates the decoded
    (dequantised) tensors. Modes fp32/fp16/int8/top-k; top-k uses client-side error feedback with residual = (noisy update + residual) - sent.
    Unit test asserts the int8 round trip is lossy and the aggregate differs from the fp32 aggregate. The v2 int8 code was already correct (smoke:
    int8 VAL logits differ from fp32 by up to 0.056, mean 0.006, while VAL AUROC agrees to 4 decimals - a lossless-in-AUROC result, not a no-op).
25. **Fairness**: F1 weights are inverse group frequency scaled so the rarest (>=1% share) group has weight 1, all weights in [0.2, 1] (was mean-1,
    up to 3). F2 (per-group thresholds) is post-processing reported for every run: thresholds fit on VAL half A (by patient), EO gaps on half B.
    E7 = E4 + F1. Audit adds patient-bootstrap CIs per group, worst-group AUROC, EO TPR/FPR gaps and per-group ECE.
26. **TEST discipline**: TEST is read only by `run.py --final`, at most once per run_id (`<results>/test_access.jsonl`), never in `--smoke`
    (raises `TestAccessDenied`). The sanity gate's linear-probe floor now uses VAL (v2 used TEST). Trials and the DP sweep select on VAL only.
27. **Smoke subset**: 1,500 FL-pool + 1,000 VAL patients sampled from the full meta and re-partitioned (min 150 patients/client). Smoke extraction
    covers all smoke rows (~7.2k images) because the runs need features for every row; `--limit 500` is the separate extractor probe.
28. **Shuffled-label control**: smoke gives 0.525 with nih only (0.69 real) but 0.58 with nih+mimic_ch+pc (0.53-0.61 over 3 shuffles; 0.73 real).
    The extra encoders strengthen the label-agnostic direction of item 12. The real-vs-shuffled margin stays large. The smoke tolerance is 0.10.
    The full-scale gate is still 0.07, so at full scale the night stops if shuffled >= 0.57 (see NIGHT_RUN.md).
29. **Performance**: features live on the GPU as float16 [N, E, 1024] (pinned CPU fallback if they would exceed 45% of free VRAM); a client's
    patient -> image table is pre-padded once, so a DP batch is Poisson mask + top-k + one gather; non-DP forward under bf16 autocast, DP in fp32;
    run_id hashes exclude performance-only keys (dp.vmap_chunk, features.on_gpu, extract workers/batch) so OOM retries keep the same run_id.
30. **Repo hygiene**: v2-era logs, superseded drivers (run_all, run_extensions, run_fl_all), small.yaml, cache_small/splits_small/results_small,
    results_sweep and results_aliasing_bug moved to `archive/` (gitignored, safe to delete). v2 results in `results/E*` are untouched.
31. **Scale ladder**: `configs/scale_{small,medium}.yaml` subsample the full cache (2k / 20k FL patients, min 150 / 1000 per client) and run the v2.1
    protocol with 15 rounds × 50 steps on nih features (`scripts/run_scale.sh`). The DP lr is re-swept on VAL per scale. TEST is copied into each
    scale cache and read once per final run. `scripts/compare_scales.py` puts them next to the v2 full-data numbers, labelled as a different protocol.
32. **v2 large results archived**: before the v2.1 overnight run, `results/E*`, `results/summary.csv` and the v2 tables/figures were copied to
    `archive/large_v2_results/` (copy, not move). The v2 head (`img` projection, no mix layer) does not load into the v2.1 `FusionHead`;
    these runs are kept for results only.
33. **Membership-inference stress test** (`scripts/mia_demo.py`): an overfit non-private head vs patient-level DP-SGD on the same 300 patients.
    It is a worst case built to show what DP prevents, not a claim about the main runs, where even non-private models give AUC ≈ 0.5.
    The live attack in the dashboard uses FL-pool rows minus local-val rows as members and VAL as non-members.
34. **Demo images**: `scripts/make_demo_set.py` reads `train.csv` (never `valid.csv`) and copies 56 frontal images to `demo/images/`:
    per finding, 3 VAL-patient images (held out) and 1 FL-pool image, preferring cases with ≤ 3 positive findings (only No Finding for that class).
    `demo/images/` and `demo/patient_records.jsonl` are gitignored (CheXpert images may not be redistributed; records may hold patient details).
35. **Inference path**: `features.extract.preprocess` is the single preprocessing function for extraction and for the dashboard; uploaded images use
    fp32 encoder features (the cache used AMP fp16, difference < 2e-2). Missing metadata is passed as zeros, the input the FiLM head saw under
    modality dropout.
36. **Repo hygiene (2)**: the root copy of `Review2_v1.pdf` (identical to `docs/reference/`) was removed; `docs/IMPLEMENTATION_PROMPT.md` moved to
    `archive/docs/`.
37. **Full-scale sanity tolerance**: the night run passes `--set sanity.shuffle_tol=0.10` to the sanity gate (base.yaml keeps 0.07).
    With three encoders the shuffled-label probe gives 0.53-0.61 in smoke (item 28), so 0.07 would stop the night on a known,
    explained effect. The second condition is unchanged: real-label AUROC must exceed the shuffled one by at least 0.10.
38. **Optimisations before the full run** (no change to any reported quantity):
    - AUROC from sorted scores (Mann-Whitney, ties 1/2) instead of sklearn: identical values (test against sklearn, and stored medium
      results reproduced to 1e-7), ~10x faster. Patient bootstraps are vectorised as row-multiplicity weights (one sort per label).
      Post-training evaluation fell from 40-95 s to ~10 s per run. CIs differ from earlier runs only by bootstrap noise
      (multinomial draws replace per-patient index draws; the resampling distribution is the same).
    - DP clipping without gradient-sized temporaries (fused norms, clipped sum as one contraction) and the clipped fraction kept on
      the GPU (one host sync per local round). Peak extra memory halves, so the default `dp.vmap_chunk` is 256 (OOM retries 128, 64).
      DP step with 3 encoders: 0.042 -> 0.033 s (medium-batch benchmark). Dropping padded image slots was tried and was slower:
      the step is bound by kernel launches, not arithmetic.
    - Extraction is disk-bound (70-150 img/s with 4 or 8 workers against 287 img/s on the GPU); left as is.
39. **Saved models**: `scripts/export_model.py` (last step of the night) copies the seed-0 final E4 ε=3, E3, E2 and E1 models with
    config, calibration, metrics and privacy report to `models/v2.1-large/`; the dashboard lists them as "saved (v2.1-large)".
40. **Stronger membership inference (LiRA) and a DP audit** (`eval/lira.py`, `scripts/mia_lira.py`): the loss-threshold audit (E9) could not
    separate private from non-private full-scale models, so it understated the central model's leakage. Added:
    - patient-level LiRA with shadow models (stress test) and reference models (full-scale models);
    - TPR at 1% and 0.1% FPR;
    - the (ε, δ) ROC bound;
    - an empirical ε lower bound (Clopper–Pearson; threshold chosen on one half of the trials, bound computed on the other).
    The statistic −log(mean patient loss) was chosen on the overfit shadow arm, which an attacker controls. Reference models for the
    full-scale attack are central non-DP models: matched to E2, conservative for E3 and E4. E9 (`mia.json`) and the live dashboard
    attack now also record TPR at 1% FPR for new runs; stored runs keep their files. Results are in `results/mia_lira/` and
    IMPLEMENTATION 5.2b.

## Repository clean-up (2026-11-12)

33. **Unused files removed, layout tidied.** Moved out of the repository (kept in a local backup, not deleted): `main.py` (placeholder),
    `archive/` (byte-identical copy of the v2 results plus superseded small-scale, aliasing-bug and sweep results and old scripts),
    `results/smoke/` and `cache/smoke/` (smoke-test outputs, regenerated by `scripts/run_smoke.sh`), `cache/*_chunks/` (extraction
    chunks, redundant once the merged `feat_*.npy` exist; only read when a merged file is missing), `report/first-ACK.tex` and
    `report/second.tex` (merged into `report/main.tex`), and generated diagrams the report no longer uses. The v2 full-scale results
    moved from `results/E*` to `results/v2-large/`; `serve.py`, the dashboard, `compare_scales.py` and the tests follow the new path.
    `report/make_figures.py` now draws only the flow chart and work plan and copies the corrected result figures from
    `paper/journal/figures` before `paper/figures`. Phase-1 report and survey paper moved to `docs/reference/`.
