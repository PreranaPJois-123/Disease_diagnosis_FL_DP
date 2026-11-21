#!/usr/bin/env bash
# v2.1-final overnight plan. Resumable: completed steps/runs are skipped on restart. A failed run is logged, not fatal.
#   bash scripts/run_night.sh            (from the repo root; logs in results/v2.1-final/logs/)
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-python}
ROOT=results/v2.1-final
LOG=$ROOT/logs
mkdir -p "$LOG"
FAILS=$LOG/failures.txt
SEEDS=${SEEDS:-"0 1 2"}
note() { echo "$(date '+%F %T') $*" | tee -a "$LOG/night.log"; }

step() {   # step <name> <cmd...> : once-only step with a .ok marker
  local name=$1; shift
  [ -f "$LOG/$name.ok" ] && { note "skip $name (done)"; return 0; }
  note "start $name"
  if "$@" > "$LOG/$name.log" 2>&1; then touch "$LOG/$name.ok"; note "ok $name"; return 0; fi
  note "FAILED $name (see $LOG/$name.log)"; echo "$name" >> "$FAILS"; return 1
}

run() {    # run <name> <run.py args...> : resumable, 3 attempts (2nd/3rd with smaller vmap chunks 128, 64 after a GPU OOM)
  local name=$1; shift
  local chunk
  for chunk in 256 128 64; do
    if $PY scripts/run.py --resume "$@" --set dp.vmap_chunk=$chunk >> "$LOG/runs/$name.log" 2>&1; then return 0; fi
    note "retry $name (attempt failed, vmap_chunk=$chunk)"
  done
  note "FAILED run $name"; echo "run $name" >> "$FAILS"
}
mkdir -p "$LOG/runs"
note "=== night run start ==="

# 1. features (resumable). Fall back to nih-only if extraction fails.
ENCS_ALL="[nih,mimic_ch,pc]"
if ! step extract $PY scripts/02_extract.py; then ENCS_ALL="[nih]"; note "extraction failed -> nih-only fallback"; fi

# 2. sanity gate (stop on failure). Shuffled-label tolerance 0.10 with three encoders (DECISIONS 28, 37); the 0.10 real-vs-shuffled margin still applies
step sanity $PY scripts/03_sanity.py --set "features.encoders=$ENCS_ALL" --set sanity.shuffle_tol=0.10 || { note "SANITY FAILED - stopping"; exit 1; }

# 3. model trials (central, seed 0, VAL only) -> frozen model config
for enc in "[nih]" "$ENCS_ALL"; do
  en=$( [ "$enc" = "[nih]" ] && echo nih || echo all )
  for head in film linear; do for loss in bce_pw asl; do
    run trial_${en}_${head}_${loss} --exp E2 --seed 0 --tag trial_${en}_${head}_${loss} \
      --set "features.encoders=$enc" --set model.head=$head --set model.loss=$loss
  done; done
  run trial_${en}_film_bce_pw_ema --exp E2 --seed 0 --tag trial_${en}_film_bce_pw_ema \
    --set "features.encoders=$enc" --set model.head=film --set model.loss=bce_pw --set model.ema=0.999
done
step freeze_model $PY scripts/freeze.py --stage model || exit 1
FROZEN=$ROOT/frozen.yaml

# 4. DP sweep at eps=3 (seed 0, VAL only) -> frozen DP config
for lr in 0.03 0.1 0.3; do for cb in "1.0 256" "0.5 256" "1.0 512"; do
  set -- $cb
  run dpsweep_lr${lr}_C$1_B$2 --exp E4 --eps 3 --seed 0 --config $FROZEN --tag dpsweep_lr${lr}_C$1_B$2 \
    --set dp.lr=$lr --set dp.clip=$1 --set dp.batch=$2
done; done
step freeze_dp $PY scripts/freeze.py --stage dp || exit 1

# 5. final runs: 3 seeds, TEST evaluated once per run (E9 MIA + F2 report are written by every run)
F="--config $FROZEN --final"
for s in $SEEDS; do
  run E0_s$s --exp E0 --seed $s $F
  run E1_s$s --exp E1 --seed $s $F
  run E2_s$s --exp E2 --seed $s $F
  run E3_s$s --exp E3 --seed $s $F
  for e in 1 3 8; do run E4_eps${e}_s$s --exp E4 --eps $e --seed $s $F; done
  run E5_eps8_s$s --exp E5 --eps 8 --seed $s $F
  run E7_eps3_s$s --exp E7 --eps 3 --seed $s $F
  run E8_E3_int8_s$s --exp E8 --seed $s $F
  run E8_eps3_int8_s$s --exp E8 --eps 3 --seed $s $F
  run E8_eps3_topk_s$s --exp E8 --eps 3 --seed $s $F --set comm.up=topk --set comm.down=fp16
done

# 6. tables + figures, scale comparison, demo models
step aggregate $PY scripts/aggregate.py
step compare_scales $PY scripts/compare_scales.py
step export_model $PY scripts/export_model.py
note "=== night run end. failures: $( [ -f "$FAILS" ] && wc -l < "$FAILS" || echo 0 ) ==="
