#!/usr/bin/env bash
# One rung of the scale ladder: DP lr sweep (VAL only) -> freeze -> final runs (3 seeds, TEST once each) -> aggregate.
# Resumable. Usage: bash scripts/run_scale.sh small|medium     (SEEDS="0 1 2" by default)
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-python}
SC=$1
CFG=configs/scale_$SC.yaml
ROOT=results/scale_$SC
LOG=$ROOT/logs
mkdir -p "$LOG"
SEEDS=${SEEDS:-"0 1 2"}
note() { echo "$(date '+%F %T') [$SC] $*" | tee -a "$LOG/scale.log"; }
run() {   # run <name> <run.py args...>
  local name=$1; shift
  for chunk in 256 128 64; do
    $PY scripts/run.py --config $CFG --resume "$@" --set dp.vmap_chunk=$chunk >> "$LOG/$name.log" 2>&1 && return 0
  done
  note "FAILED $name"; echo "$name" >> "$LOG/failures.txt"
}
note "start"
[ -f cache/scale_$SC/meta_train.parquet ] || $PY scripts/01_prepare.py --config $CFG > "$LOG/prepare.log" 2>&1
for lr in 0.03 0.1 0.3; do run sweep_lr$lr --exp E4 --eps 3 --seed 0 --tag dpsweep_lr$lr --set dp.lr=$lr; done
$PY scripts/freeze.py --config $CFG --stage dp > "$LOG/freeze.log" 2>&1; note "$(tail -1 $LOG/freeze.log)"
F="--config $ROOT/frozen.yaml --final"
for s in $SEEDS; do
  run E0_s$s --exp E0 --seed $s $F
  run E1_s$s --exp E1 --seed $s $F
  run E2_s$s --exp E2 --seed $s $F
  run E3_s$s --exp E3 --seed $s $F
  for e in 1 3 8; do run E4_eps${e}_s$s --exp E4 --eps $e --seed $s $F; done
  run E5_eps8_s$s --exp E5 --eps 8 --seed $s $F
  run E7_eps3_s$s --exp E7 --eps 3 --seed $s $F
  run E8_E3_int8_s$s --exp E8 --seed $s $F
  note "seed $s done"
done
$PY scripts/aggregate.py --config $CFG > "$LOG/aggregate.log" 2>&1
note "finished"
