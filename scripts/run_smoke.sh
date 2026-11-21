#!/usr/bin/env bash
# Small-scale end-to-end check (minutes). Writes only to cache/smoke and results/smoke; TEST is never touched.
#   bash scripts/run_smoke.sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-python}
LOG=results/smoke/logs
mkdir -p "$LOG"
t() { local name=$1; shift; local s=$(date +%s); "$@" > "$LOG/$name.log" 2>&1; echo "$name $(( $(date +%s) - s ))s" | tee -a "$LOG/timings.txt"; }
: > "$LOG/timings.txt"

t pytest $PY -m pytest
[ -f cache/smoke/meta_train.parquet ] || t prepare $PY scripts/01_prepare.py --smoke
t extract_probe $PY scripts/02_extract.py --smoke --limit 500
t extract $PY scripts/02_extract.py --smoke
t sanity $PY scripts/03_sanity.py --smoke
R="$PY scripts/run.py --smoke --seed 0 --resume"
t E0 $R --exp E0
t E1 $R --exp E1
t E2_film_asl $R --exp E2 --set model.head=film --set model.loss=asl --tag film_asl
t E3 $R --exp E3
t E4_eps3 $R --exp E4 --eps 3
t E8_int8 $R --exp E8
t E8_eps3_topk $R --exp E8 --eps 3 --set comm.up=topk --set comm.down=fp16
t E7_eps3_F1 $R --exp E7 --eps 3
t aggregate $PY scripts/aggregate.py --smoke
$PY scripts/smoke_checks.py
