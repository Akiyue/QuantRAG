#!/usr/bin/env bash
#
# Spread the grid across every GPU, several workers per card.
#
#   bash scripts/run_parallel.sh main          # behavioural grid
#   bash scripts/run_parallel.sh dose          # evidence-pressure ladder
#   bash scripts/run_parallel.sh filter        # C0 known-subset pass
#
#   WORKERS_PER_GPU=3 bash scripts/run_parallel.sh main
#
# Each (model, precision) is an independent job writing its own JSONL, so
# sharding needs no coordination and a worker that dies just leaves its shard
# unfinished - rerun and the resume logic picks it up.
#
# These models are small enough that one alone leaves a 32 GB card mostly idle:
# a 1.5B at F16 is about 3 GB, and a single stream is latency-bound rather than
# throughput-bound. Two or three workers per card is usually the sweet spot.
# Past that they contend and the wall clock stops improving.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS="${1:-main}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
MODELS="${MODELS:-qwen2.5-0.5b qwen2.5-1.5b qwen2.5-3b}"
PRECISIONS="${PRECISIONS:-F16 Q8_0 Q4_K_M Q3_K_M}"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

PY="${PY:-python}"

N_GPU="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
[[ "$N_GPU" -gt 0 ]] || { echo "no GPUs detected"; exit 1; }

# Build the job list, largest models first so the long poles start immediately
# instead of trailing at the end.
JOBS=()
for m in $(echo "$MODELS" | tr ' ' '\n' | tac); do
  for p in $PRECISIONS; do
    JOBS+=("$m|$p")
  done
done

TOTAL_WORKERS=$(( N_GPU * WORKERS_PER_GPU ))
echo "pass       : $PASS"
echo "gpus       : $N_GPU"
echo "workers    : $TOTAL_WORKERS ($WORKERS_PER_GPU per gpu)"
echo "jobs       : ${#JOBS[@]}"
echo

stamp="$(date +%Y%m%d-%H%M%S)"
pids=()

for i in "${!JOBS[@]}"; do
  IFS='|' read -r model precision <<< "${JOBS[$i]}"
  gpu=$(( i % N_GPU ))
  logfile="$LOGS/${PASS}__${model}__${precision}__${stamp}.log"

  echo "  gpu$gpu  $model $precision"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" scripts/run_grid.py --pass "$PASS" \
      --models "$model" --precisions "$precision" \
      > "$logfile" 2>&1 &
  pids+=($!)

  # Throttle to the worker budget rather than launching every job at once.
  while [[ "$(jobs -rp | wc -l)" -ge "$TOTAL_WORKERS" ]]; do
    sleep 5
  done
done

echo
echo "waiting for ${#pids[@]} jobs; logs in $LOGS/"
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done

echo
if [[ "$failed" -eq 0 ]]; then
  echo "all jobs finished"
else
  echo "$failed job(s) failed - check the logs, then rerun this script."
  echo "Completed cells are skipped, so a rerun only redoes what is missing."
  exit 1
fi
