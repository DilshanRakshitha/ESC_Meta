#!/usr/bin/env bash
set -euo pipefail

# Batch-run post-hoc latency measurement for saved models.
# Usage:
#   bash run_latency_batch.sh            # CPU
#   bash run_latency_batch.sh --gpu      # Use GPU if available
#   BATCH_SIZE=32 bash run_latency_batch.sh  # Override batch size
#
# Models measured:
#   ickan, ickan_light, ickan_deep, rapid_kan, rapid_kan_lite, rapid_kan_power

USE_GPU=false
if [[ ${1:-} == "--gpu" ]]; then
  USE_GPU=true
fi

BATCH_SIZE=${BATCH_SIZE:-32}
WARMUP=${WARMUP:-3}
TIMED=${TIMED:-20}

MODELS=(
#   ickan
#   ickan_light
#   ickan_deep
  rapid_kan
  rapid_kan_lite
  rapid_kan_power
)

for MODEL in "${MODELS[@]}"; do
  echo "\n==== Measuring latency for $MODEL (batch=$BATCH_SIZE) ===="
  if $USE_GPU; then
    python -m inference.latency_report --model "$MODEL" --device cuda --batch-size "$BATCH_SIZE" --warmup "$WARMUP" --timed "$TIMED" | tee -a "latency_${MODEL}.txt"
  else
    python -m inference.latency_report --model "$MODEL" --device cpu --batch-size "$BATCH_SIZE" --warmup "$WARMUP" --timed "$TIMED" | tee -a "latency_${MODEL}.txt"
  fi
  echo "==== Completed latency for $MODEL ===="
  sleep 1
done

echo "All latency measurements completed. Check latency_*.txt and <model>.txt/.json reports."
