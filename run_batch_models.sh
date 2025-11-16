#!/usr/bin/env bash
set -euo pipefail

# Batch-train selected models with 50 epochs and no latency measurement.
# Usage:
#   bash run_batch_models.sh              # CPU
#   bash run_batch_models.sh --gpu        # Use GPU if available
#   EPOCHS=50 bash run_batch_models.sh    # Override epochs (default 50)
#
# Models covered:
#   ickan ickan_light ickan_deep rapid_kan rapid_kan_lite rapid_kan_power

EPOCHS_DEFAULT=50
EPOCHS=${EPOCHS:-$EPOCHS_DEFAULT}
USE_GPU=false

# if [[ ${1:-} == "--gpu" ]]; then
#   USE_GPU=true
# fi

MODELS=(
  # ickan
  # ickan_light
  # ickan_deep
  rapid_kan
  rapid_kan_lite
  rapid_kan_power
)

echo "Starting batch training: ${#MODELS[@]} models, epochs=$EPOCHS, gpu=$USE_GPU"

for MODEL in "${MODELS[@]}"; do
  echo "\n==== Training $MODEL (${EPOCHS} epochs) ===="
  if $USE_GPU; then
    python main.py --model "$MODEL" --epochs "$EPOCHS" --no-latency --gpu | tee -a "logs_${MODEL}.txt"
  else
    python main.py --model "$MODEL" --epochs "$EPOCHS" --no-latency | tee -a "logs_${MODEL}.txt"
  fi
  echo "==== Completed $MODEL ===="
  # Small pause to avoid file handle contention
  sleep 2
done

echo "All trainings completed. Reports and .pth files should be in the repo root."
