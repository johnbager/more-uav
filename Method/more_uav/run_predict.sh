#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${1:-./dataset}"
SPLIT="${2:-val}"
CHECKPOINT_DIR="${3:-./checkpoints/more_uav_qwen/best}"
OUTPUT_FILE="${4:-./outputs/more_uav_val_predictions.json}"

cd "${ROOT_DIR}"
python3 Method/more_uav/predict.py --dataset_root "${DATASET_ROOT}" --split "${SPLIT}" --checkpoint_dir "${CHECKPOINT_DIR}" --output "${OUTPUT_FILE}"
