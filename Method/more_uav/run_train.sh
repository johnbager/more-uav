#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${1:-./dataset}"
OUTPUT_DIR="${2:-./checkpoints/more_uav_qwen}"

cd "${ROOT_DIR}"
python3 Method/more_uav/train.py --dataset_root "${DATASET_ROOT}" --output_dir "${OUTPUT_DIR}"
