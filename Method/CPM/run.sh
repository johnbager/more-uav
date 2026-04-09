#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SPLIT="${1:-val}"
DATASET_ROOT="${2:-./dataset}"
OUTPUT_FILE="${3:-./outputs/cpm_predictions.json}"

cd "${ROOT_DIR}"
python3 Method/CPM/infer.py --split "${SPLIT}" --dataset_root "${DATASET_ROOT}" --output "${OUTPUT_FILE}"
