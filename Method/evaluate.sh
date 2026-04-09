#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT="${1:-val}"
GT_ROOT="${2:-./dataset}"
PRED_FILE="${3:-./outputs/predictions.json}"

cd "${ROOT_DIR}"
python3 Method/metric.py --gt_root "${GT_ROOT}" --split "${SPLIT}" --pred_file "${PRED_FILE}"
