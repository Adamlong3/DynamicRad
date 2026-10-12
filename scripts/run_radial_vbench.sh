#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m scripts.inference_wan \
  --prompt "${PROMPT:-A fast-moving car crossing a city street}" \
  --pattern radial \
  --topk_mode dynamic_threshold \
  --height 768 \
  --width 1280 \
  --num_frames 69 \
  --block_size 32 \
  --output_file "${OUTPUT_FILE:-outputs/wan_dynamic.mp4}" \
  "$@"
