#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="${1:-scene_3}"
if (( $# > 0 )); then shift; fi
: "${DATA_ROOT:?Set DATA_ROOT to the selected 3D-FRONT-T scene directory}"
cd "$REPO_ROOT"
# 6,000 updates: interface [0,1000), shared [1000,3000), then transmission.
# Separate reflection stays disabled even when resuming beyond this experiment.
python -m glint.train \
  --config "configs/exps/glint/3d-front-t/${SCENE}.yaml" \
  --data-root "$DATA_ROOT" \
  --output-dir "${OUTPUT_DIR:-results/shared-reflection-first/${SCENE}}" \
  --max-steps 6000 \
  --shared-reflection-start 1000 \
  --transmission-start 3000 \
  --reflection-start 6000 \
  --no-separate-reflection \
  --save-every 1000 \
  --image-every 1000 \
  --source-surface-filter \
  "$@"
