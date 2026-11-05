#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="${1:-scene_4}"
if (( $# > 0 )); then shift; fi
: "${DATA_ROOT:?Set DATA_ROOT to the selected 3D-FRONT-T scene directory}"
cd "$REPO_ROOT"
python -m glint.train \
  --config "configs/exps/glint/3d-front-t/${SCENE}.yaml" \
  --data-root "$DATA_ROOT" \
  --output-dir "${OUTPUT_DIR:-results/shared-reflection/${SCENE}}" \
  --max-steps 40000 \
  --interface-geometry-freeze 20000 \
  --transmission-start 1000 \
  --shared-reflection-start 3000 \
  --reflection-start 6000 \
  --source-surface-filter \
  "$@"
