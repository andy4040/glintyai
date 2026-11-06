#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the 3D-FRONT-T scene_3 directory}"
cd "$REPO_ROOT"
python -m glint.train \
  --config configs/exps/glint/3d-front-t/scene_3_material_first.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "${OUTPUT_DIR:-results/material-first/scene_3}" \
  --max-steps 9000 \
  --no-separate-reflection \
  --source-surface-filter \
  --source-normal-cosine 0.9 \
  --save-every 1000 \
  --image-every 1000 \
  "$@"
