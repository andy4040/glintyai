#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the 3D-FRONT-T scene_3 directory}"
cd "$REPO_ROOT"
python -m glint.train \
  --config configs/exps/glint/3d-front-t/scene_3_no_trans_aux_2x.yaml \
  --data-root "$DATA_ROOT" \
  --output-dir "${OUTPUT_DIR:-results/no-trans-aux-2x/scene_3}" \
  --max-steps 18000 \
  --no-separate-reflection --source-surface-filter --source-normal-cosine 0.9 \
  --save-every 2000 --image-every 2000 \
  "$@"
