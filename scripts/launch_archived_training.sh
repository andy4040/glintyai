#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_DIR:?Set a NEW OUTPUT_DIR}"
: "${RELEASE_TAG:?Set a unique RELEASE_TAG}"
for arg in "$@"; do
  case "$arg" in --output-dir|--output-dir=*|--config|--config=*)
    echo "Use OUTPUT_DIR for the run path; this launcher uses the material-first config." >&2; exit 2;;
  esac
done
python -m glint.archive_run --check-auth
mkdir -p "$OUTPUT_DIR"
nohup python -u -m glint.archive_run \
  --tag "$RELEASE_TAG" --run-dir "$OUTPUT_DIR" \
  --config configs/exps/glint/3d-front-t/scene_3_material_first.yaml \
  --data-root "$DATA_ROOT" \
  --train bash scripts/train_material_first.sh "$@" \
  > "$OUTPUT_DIR/server-job.log" 2>&1 < /dev/null &
echo "$!" > "$OUTPUT_DIR/server-job.pid"
echo "Started server job $(cat "$OUTPUT_DIR/server-job.pid"); log: $OUTPUT_DIR/server-job.log"
