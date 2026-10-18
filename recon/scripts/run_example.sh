#!/usr/bin/env bash
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="$REPO/examples/power_drill"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
  echo 'Run all 169 frames with supplied first-frame depth/intrinsics and automatic contact.'
  echo 'Usage: bash scripts/run_example.sh [--gpu 0] [--tag power_drill_auto] [--force]'
  exit 0
fi
source "$REPO/scripts/setup/runtime_env.sh"
galatea_load_conda "$REPO"
export DEPTH_RECTIFY_CAMERA="${DEPTH_RECTIFY_CAMERA:-1}"
export DEPTH_CALIBRATION="${DEPTH_CALIBRATION:-reference}"
"$CONDA_BASE/bin/python" - "$EXAMPLE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for name, expected in json.loads((root / "manifest.json").read_text())["sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Example asset missing or changed: %s. Restore the bundled example files." % path)
print("Bundled example inputs verified (169 frames, 24 FPS).")
PY
exec bash "$REPO/scripts/run_pipeline_custom.sh" \
  --video "$EXAMPLE/video.mp4" --mesh "$EXAMPLE/object.obj" \
  --object-name 'orange pistol-shaped cordless power drill model' \
  --contact-mode auto --tag power_drill_auto --with-qc "$@"
