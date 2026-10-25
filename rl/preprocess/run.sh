#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: bash preprocess/run.sh <raw_dir> <output_dir>" >&2
  exit 1
fi
RAW=$(realpath "$1")
OUT=$(realpath -m "$2")
PYTHON=${PYTHON:-python}
WORKERS=${WORKERS:-32}
IK_TIMEOUT=${IK_TIMEOUT:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
URDF=$ROOT/assets/robot/fr3_sharpa_right.urdf
export PYTHONPATH=$ROOT:$ROOT/preprocess${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
cd "$ROOT"

"$PYTHON" preprocess/quality.py --raw "$RAW" --output "$OUT/source"
"$PYTHON" preprocess/prepare.py --source "$OUT/source" --output "$OUT/short"
"$PYTHON" preprocess/trim_tail.py --bank "$OUT/short"
"$PYTHON" preprocess/wrist_targets.py --bank "$OUT/short"
"$PYTHON" preprocess/placement.py --bank "$OUT/short" --urdf "$URDF" --output "$OUT/short/placement.json"
"$PYTHON" preprocess/solve_ik.py --bank "$OUT/short" --plan "$OUT/short/placement.json" --urdf "$URDF" \
  --work "$OUT/work/ik" --workers "$WORKERS" --timeout "$IK_TIMEOUT"
"$PYTHON" preprocess/filter_bank.py --source "$OUT/short" --output "$OUT/reachable"
"$PYTHON" preprocess/slow_reference.py --source "$OUT/reachable" --output "$OUT/slow"
"$PYTHON" preprocess/smooth_ik.py --bank "$OUT/slow" --urdf "$URDF" --workers "$WORKERS"
"$PYTHON" preprocess/reset_safety.py --bank "$OUT/slow" --output "$OUT/safe" --urdf "$URDF" --workers "$WORKERS"
"$PYTHON" preprocess/retime.py --source "$OUT/safe" --output "$OUT/final"
"$PYTHON" preprocess/finalize.py --bank "$OUT/final" --urdf "$URDF" --workers "$WORKERS"
"$PYTHON" preprocess/collision_models.py --bank "$OUT/final"
