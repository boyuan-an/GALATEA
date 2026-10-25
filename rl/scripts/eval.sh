#!/usr/bin/env bash
set -euo pipefail
OBJECT=$1
CHECKPOINT=$2
shift 2
export OMP_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py object="$OBJECT" checkpoint="$CHECKPOINT" frame0_eval_only=true experiment="${OBJECT}_eval" \
    task.env.gravityInitialScale=1.0 task.env.tightenSchedule=const "$@"
