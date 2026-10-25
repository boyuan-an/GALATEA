#!/usr/bin/env bash
set -euo pipefail
OBJECT=${1:-power_drill}
shift || true
export OMP_NUM_THREADS=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=2 train.py object="$OBJECT" multi_gpu=true num_envs=12288 minibatch_size=73728 "$@"
