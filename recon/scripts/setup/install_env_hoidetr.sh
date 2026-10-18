#!/usr/bin/env bash
set -eo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
REQUESTED_ARCH="${TORCH_CUDA_ARCH_LIST:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOIDETR_ENV="${HOIDETR_ENV:-codetr-cu121}"
HOIDETR_ROOT="${HOIDETR_ROOT:-$REPO/3rd/HOI-DETR-cu121}"
MMCV_ROOT="${HOIDETR_MMCV_ROOT:-$REPO/imports/mmcv-hoidetr-cu121}"
HOIDETR_REVISION=1b367292f3833afd64a204bd4d9d84519541d035
MMCV_REVISION=4c01b026f0afa5a91a5f54aea313788da1e40f95
SKIP_WEIGHTS=false; CHECK_ONLY=false; CHECK_GPU=false; SKIP_CACHE_REDIRECT=false; USE_CUDA128=false
for arg in "$@"; do case "$arg" in
  --skip-weights) SKIP_WEIGHTS=true ;;
  --cuda128) USE_CUDA128=true ;;
  --skip-cache-redirect) SKIP_CACHE_REDIRECT=true ;;
  --check-only) CHECK_ONLY=true ;;
  --gpu) CHECK_GPU=true ;;
  -h|--help)
    echo 'Install automatic contact detection on Linux x86_64; CUDA 12.1 or --cuda128.'
    echo 'Usage: bash scripts/setup/install_env_hoidetr.sh [--cuda128] [--skip-weights] [--check-only] [--gpu]'
    echo "Defaults: dependencies only; obtain the author's weights using docs/model-assets.md."
    exit 0 ;;
  *) echo "Unknown argument: $arg" >&2; exit 1 ;;
esac; done
source "$REPO/scripts/setup/runtime_env.sh"
galatea_load_conda "$REPO"
CONSTRAINTS="$REPO/scripts/setup/hoidetr-constraints.txt"
CUDA_VERSION=12.1; TORCH_VERSION=2.4.1; TV_VERSION=0.19.1; CUDA_TAG=cu121
if [ "$USE_CUDA128" = true ]; then
  CUDA_VERSION=12.8; TORCH_VERSION=2.7.1; TV_VERSION=0.22.1; CUDA_TAG=cu128
  CONSTRAINTS="$REPO/scripts/setup/hoidetr-cu128-constraints.txt"
fi

if [ "$CHECK_ONLY" = false ]; then
  [ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || { echo 'This installer supports Linux x86_64; see docs/hoidetr-setup.md for ARM.' >&2; exit 1; }
  if [ "$SKIP_CACHE_REDIRECT" = false ]; then
    source "$REPO/scripts/setup/project_cache.sh"
    galatea_project_cache "${CACHE_ROOT:-$REPO/.cache}"
  fi
  source "$REPO/scripts/setup/source_checkout.sh"
  galatea_checkout_source https://github.com/AhmadDarKhalil/HOI-DETR.git "$HOIDETR_ROOT" "$HOIDETR_REVISION" checkpoints
  galatea_checkout_source https://github.com/open-mmlab/mmcv.git "$MMCV_ROOT" "$MMCV_REVISION"
  HOIDETR_PREFIX="$(galatea_env_prefix "$HOIDETR_ENV")"
  galatea_ensure_env "$HOIDETR_PREFIX" 3.10
  conda install -y --prefix "$HOIDETR_PREFIX" --override-channels -c conda-forge gcc_linux-64=12 gxx_linux-64=12 \
    "cuda-nvcc=$CUDA_VERSION" "cuda-cudart-dev=$CUDA_VERSION" "cuda-cccl=$CUDA_VERSION" "cuda-nvrtc-dev=$CUDA_VERSION"
  galatea_activate "$HOIDETR_ENV"
  python -m pip install 'pip==25.0.1' 'setuptools==69.5.1' 'wheel==0.45.1' 'ninja==1.11.1.1'
  python -m pip install -c "$CONSTRAINTS" "torch==$TORCH_VERSION+$CUDA_TAG" "torchvision==$TV_VERSION+$CUDA_TAG" \
    --extra-index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  python -m pip install -c "$CONSTRAINTS" -r "$HOIDETR_ROOT/requirements/runtime.txt" \
    opencv-python huggingface-hub addict packaging Pillow PyYAML
  ARCH="$(python "$REPO/scripts/setup/cuda_arch.py" --arch "$REQUESTED_ARCH")"
  echo "Building MMCV 1.7.2 for CUDA architectures: $ARCH"
  (
    source "$REPO/scripts/setup/build_env.sh"
    galatea_build_env "$CONDA_PREFIX"
    export TORCH_CUDA_ARCH_LIST="$ARCH" MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-4}"
    python -m pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall "$MMCV_ROOT"
  )
  python "$REPO/scripts/setup/patch_hoidetr_compat.py"
  python -m pip install --no-build-isolation --no-deps -e "$HOIDETR_ROOT"
  SKIP_WEIGHTS=true
  echo "Dependencies installed. Obtain epoch_5.pth from the author; see docs/model-assets.md."
else
  galatea_activate "$HOIDETR_ENV"
fi
CHECK_ARGS=(--root "$HOIDETR_ROOT")
[ "$SKIP_WEIGHTS" = true ] && CHECK_ARGS+=(--skip-weights)
[ "$CHECK_GPU" = true ] && CHECK_ARGS+=(--gpu)
python "$REPO/scripts/setup/check_hoidetr.py" "${CHECK_ARGS[@]}"
