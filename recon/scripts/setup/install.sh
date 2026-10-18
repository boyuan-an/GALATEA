#!/usr/bin/env bash
set -eo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; exit 1; }
section() { echo -e "\n${CYAN}== $1 ==${NC}"; }
usage() {
  cat <<'EOF'
GALATEA reconstruction environment installer.

Builds the three conda environments used by the reconstruction pipeline.
Package and model caches stay under this checkout's .cache/ (override CACHE_ROOT).

  sam3d-objects  py3.11  torch 2.5.1+cu121  depth, segmentation, hands, optimization, rendering
  factory        py3.8   torch 2.4.1+cu121  FoundationPose object tracking
  codetr-cu121   py3.10  torch 2.4.1+cu121  HOI-DETR automatic contact windows

Re-running skips completed work. pytorch3d, nvdiffrast and the FoundationPose
extensions are compiled against an in-environment CUDA 12.1 nvcc and conda gcc-12;
--cuda128 selects torch 2.7.1 and Python 3.10 for factory. No system CUDA, Eigen
or compiler is required. Build targets: TORCH_CUDA_ARCH_LIST or the visible GPU.

Usage: bash scripts/setup/install.sh [--skip-cache-redirect] [--skip-hoidetr] [--cuda128]

Installs dependencies only. Obtain model weights from the original providers
(docs/model-assets.md), then run scripts/setup/check_install.sh.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

REQUESTED_ARCH="${TORCH_CUDA_ARCH_LIST:-}"
export PYTHONNOUSERSITE=1
[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || fail "This installer supports Linux x86_64."
SKIP_CACHE_REDIRECT=false; SKIP_HOIDETR=false; USE_CUDA128=false
for a in "$@"; do case "$a" in
  --skip-cache-redirect) SKIP_CACHE_REDIRECT=true ;;
  --skip-hoidetr)        SKIP_HOIDETR=true ;;
  --cuda128)             USE_CUDA128=true ;;
  -h|--help) usage; exit 0 ;;
  *) fail "unknown option: $a" ;;
esac; done

source "$PROJECT_ROOT/scripts/setup/runtime_env.sh"
galatea_load_conda "$PROJECT_ROOT"
CUDA_VERSION=12.1
MAIN_TORCH=2.5.1; MAIN_TV=0.20.1
FACTORY_TORCH=2.4.1; FACTORY_TV=0.19.1; FACTORY_PYTHON=3.8
CUDA_TAG=cu121
if [ "$USE_CUDA128" = true ]; then
  CUDA_VERSION=12.8; CUDA_TAG=cu128
  MAIN_TORCH=2.7.1; MAIN_TV=0.22.1
  FACTORY_TORCH=2.7.1; FACTORY_TV=0.22.1; FACTORY_PYTHON=3.10
fi

CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/.cache}"
source "$PROJECT_ROOT/scripts/setup/source-revisions.sh"
source "$PROJECT_ROOT/scripts/setup/build_env.sh"

if [ "$SKIP_CACHE_REDIRECT" = false ]; then
  section "Cache/temp redirect -> $CACHE_ROOT"
  source "$PROJECT_ROOT/scripts/setup/project_cache.sh"
  galatea_project_cache "$CACHE_ROOT"
  ok "caches -> $CACHE_ROOT ; pip cache: $(pip cache dir 2>/dev/null)"
fi

section "Third-party sources -> imports/"
mkdir -p imports
source "$PROJECT_ROOT/scripts/setup/source_checkout.sh"
galatea_checkout_source https://github.com/microsoft/MoGe.git imports/MoGe "$MOGE_REVISION"
galatea_checkout_source https://github.com/NVlabs/FoundationPose.git imports/FoundationPose "$FP_REVISION" weights
ok "Pinned MoGe + FoundationPose sources ready"

build_env() {
  galatea_build_env "$1"
  export TORCH_CUDA_ARCH_LIST="$ARCH" FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-8}"
}
PT3D="git+https://github.com/facebookresearch/pytorch3d.git@$PT3D_REVISION"

section "sam3d-objects env"
SAM3D="$(galatea_env_prefix sam3d-objects)"
galatea_ensure_env "$SAM3D" 3.11
conda install -y --prefix "$SAM3D" --override-channels -c conda-forge gcc_linux-64=12 gxx_linux-64=12 \
  libboost-devel=1.84 libboost=1.84 eigen=3.4 cmake=3.26.4 \
  "cuda-nvcc=$CUDA_VERSION" "cuda-cudart-dev=$CUDA_VERSION" "cuda-cccl=$CUDA_VERSION" "cuda-nvrtc-dev=$CUDA_VERSION"
conda activate "$SAM3D"
SAM3D="$CONDA_PREFIX"
python -m pip install 'pip==25.0.1' 'setuptools==75.3.4' 'wheel==0.45.1'
python -m pip install "torch==$MAIN_TORCH+$CUDA_TAG" "torchvision==$MAIN_TV+$CUDA_TAG" --extra-index-url "https://download.pytorch.org/whl/$CUDA_TAG"
ARCH="$(python "$PROJECT_ROOT/scripts/setup/cuda_arch.py" --arch "$REQUESTED_ARCH")"
ok "CUDA build targets: $ARCH"
printf 'torch==%s\ntorchvision==%s\n' "$MAIN_TORCH" "$MAIN_TV" > "$SAM3D/galatea-torch-constraints.txt"
cat scripts/setup/main-requirements.txt >> "$SAM3D/galatea-torch-constraints.txt"
export PIP_CONSTRAINT="$SAM3D/galatea-torch-constraints.txt"
( build_env "$SAM3D"; python -m pip install --no-build-isolation -r scripts/setup/main-requirements.txt )
python -m pip install "git+https://github.com/warmshao/WiLoR-mini.git@$WILOR_REVISION" --no-deps
python -m pip install -q chumpy==0.70 --no-build-isolation
CH=$(python -c "import importlib.util as u,os;s=u.find_spec('chumpy');print(os.path.dirname(s.origin) if s else '')")
if [ -n "$CH" ]; then
  sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' "$CH/__init__.py" 2>/dev/null || true
  sed -i 's/inspect\.getargspec/inspect.getfullargspec/g' "$CH/ch.py" 2>/dev/null || true
fi
python -m pip install -q -e imports/MoGe --no-deps
python -m pip install -q "git+https://github.com/EasternJournalist/utils3d.git@$UTILS3D_REVISION"
python -m pip install -q -e . --no-deps
if ! python -c "import torch, pytorch3d._C" 2>/dev/null; then
  ( build_env "$SAM3D"; python -m pip install "$PT3D" --no-build-isolation --no-deps )
fi
galatea_write_build_hooks "$SAM3D" "$PROJECT_ROOT/scripts/setup/build_env.sh"
python "$PROJECT_ROOT/scripts/setup/check_environment.py" main --skip-weights
unset PIP_CONSTRAINT
conda deactivate

section "factory env + FoundationPose"
FACTORY="$(galatea_env_prefix factory)"
galatea_ensure_env "$FACTORY" "$FACTORY_PYTHON"
conda install -y --prefix "$FACTORY" --override-channels -c conda-forge gcc_linux-64=12 gxx_linux-64=12 \
  libboost-devel=1.84 libboost=1.84 eigen=3.4 cmake=3.26.4 libxcrypt \
  "cuda-nvcc=$CUDA_VERSION" "cuda-cudart-dev=$CUDA_VERSION" "cuda-cccl=$CUDA_VERSION" "cuda-nvrtc-dev=$CUDA_VERSION"
conda activate "$FACTORY"
FACTORY="$CONDA_PREFIX"
python -m pip install 'pip==25.0.1' 'setuptools==75.3.4' 'wheel==0.45.1'
python -m pip install "torch==$FACTORY_TORCH+$CUDA_TAG" "torchvision==$FACTORY_TV+$CUDA_TAG" --extra-index-url "https://download.pytorch.org/whl/$CUDA_TAG"
printf 'torch==%s\ntorchvision==%s\n' "$FACTORY_TORCH" "$FACTORY_TV" > "$FACTORY/galatea-torch-constraints.txt"
cat scripts/setup/factory-requirements.txt >> "$FACTORY/galatea-torch-constraints.txt"
export PIP_CONSTRAINT="$FACTORY/galatea-torch-constraints.txt"
( build_env "$FACTORY"; python -m pip install --no-build-isolation -r scripts/setup/factory-requirements.txt )
python -c "import torch, pytorch3d._C" 2>/dev/null || ( build_env "$FACTORY"; python -m pip install "$PT3D" --no-build-isolation --no-deps )
python -c "import nvdiffrast" 2>/dev/null || ( build_env "$FACTORY"; python -m pip install "git+https://github.com/NVlabs/nvdiffrast.git@$NVDIFFRAST_REVISION" --no-build-isolation )
( build_env "$FACTORY"
  cmake --fresh -G Ninja -S imports/FoundationPose/mycpp -B imports/FoundationPose/mycpp/build \
    -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=ON \
    -DCMAKE_PREFIX_PATH="$FACTORY;$(python -m pybind11 --cmakedir)" \
    -DPython_ROOT_DIR="$FACTORY" -DPython_EXECUTABLE="$FACTORY/bin/python" \
    -DEigen3_DIR="$FACTORY/share/eigen3/cmake"
  cmake --build imports/FoundationPose/mycpp/build -j"${MAX_JOBS:-8}" )
if ! python - "$PROJECT_ROOT/imports/FoundationPose" <<'PY' 2>/dev/null
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root), str(root / "bundlesdf/mycuda")]
from bundlesdf.mycuda import common
import gridencoder
assert root in Path(common.__file__).resolve().parents
assert root in Path(gridencoder.__file__).resolve().parents
PY
then
  ( build_env "$FACTORY"
    cd imports/FoundationPose/bundlesdf/mycuda
    sed -i 's/-std=c++14/-std=c++17/g' setup.py
    sed -i 's/\.type()/\.scalar_type()/g' common.cu
    rm -rf build *.egg-info *.so
    python -m pip install -e . --no-build-isolation )
fi
galatea_write_build_hooks "$FACTORY" "$PROJECT_ROOT/scripts/setup/build_env.sh"
python "$PROJECT_ROOT/scripts/setup/check_environment.py" factory --skip-weights
unset PIP_CONSTRAINT

conda deactivate
if [ "$SKIP_HOIDETR" = false ]; then
  HOIDETR_ARGS=()
  [ "$USE_CUDA128" = true ] && HOIDETR_ARGS+=(--cuda128)
  [ "$SKIP_CACHE_REDIRECT" = true ] && HOIDETR_ARGS+=(--skip-cache-redirect)
  bash "$PROJECT_ROOT/scripts/setup/install_env_hoidetr.sh" "${HOIDETR_ARGS[@]}"
else
  warn "HOI-DETR skipped: provide --contact-windows or install it separately."
fi

CHECK_ARGS=(--skip-weights)
[ "$SKIP_HOIDETR" = true ] && CHECK_ARGS+=(--skip-hoidetr)
bash "$PROJECT_ROOT/scripts/setup/check_install.sh" "${CHECK_ARGS[@]}"
ok "Requested dependencies installed and checked (weights not checked)."
echo "Obtain model assets from their original providers: docs/model-assets.md"
echo "Then run scripts/setup/check_install.sh --gpu before reconstruction."
