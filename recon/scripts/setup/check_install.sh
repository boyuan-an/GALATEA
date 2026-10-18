#!/usr/bin/env bash
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARGS=(); HOI_ARGS=(--check-only); SKIP_HOIDETR=false
for arg in "$@"; do case "$arg" in
  --gpu) ARGS+=(--gpu); HOI_ARGS+=(--gpu) ;;
  --skip-weights) ARGS+=(--skip-weights); HOI_ARGS+=(--skip-weights) ;;
  --skip-hoidetr) SKIP_HOIDETR=true ;;
  -h|--help)
    echo 'Read-only installation check. No dataset, model download, or package install.'
    echo 'Usage: bash scripts/setup/check_install.sh [--gpu] [--skip-weights] [--skip-hoidetr]'
    exit 0 ;;
  *) echo "Unknown option: $arg" >&2; exit 1 ;;
esac; done
source "$REPO/scripts/setup/runtime_env.sh"
galatea_load_conda "$REPO"
result=0
for component in main factory; do
  env_name=sam3d-objects
  [ "$component" = factory ] && env_name=factory
  (
    galatea_activate "$env_name" || exit 1
    export PYTHONNOUSERSITE=1
    python "$REPO/scripts/setup/check_environment.py" "$component" "${ARGS[@]}"
  ) || result=1
done
if [ "$SKIP_HOIDETR" = false ]; then
  bash "$REPO/scripts/setup/install_env_hoidetr.sh" "${HOI_ARGS[@]}" || result=1
fi
if [ "$result" -ne 0 ]; then
  echo 'Installation check FAILED. Resolve the errors above before reconstruction.' >&2
else
  echo 'Selected installation checks passed. --gpu additionally tests CUDA kernels and EGL rendering.'
  echo 'MoGe/SAM2/GroundingDINO use their official model hosts on first use; this check does not download them.'
  echo 'WiLoR/MANO/FoundationPose/HOI-DETR need local assets; see docs/model-assets.md (--skip-weights bypasses their checks).'
fi
exit "$result"
