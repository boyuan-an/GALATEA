#!/usr/bin/env bash
galatea_load_conda() {
  local repo="$1" candidate
  if ! command -v conda >/dev/null 2>&1; then
    for candidate in "${CONDA_EXE:-}" "$repo/.bootstrap/miniconda/bin/conda" \
      "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" /opt/conda/bin/conda; do
      if [ -x "$candidate" ]; then
        export PATH="$(dirname "$candidate"):$PATH"
        break
      fi
    done
  fi
  CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo 'Conda not found. Install Miniforge or put your conda executable on PATH; see docs/installation.md.' >&2
    return 1
  }
  source "$CONDA_BASE/etc/profile.d/conda.sh" || return
  export PYTHONNOUSERSITE=1
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-}"
}

galatea_env_prefix() {
  local directory="${CONDA_ENVS_PATH:-}"
  directory="${directory%%:*}"
  if [ -z "$directory" ]; then
    directory="$(conda info --json | "$CONDA_BASE/bin/python" -c \
      'import json,sys; print(json.load(sys.stdin)["envs_dirs"][0])')" || return
  fi
  "$CONDA_BASE/bin/python" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(os.path.join(sys.argv[1], sys.argv[2]))))' "$directory" "$1"
}

galatea_activate() {
  local prefix
  prefix="$(galatea_env_prefix "$1")" || return
  conda activate "$prefix"
}

galatea_ensure_env() {
  local prefix="$1" version="$2"
  if [ ! -f "$prefix/conda-meta/history" ]; then
    conda create -y --prefix "$prefix" --override-channels -c conda-forge "python=$version" || return
  fi
  "$prefix/bin/python" -c \
    'import sys; assert "%d.%d" % sys.version_info[:2] == sys.argv[1], sys.version' "$version" || {
    echo "Existing environment $prefix must use Python $version. Set CONDA_ENVS_PATH to a fresh directory." >&2
    return 1
  }
}

galatea_load_hf_token() {
  [ -z "${HF_TOKEN:-}" ] || return 0
  [ -n "${HF_TOKEN_FILE:-}" ] || return 0
  if [ ! -r "$HF_TOKEN_FILE" ]; then
    echo "HF_TOKEN_FILE is not readable: $HF_TOKEN_FILE" >&2
    return 1
  fi
  local token
  token="$(tr -d ' \t\r\n' < "$HF_TOKEN_FILE")" || return
  if [ -z "$token" ]; then
    echo 'HF_TOKEN_FILE is empty. Supply an authorized token or use an existing Hugging Face login.' >&2
    return 1
  fi
  export HF_TOKEN="$token"
}
