#!/usr/bin/env bash
galatea_project_cache() {
  local root="$1" key value
  local token_path="${HF_TOKEN_PATH:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/token}"
  mkdir -p "$root"/{pip,huggingface,torch,nv,tmp} || return
  root="$(cd "$root" && pwd)" || return
  local keys=(XDG_CACHE_HOME PIP_CACHE_DIR HF_HOME TORCH_HOME CUDA_CACHE_PATH TMPDIR TMP TEMP HF_HUB_DISABLE_XET)
  local values=("$root" "$root/pip" "$root/huggingface" "$root/torch" "$root/nv" "$root/tmp" "$root/tmp" "$root/tmp" 1)
  if [ -f "$token_path" ]; then
    keys+=(HF_TOKEN_PATH)
    values+=("$token_path")
  fi
  local index
  {
    for index in "${!keys[@]}"; do
      key="${keys[$index]}"; value="${values[$index]}"
      printf 'export %s=%q\n' "$key" "$value"
    done
  } > "$root/env.sh" || return
  source "$root/env.sh"
}
