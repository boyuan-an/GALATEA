#!/usr/bin/env bash
galatea_build_env() {
  local prefix="$1" sp includes
  sp="$("$prefix/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" || return
  includes="$prefix/include/eigen3:$prefix/include:$prefix/targets/x86_64-linux/include"
  local lib
  for lib in cusparse cublas cusolver cuda_runtime cuda_nvrtc; do
    includes="$includes:$sp/nvidia/$lib/include"
  done
  export CUDA_HOME="$prefix"
  export CPATH="$includes${CPATH:+:$CPATH}"
  export CC="$prefix/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$prefix/bin/x86_64-conda-linux-gnu-c++"
  export CUDAHOSTCXX="$CXX"
  export NVCC_PREPEND_FLAGS="-ccbin $CXX"
  export CMAKE_PREFIX_PATH="$prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
}

galatea_write_build_hooks() {
  local prefix="$1" helper="$2"
  mkdir -p "$prefix/etc/conda/activate.d" "$prefix/etc/conda/deactivate.d"
  cp "$helper" "$prefix/etc/conda/galatea_build_env.sh"
  cat > "$prefix/etc/conda/activate.d/zz_galatea_build_env.sh" <<'EOF'
#!/usr/bin/env bash
if [ "${_GALATEA_BUILD_ACTIVE:-}" != "$CONDA_PREFIX" ]; then
  for _galatea_key in CUDA_HOME CPATH CC CXX CUDAHOSTCXX NVCC_PREPEND_FLAGS CMAKE_PREFIX_PATH; do
    _galatea_saved="_GALATEA_OLD_${_galatea_key}"
    _galatea_set="_GALATEA_SET_${_galatea_key}"
    printf -v "$_galatea_set" '%s' "${!_galatea_key+x}"
    printf -v "$_galatea_saved" '%s' "${!_galatea_key-}"
  done
  source "$CONDA_PREFIX/etc/conda/galatea_build_env.sh"
  galatea_build_env "$CONDA_PREFIX"
  _GALATEA_BUILD_ACTIVE="$CONDA_PREFIX"
  unset _galatea_key _galatea_saved _galatea_set
fi
EOF
  cat > "$prefix/etc/conda/deactivate.d/zz_galatea_build_env.sh" <<'EOF'
#!/usr/bin/env bash
if [ -n "${_GALATEA_BUILD_ACTIVE:-}" ]; then
  for _galatea_key in CUDA_HOME CPATH CC CXX CUDAHOSTCXX NVCC_PREPEND_FLAGS CMAKE_PREFIX_PATH; do
    _galatea_saved="_GALATEA_OLD_${_galatea_key}"
    _galatea_set="_GALATEA_SET_${_galatea_key}"
    if [ "${!_galatea_set-}" = x ]; then
      printf -v "$_galatea_key" '%s' "${!_galatea_saved}"
      export "$_galatea_key"
    else
      unset "$_galatea_key"
    fi
    unset "$_galatea_saved" "$_galatea_set"
  done
  unset _GALATEA_BUILD_ACTIVE _galatea_key _galatea_saved _galatea_set
fi
EOF
}
