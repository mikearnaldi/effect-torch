#!/usr/bin/env bash
set -euo pipefail

directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
environment=${REFERENCE_ENVIRONMENT:-"${HOME}/.cache/effect-torch/reference-2.10"}
command=${1:-}

if [[ ${command} == setup ]]; then
  [[ $# -eq 1 ]] || { printf 'setup accepts no arguments\n' >&2; exit 1; }
  [[ ! -e ${environment} ]] || { printf 'Refusing to overwrite %s\n' "${environment}" >&2; exit 1; }
  uv venv --python python3.12 "${environment}"
  uv pip install --python "${environment}/bin/python" torch==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "${environment}/bin/python" \
    'transformers @ https://github.com/huggingface/transformers/archive/93ebf6b11127967f2725cf4d012aae55c3654f5a.tar.gz' \
    safetensors pillow
  exit 0
fi

case ${command} in
  download) exec python3 "${directory}/reference.py" "$@" ;;
  state|oracle|components) ;;
  *) printf 'Usage: reference.sh setup|download|state|oracle|components [checkpoint] [new-output-directory]\n' >&2; exit 1 ;;
esac

# PyTorch ships a consistent CUDA library set. Mixing its cuBLAS/cuBLASLt with
# the native runtime's Nix CUDA libraries produces invalid-value errors.
# Retain the Nix C++ runtime and driver shim required by this Python environment.
if [[ -n ${CUDA_PATH:-} ]]; then
  reference_libraries=
  IFS=: read -ra reference_paths <<< "${LD_LIBRARY_PATH:-}"
  for reference_path in "${reference_paths[@]}"; do
    case ${reference_path} in
      *cuda_cudart*|*cuda_nvrtc*|*libcublas*|*libcurand*) ;;
      *) reference_libraries="${reference_libraries:+${reference_libraries}:}${reference_path}" ;;
    esac
  done
  export LD_LIBRARY_PATH=${reference_libraries}
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "${environment}/bin/python" "${directory}/reference.py" "$@"
