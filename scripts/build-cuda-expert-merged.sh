#!/usr/bin/env bash
# Explicit experimental artifact build; never called by the package build.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: $0 CUTLASS_INCLUDE OUTPUT_PTX" >&2
  exit 2
fi
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
nvcc -O3 -std=c++17 --ptx -arch=compute_120 --expt-relaxed-constexpr \
  -I "$1" "$script_root/crates/runtime-cuda/src/kernels/expert_merged.cu" -o "$2"
