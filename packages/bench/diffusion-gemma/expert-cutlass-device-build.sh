#!/usr/bin/env bash
set -euo pipefail
source_directory=${1:-/root}
binary_directory=${2:-/root}
cutlass_include=${CUTLASS_INCLUDE:-/root/.cache/effect-torch/vllm-0.24.0/lib/python3.12/site-packages/flashinfer/data/cutlass/include}
nvcc_binary=${NVCC:-nvcc}
for tile in 32x64 32x128; do
  "$nvcc_binary" -O3 -std=c++17 -arch=sm_120 --expt-relaxed-constexpr \
    -I "$cutlass_include" -DET_CUTLASS_M="${tile%x*}" -DET_CUTLASS_N="${tile#*x}" \
    -DET_CUTLASS_K=64 -DET_CUTLASS_STAGES=3 \
    "$source_directory/expert-cutlass-device.cu" -lcublas -lcuda \
    -o "$binary_directory/expert-cutlass-device-${tile}"
done
