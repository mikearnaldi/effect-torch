#!/usr/bin/env bash
set -euo pipefail
source_directory=${1:-/root}
binary_directory=${2:-/root}
cutlass_include=${CUTLASS_INCLUDE:-/root/.cache/effect-torch/vllm-0.24.0/lib/python3.12/site-packages/flashinfer/data/cutlass/include}
nvcc_binary=${NVCC:-nvcc}
for tile in 16x64 16x128 32x64 32x128; do
  rows=${tile%x*}
  columns=${tile#*x}
  "$nvcc_binary" -O3 -std=c++17 -arch=sm_120 -I "$cutlass_include" \
    -DET_CUTLASS_M="$rows" -DET_CUTLASS_N="$columns" -DET_CUTLASS_K=64 \
    -DET_CUTLASS_STAGES=3 "$source_directory/expert-cutlass-subset.cu" \
    -lcublas -lcuda -o "$binary_directory/expert-cutlass-subset-$tile"
done
