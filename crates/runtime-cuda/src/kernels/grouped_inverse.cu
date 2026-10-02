// Stable row compaction: one CTA per expert distributes consecutive chunks
// across eight warps while preserving the original ascending-row permutation.
extern "C" __global__ void et_grouped_rows_inverse_block(CudaKernelArgs a) {
    if (a.scratch[3] && *((const et_u64*)a.scratch[3]) != 0) return;
    __shared__ unsigned int counts[8], prefix[9];
    unsigned int lane = threadIdx.x & 31U, warp = threadIdx.x / 32;
    for (et_u64 expert = blockIdx.x; expert < a.integers[0]; expert += gridDim.x) {
        unsigned int offset = ((const unsigned int *)a.inputs[1])[expert + 1];
        for (et_u64 base = 0; base < a.elements; base += 256) {
            et_u64 row = base + threadIdx.x;
            bool selected = row < a.elements && ((const unsigned int *)a.inputs[0])[row] == expert;
            unsigned int mask = __ballot_sync(0xffffffffU, selected);
            if (!lane) counts[warp] = __popc(mask);
            __syncthreads();
            if (!threadIdx.x) {
                unsigned int total = 0;
                for (unsigned int w = 0; w < 8; ++w) {
                    prefix[w] = total;
                    total += counts[w];
                }
                prefix[8] = total;
            }
            __syncthreads();
            if (selected) {
                unsigned int sorted = offset + prefix[warp] + __popc(mask & ((1U << lane) - 1U));
                ((unsigned int *)a.output)[sorted] = (unsigned int)row;
                ((unsigned int *)a.scratch[0])[row] = sorted;
            }
            offset += prefix[8];
            // Counts and prefixes are disjoint. The next chunk's first barrier
            // completes all prefix reads before thread zero updates prefixes.
        }
    }
}

extern "C" __global__ void et_grouped_rows_inverse_warp(CudaKernelArgs a) {
    if (a.scratch[3] && *((const et_u64*)a.scratch[3]) != 0) return;
    const unsigned int lane = threadIdx.x & 31U;
    for (et_u64 e = et_thread() / 32; e < a.integers[0]; e += (et_u64)gridDim.x * (blockDim.x / 32)) {
        unsigned int offset = ((const unsigned int *)a.inputs[1])[e + 1];
        for (et_u64 base = 0; base < a.elements; base += 32) {
            et_u64 row = base + lane;
            bool selected = row < a.elements && ((const unsigned int *)a.inputs[0])[row] == e;
            unsigned int mask = __ballot_sync(0xffffffffU, selected);
            if (selected) {
                unsigned int sorted = offset + __popc(mask & ((1U << lane) - 1U));
                ((unsigned int *)a.output)[sorted] = (unsigned int)row;
                ((unsigned int *)a.scratch[0])[row] = sorted;
            }
            offset += __popc(mask);
        }
    }
}
