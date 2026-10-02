// One thread owns each complete token route row. Prior failure may leave TopK
// output unwritten; never let those bytes become foreign-kernel addresses.
extern "C" __global__ void et_fused_moe75_guard(
    const unsigned int* original_ids, const unsigned int* original_scale_bits,
    int* ids, unsigned int* scale_bits, unsigned long long* status,
    unsigned int context, unsigned int tokens) {
    unsigned int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) return;
    bool invalid = atomicCAS(status, 0ULL, 0ULL) != 0;
    unsigned int selected[8];
    if (!invalid) {
        for (unsigned int rank = 0; rank < 8; ++rank) {
            selected[rank] = original_ids[token * 8 + rank];
            if (selected[rank] >= 128) invalid = true;
            for (unsigned int previous = 0; previous < rank; ++previous)
                if (selected[previous] == selected[rank]) invalid = true;
        }
        if (invalid) atomicCAS(status, 0ULL, (static_cast<unsigned long long>(context) << 32) | 1ULL);
    }
    for (unsigned int rank = 0; rank < 8; ++rank) {
        ids[token * 8 + rank] = invalid ? static_cast<int>(rank) : static_cast<int>(selected[rank]);
        scale_bits[token * 8 + rank] = invalid ? 0U : original_scale_bits[token * 8 + rank];
    }
}
