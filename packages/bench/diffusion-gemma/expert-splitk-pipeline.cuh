// Experimental variant only; not loaded by production.
#ifndef ET_EXPERT_ROW_TILE
#define ET_EXPERT_ROW_TILE 32
#endif
// Exact BF16 split-K partial rounding, matching the expert cuBLAS schedule.
#include <cuda_bf16.h>
#include <mma.h>
struct EtExpertSplitKDescriptor {
    unsigned long long x, weight, out, partial_offset;
    unsigned int rows, splits, slice_k, compute_prefix, reduce_prefix, reserved[3];
};
static_assert(sizeof(EtExpertSplitKDescriptor) == 64, "descriptor ABI");
struct EtExpertSplitKUpload { EtExpertSplitKDescriptor descriptors[32]; };
extern "C" __global__ void et_expert_splitk_upload(EtExpertSplitKDescriptor* descriptors, unsigned int first, unsigned int count, EtExpertSplitKUpload batch) {
    unsigned int i = threadIdx.x;
    if (i < count) descriptors[first + i] = batch.descriptors[i];
}
__device__ __forceinline__ unsigned int et_expert_splitk_find(const EtExpertSplitKDescriptor* d, unsigned int count, unsigned int block, bool reduce) {
    unsigned int low = 0, high = count;
    while (low + 1 < high) {
        unsigned int mid = (low + high) >> 1;
        unsigned int prefix = reduce ? d[mid].reduce_prefix : d[mid].compute_prefix;
        if (prefix <= block) low = mid; else high = mid;
    }
    return low;
}
// Experimental double-buffered 16-byte copies. The WMMA K16 accumulation
// sequence and BF16 partial rounding are identical to the reference kernel.
#ifndef ET_PIPELINE_K
#define ET_PIPELINE_K 64
#endif
#ifndef ET_PIPELINE_LD
#define ET_PIPELINE_LD ET_PIPELINE_K
#endif
__device__ __forceinline__ void et_expert_copy16(void* dst, const void* src, unsigned int bytes) {
    unsigned int shared = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(shared), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void et_expert_prefetch(
    __nv_bfloat16* a, __nv_bfloat16* b,
    const __nv_bfloat16* w, const __nv_bfloat16* x,
    unsigned int nr, unsigned int mr, unsigned int rows,
    unsigned int columns, unsigned int inner, unsigned int start, unsigned int end) {
    // Every vector has eight BF16 elements. Current eligible inner/slice widths
    // are multiples of sixteen, so vectors are either complete or zero-filled.
    for (unsigned int v = threadIdx.x; v < 32 * (ET_PIPELINE_K / 8); v += blockDim.x) {
        unsigned int row = v / (ET_PIPELINE_K / 8), k = start + (v % (ET_PIPELINE_K / 8)) * 8;
        bool valid = k < end && nr + row < columns;
        et_expert_copy16(a + row * ET_PIPELINE_LD + (v % (ET_PIPELINE_K / 8)) * 8,
            valid ? w + static_cast<unsigned long long>(nr + row) * inner + k : w,
            valid ? 16 : 0);
    }
    for (unsigned int v = threadIdx.x; v < ET_EXPERT_ROW_TILE * (ET_PIPELINE_K / 8); v += blockDim.x) {
        unsigned int row = v / (ET_PIPELINE_K / 8), k = start + (v % (ET_PIPELINE_K / 8)) * 8;
        bool valid = k < end && mr + row < rows;
        et_expert_copy16(b + row * ET_PIPELINE_LD + (v % (ET_PIPELINE_K / 8)) * 8,
            valid ? x + static_cast<unsigned long long>(mr + row) * inner + k : x,
            valid ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;");
}
extern "C" __global__ void et_expert_splitk_compute(const EtExpertSplitKDescriptor* descriptors, unsigned int count, __nv_bfloat16* partials, unsigned int columns, unsigned int inner) {
    using namespace nvcuda;
    const auto d = descriptors[et_expert_splitk_find(descriptors, count, blockIdx.x, false)];
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(d.x);
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(d.weight);
    unsigned int nt = (columns + 31) / 32, mt = (d.rows + ET_EXPERT_ROW_TILE - 1) / ET_EXPERT_ROW_TILE, block = blockIdx.x - d.compute_prefix;
    unsigned int nr = (block % nt) * 32, mr = ((block / nt) % mt) * ET_EXPERT_ROW_TILE, split = block / (nt * mt);
    unsigned int begin = split * d.slice_k, end = min(inner, begin + d.slice_k), lane = threadIdx.x, warp = lane / 32;
    // B is no longer needed after the last MMA, so reuse its shared storage
    // for the epilogue. RowTile128 should use K32/LD48 or K64/LD64
    // to remain beneath 48 KiB static shared.
    __shared__ __align__(32) __nv_bfloat16 a[2][32 * ET_PIPELINE_LD];
    __shared__ __align__(32) union {
        __nv_bfloat16 b[2][ET_EXPERT_ROW_TILE * ET_PIPELINE_LD];
        float c[32 * ET_EXPERT_ROW_TILE];
    } stage_data;
    auto& b = stage_data.b;
    auto& c = stage_data.c;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    unsigned int stage = 0;
    et_expert_prefetch(a[stage], b[stage], w, x, nr, mr, d.rows, columns, inner, begin, end);
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    for (unsigned int start = begin; start < end; start += ET_PIPELINE_K) {
        bool next = start + ET_PIPELINE_K < end;
        if (next) et_expert_prefetch(a[stage ^ 1], b[stage ^ 1], w, x, nr, mr, d.rows, columns, inner, start + ET_PIPELINE_K, end);
        #pragma unroll
        for (unsigned int k = 0; k < ET_PIPELINE_K; k += 16) if (start + k < end) {
            wmma::load_matrix_sync(af, a[stage] + (warp % 2) * 16 * ET_PIPELINE_LD + k, ET_PIPELINE_LD);
            wmma::load_matrix_sync(bf, b[stage] + (warp / 2) * 16 * ET_PIPELINE_LD + k, ET_PIPELINE_LD);
            wmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
        if (next) {
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
        stage ^= 1;
    }
    wmma::store_matrix_sync(c + (warp % 2) * 16 + (warp / 2) * 16 * 32, acc, 32, wmma::mem_col_major);
    __syncthreads();
    for (unsigned int i = lane; i < 32 * ET_EXPERT_ROW_TILE; i += blockDim.x) {
        unsigned int n = nr + i % 32, m = mr + i / 32;
        if (n < columns && m < d.rows) partials[d.partial_offset + (static_cast<unsigned long long>(split) * d.rows + m) * columns + n] = __float2bfloat16_rn(c[i]);
    }
}
extern "C" __global__ void et_expert_splitk_reduce(const EtExpertSplitKDescriptor* descriptors, unsigned int count, const __nv_bfloat16* partials, unsigned int columns) {
    const auto d = descriptors[et_expert_splitk_find(descriptors, count, blockIdx.x, true)];
    unsigned int i = (blockIdx.x - d.reduce_prefix) * blockDim.x + threadIdx.x, elements = d.rows * columns;
    if (i >= elements) return;
    float value = 0.0f;
    for (unsigned int split = 0; split < d.splits; ++split) value += __bfloat162float(partials[d.partial_offset + static_cast<unsigned long long>(split) * elements + i]);
    reinterpret_cast<__nv_bfloat16*>(d.out)[i] = __float2bfloat16_rn(value);
}
