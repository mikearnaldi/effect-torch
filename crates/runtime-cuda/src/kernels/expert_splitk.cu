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
extern "C" __global__ void et_expert_splitk_compute(const EtExpertSplitKDescriptor* descriptors, unsigned int count, __nv_bfloat16* partials, unsigned int columns, unsigned int inner) {
    using namespace nvcuda;
    const auto d = descriptors[et_expert_splitk_find(descriptors, count, blockIdx.x, false)];
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(d.x);
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(d.weight);
    unsigned int nt = (columns + 31) / 32, mt = (d.rows + 31) / 32, block = blockIdx.x - d.compute_prefix;
    unsigned int nr = (block % nt) * 32, mr = ((block / nt) % mt) * 32, split = block / (nt * mt);
    unsigned int begin = split * d.slice_k, end = min(inner, begin + d.slice_k), lane = threadIdx.x, warp = lane / 32;
    __shared__ __align__(32) __nv_bfloat16 a[32 * 64], b[32 * 64];
    __shared__ __align__(32) float c[32 * 32];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (unsigned int start = begin; start < end; start += 64) {
        for (unsigned int i = lane; i < 32 * 64; i += blockDim.x) {
            unsigned int outer = i / 64, k = start + i % 64;
            a[i] = nr + outer < columns && k < end ? w[(nr + outer) * inner + k] : __float2bfloat16(0.0f);
            b[i] = mr + outer < d.rows && k < end ? x[(mr + outer) * inner + k] : __float2bfloat16(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int k = 0; k < 64; k += 16) if (start + k < end) {
            wmma::load_matrix_sync(af, a + (warp % 2) * 16 * 64 + k, 64);
            wmma::load_matrix_sync(bf, b + (warp / 2) * 16 * 64 + k, 64);
            wmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c + (warp % 2) * 16 + (warp / 2) * 16 * 32, acc, 32, wmma::mem_col_major);
    __syncthreads();
    for (unsigned int i = lane; i < 32 * 32; i += blockDim.x) {
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

// Exact SM120/cuBLAS 12.9.1 second-projection M1: 32 contiguous 22-product lanes.
extern "C" __global__ void et_expert_gemv_second(
    const EtExpertSplitKDescriptor* descriptors, unsigned count) {
  const auto d = descriptors[et_expert_splitk_find(descriptors, count, blockIdx.x, false)];
  const auto* x = reinterpret_cast<const __nv_bfloat16*>(d.x);
  const auto* w = reinterpret_cast<const __nv_bfloat16*>(d.weight);
  const unsigned lane = threadIdx.x;
  const unsigned col = (blockIdx.x - d.compute_prefix) * 4 + threadIdx.y;
  float value = 0.0f;
  for (unsigned j = 0; j < 22; ++j) {
    const unsigned k = lane * 22 + j;
    value = fmaf(__bfloat162float(x[k]), __bfloat162float(w[col * 704 + k]), value);
  }
  __shared__ float partial[128];
  const unsigned i = threadIdx.y * 32 + lane;
  partial[i] = value;
  __syncthreads();
  for (unsigned stride = 16; stride; stride >>= 1) {
    if (lane < stride) partial[i] += partial[i + stride];
    __syncthreads();
  }
  if (lane == 0) reinterpret_cast<__nv_bfloat16*>(d.out)[col] = __float2bfloat16_rn(partial[i]);
}
