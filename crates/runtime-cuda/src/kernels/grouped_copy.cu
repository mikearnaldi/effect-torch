// Bit-preserving BF16 row permutation. Dispatch proves 16-byte alignment and
// supported whole-vector widths; the generic copy kernels handle other layouts.
struct __align__(16) EtGroupedCopyVector { unsigned int a, b, c, d; };
template<bool Scatter> __device__ void et_grouped_vector_copy(CudaKernelArgs a) {
    et_u64 width = a.integers[0] / 8, rows = a.elements / a.integers[0];
    for (et_u64 row = blockIdx.x; row < rows; row += gridDim.x) {
        et_u64 mapped = ((const unsigned int *)a.inputs[1])[row];
        et_u64 source = Scatter ? row : mapped % a.integers[1];
        et_u64 destination = Scatter ? mapped : row;
        for (et_u64 column = threadIdx.x; column < width; column += blockDim.x)
            ((EtGroupedCopyVector *)a.output)[destination * width + column] =
                ((const EtGroupedCopyVector *)a.inputs[0])[source * width + column];
    }
}
extern "C" __global__ void et_grouped_gather_vector(CudaKernelArgs a) { et_grouped_vector_copy<false>(a); }
extern "C" __global__ void et_grouped_scatter_vector(CudaKernelArgs a) { et_grouped_vector_copy<true>(a); }
