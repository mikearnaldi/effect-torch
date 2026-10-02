// Compile-only and hardware diagnostic for the production RMS load variant.
// nvcc -O3 --fmad=false -arch=sm_120 rms-vector.cu -o /tmp/rms-vector
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../../crates/runtime-cuda/src/kernels/typed.cuh"
#define ET_COMPUTE_F32 1
#define double float
#define fabs fabsf
#define sqrt sqrtf
#define exp expf
#define log logf
#define sin sinf
#define cos cosf
#define tanh tanhf
#define erf erff
#define floor floorf
#define ceil ceilf
#define round roundf
#define pow powf
#define fmax fmaxf
#define fmin fminf
#define nearbyint nearbyintf
#include "../../../crates/runtime-cuda/src/kernels/common.cuh"
#define ET_TENSOR
#include "../../../crates/runtime-cuda/src/kernels/tensor.cu"
#include "../../../crates/runtime-cuda/src/kernels/compute.cu"
#undef double

static void check(cudaError_t result) {
    if (result != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(result)); std::exit(1); }
}
static void run(unsigned int dtype, unsigned int rows, unsigned int width, bool misaligned, bool weighted) {
    size_t bytes = dtype == 1 ? 4 : 2, count = size_t(rows) * width;
    std::vector<unsigned char> input((count + 8) * bytes), weights(width * bytes);
    auto fill = [&](std::vector<unsigned char>& data, unsigned int seed) {
        for (size_t index = 0; index < data.size() / bytes; ++index) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            if (dtype == 1) {
                unsigned int bits = (seed & 0x807fffffU) | (((seed >> 24) % 16 + 119) << 23);
                if (index % 79 == 0) bits = seed & 0x807fffffU; // Includes signed subnormals.
                std::memcpy(data.data() + index * bytes, &bits, 4);
            } else {
                unsigned short bits;
                if (dtype == 3) bits = (seed & 0x807fU) | (((seed >> 24) % 16 + 119) << 7);
                else bits = (seed & 0x83ffU) | (((seed >> 24) % 8 + 10) << 10);
                if (index % 79 == 0) bits = seed & (dtype == 3 ? 0x807fU : 0x83ffU);
                std::memcpy(data.data() + index * bytes, &bits, 2);
            }
        }
    };
    fill(input, 0x87164532); fill(weights, 0x23845162);
    void *x, *w, *first, *second;
    check(cudaMalloc(&x, input.size())); check(cudaMalloc(&w, weights.size()));
    check(cudaMalloc(&first, count * 4)); check(cudaMalloc(&second, count * 4));
    check(cudaMemcpy(x, input.data(), input.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    CudaKernelArgs a{}; a.inputs[0] = (et_u64)x + (misaligned ? bytes : 0);
    a.inputs[1] = weighted ? (et_u64)w : 0;
    a.input_dtypes[0] = a.input_dtypes[1] = dtype; a.output_dtype = 1;
    a.elements = count; a.integers[0] = width; a.scalars[0] = 1e-6;
    a.output = (et_u64)first; a.operation = 0; et_rms_norm_wide<<<rows, 512>>>(a);
    a.output = (et_u64)second; a.operation = 1; et_rms_norm_wide_vector<<<rows, 512>>>(a);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<unsigned int> expected(count), actual(count);
    check(cudaMemcpy(expected.data(), first, count * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), second, count * 4, cudaMemcpyDeviceToHost));
    size_t mismatches = 0; for (size_t i = 0; i < count; ++i) mismatches += expected[i] != actual[i];
    float timings[2]{};
    if (rows == 256 && width == 2816 && !misaligned) {
        cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        for (unsigned int mode = 0; mode < 2; ++mode) {
            a.operation = mode;
            for (unsigned int warmup = 0; warmup < 10; ++warmup) {
                if (mode) et_rms_norm_wide_vector<<<rows, 512>>>(a);
                else et_rms_norm_wide<<<rows, 512>>>(a);
            }
            check(cudaEventRecord(start));
            for (unsigned int iteration = 0; iteration < 200; ++iteration) {
                if (mode) et_rms_norm_wide_vector<<<rows, 512>>>(a);
                else et_rms_norm_wide<<<rows, 512>>>(a);
            }
            check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
            check(cudaEventElapsedTime(&timings[mode], start, end)); timings[mode] /= 200;
        }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    }
    std::printf("{\"dtype\":%u,\"rows\":%u,\"width\":%u,\"misaligned\":%s,\"weighted\":%s,\"mismatches\":%zu,\"scalarMs\":%.9f,\"vectorMs\":%.9f}\n",
        dtype, rows, width, misaligned ? "true" : "false", weighted ? "true" : "false", mismatches, timings[0], timings[1]);
    check(cudaFree(x)); check(cudaFree(w)); check(cudaFree(first)); check(cudaFree(second));
    if (mismatches) std::exit(2);
}
int main() {
    for (unsigned int dtype : {1U, 2U, 3U})
        for (unsigned int rows : {1U, 7U, 256U})
            for (unsigned int width : {1024U, 1028U, 2816U, 2818U, 4096U})
                for (bool misaligned : {false, true})
                    for (bool weighted : {false, true}) run(dtype, rows, width, misaligned, weighted);
}
