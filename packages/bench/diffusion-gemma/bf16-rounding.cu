// GPU exhaustive bit proof and conversion/ordered-add microbenchmarks.
// nvcc -O3 --fmad=false -arch=sm_120 bf16-rounding.cu -o /tmp/bf16-rounding-gpu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define ET_CUDA_F32_BF16_BITS 1
#include "../../../crates/runtime-cuda/src/kernels/typed.cuh"
static void check(cudaError_t code) {
    if (code != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(code)); std::exit(1); }
}
__global__ void exhaustive(unsigned long long* errors) {
    for (unsigned long long bits = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         bits < (1ULL << 32); bits += (unsigned long long)gridDim.x * blockDim.x) {
        float value = __uint_as_float((unsigned int)bits);
        if (et_to16(value, true) != et_to16((double)value, true)) atomicAdd(errors, 1ULL);
    }
}
template<bool Bits> __device__ unsigned short narrow(float x) {
    return Bits ? et_to16(x, true) : et_to16((double)x, true);
}
template<bool Bits, bool Sum> __global__ void conversion(const float* input, unsigned short* output, unsigned int count) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        if (!Sum) output[i] = narrow<Bits>(input[i]);
        else {
            float total = 0;
            unsigned short rounded = 0;
            #pragma unroll
            for (unsigned int step = 0; step < 8; ++step) {
                total += input[(size_t)step * count + i];
                rounded = narrow<Bits>(total);
                total = et_bfloat_float(rounded);
            }
            output[i] = rounded;
        }
    }
}
template<bool Sum> static void benchmark(unsigned int count) {
    const size_t input_count = size_t(count) * (Sum ? 8 : 1);
    std::vector<float> host(input_count);
    unsigned int random = 0x13a979bc;
    for (size_t i = 0; i < input_count; ++i) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        unsigned int bits = (random & 0x807fffffU) | (((random >> 24) % 16 + 119) << 23);
        std::memcpy(&host[i], &bits, 4);
    }
    float* input; unsigned short *old, *fast;
    check(cudaMalloc(&input, input_count * 4)); check(cudaMalloc(&old, size_t(count) * 2)); check(cudaMalloc(&fast, size_t(count) * 2));
    check(cudaMemcpy(input, host.data(), input_count * 4, cudaMemcpyHostToDevice));
    unsigned int blocks = (count + 255) / 256; if (blocks > 65535) blocks = 65535;
    conversion<false, Sum><<<blocks, 256>>>(input, old, count);
    conversion<true, Sum><<<blocks, 256>>>(input, fast, count);
    std::vector<unsigned short> expected(count), actual(count);
    check(cudaMemcpy(expected.data(), old, size_t(count) * 2, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), fast, size_t(count) * 2, cudaMemcpyDeviceToHost));
    size_t mismatches = 0; for (unsigned int i = 0; i < count; ++i) mismatches += expected[i] != actual[i];
    cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
    float time[2]{};
    for (unsigned int mode = 0; mode < 2; ++mode) {
        for (unsigned int i = 0; i < 10; ++i) {
            if (mode) conversion<true, Sum><<<blocks, 256>>>(input, fast, count);
            else conversion<false, Sum><<<blocks, 256>>>(input, old, count);
        }
        check(cudaEventRecord(start));
        for (unsigned int i = 0; i < 100; ++i) {
            if (mode) conversion<true, Sum><<<blocks, 256>>>(input, fast, count);
            else conversion<false, Sum><<<blocks, 256>>>(input, old, count);
        }
        check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        check(cudaEventElapsedTime(&time[mode], start, end)); time[mode] /= 100;
    }
    std::printf("{\"kind\":\"%s\",\"elements\":%u,\"mismatches\":%zu,\"referenceMs\":%.9f,\"bitsMs\":%.9f}\n",
        Sum ? "eight-ordered-bf16-adds" : "convert-f32-bf16", count, mismatches, time[0], time[1]);
    check(cudaEventDestroy(start)); check(cudaEventDestroy(end)); check(cudaFree(input)); check(cudaFree(old)); check(cudaFree(fast));
    if (mismatches) std::exit(2);
}
int main() {
    unsigned long long *device_errors, errors;
    check(cudaMalloc(&device_errors, sizeof(errors))); check(cudaMemset(device_errors, 0, sizeof(errors)));
    exhaustive<<<65535, 256>>>(device_errors);
    check(cudaMemcpy(&errors, device_errors, sizeof(errors), cudaMemcpyDeviceToHost));
    check(cudaFree(device_errors));
    std::printf("{\"kind\":\"exhaustive\",\"encodings\":4294967296,\"mismatches\":%llu}\n", errors);
    if (errors) return 2;
    benchmark<false>(256 * 262144);
    benchmark<false>(256 * 8 * 2816);
    benchmark<true>(256 * 2816);
}
