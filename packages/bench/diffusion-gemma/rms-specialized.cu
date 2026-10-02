// Exact RMS specialization screen; production implementations are the reference.
// nvcc -O3 --fmad=false -std=c++17 -arch=sm_120 -Xptxas=-v rms-specialized.cu -o rms-specialized
#define main unused_rms_vector_main
#include "rms-vector.cu"
#undef main

template<unsigned DType, unsigned OutType, bool Unroll>
__global__ void rms_fixed_2816(CudaKernelArgs a) {
    constexpr unsigned width = 2816;
    unsigned row = blockIdx.x, lane = threadIdx.x & 31;
    __shared__ float inverse;
    auto source_row = et_rms_source_row(a, row);
    if (threadIdx.x < 32) {
        float partial[4] = {0, 0, 0, 0};
        auto address = a.inputs[0] + source_row * width * (DType == 1 ? 4 : 2);
        bool aligned = address % (DType == 1 ? 16 : 8) == 0;
        #pragma unroll (Unroll ? 22 : 1)
        for (unsigned step = 0; step < 22; ++step) {
            unsigned k = lane * 4 + step * 128;
            float values[4];
            if (aligned) {
                if (DType == 1) {
                    auto source = address + k * 4;
                    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                        : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]) : "l"(source));
                } else {
                    et_u64 packed, source = address + k * 2;
                    asm volatile("ld.global.u64 %0, [%1];" : "=l"(packed) : "l"(source));
                    #pragma unroll
                    for (unsigned j = 0; j < 4; ++j) {
                        unsigned short bits = packed >> (j * 16);
                        values[j] = DType == 3 ? et_bfloat_float(bits) : et_half_float(bits);
                    }
                }
            } else {
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    values[j] = et_load<float>(a.inputs[0], DType, source_row * width + k + j);
            }
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                float square = values[j] * values[j];
                partial[j] += square;
            }
        }
        float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
        for (unsigned offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffU, sum, offset);
        if (!lane) inverse = rsqrtf(sum * (1.0f / float(width)) + float(a.scalars[0]));
    }
    __syncthreads();
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x) {
        float value = et_load<float>(a.inputs[0], DType, source_row * width + k) * inverse;
        if (a.inputs[1]) value *= et_load<float>(a.inputs[1], a.input_dtypes[1], k);
        et_store(a.output, OutType, row * width + k, value);
    }
}

template<unsigned DType, unsigned OutType>
static void launch_typed(CudaKernelArgs a, unsigned rows, unsigned block, bool unroll) {
    if (unroll) rms_fixed_2816<DType, OutType, true><<<rows, block, 0, cudaStreamPerThread>>>(a);
    else rms_fixed_2816<DType, OutType, false><<<rows, block, 0, cudaStreamPerThread>>>(a);
}
template<unsigned DType>
static void dispatch_out(CudaKernelArgs a, unsigned rows, unsigned block, bool unroll) {
    if (a.output_dtype == 1) launch_typed<DType, 1>(a, rows, block, unroll);
    else if (a.output_dtype == 2) launch_typed<DType, 2>(a, rows, block, unroll);
    else launch_typed<DType, 3>(a, rows, block, unroll);
}
static void launch_variant(CudaKernelArgs a, unsigned rows, unsigned mode) {
    unsigned block = mode % 3 == 0 ? 512 : mode % 3 == 1 ? 128 : 256;
    if (mode < 3 || a.integers[0] != 2816) et_rms_norm_wide_vector<<<rows, block, 0, cudaStreamPerThread>>>(a);
    else if (a.input_dtypes[0] == 1) dispatch_out<1>(a, rows, block, mode >= 6);
    else if (a.input_dtypes[0] == 2) dispatch_out<2>(a, rows, block, mode >= 6);
    else dispatch_out<3>(a, rows, block, mode >= 6);
}

static std::vector<unsigned char> data(unsigned dtype, size_t count, unsigned seed, unsigned pattern) {
    unsigned bytes = dtype == 1 ? 4 : 2;
    std::vector<unsigned char> result(count * bytes);
    for (size_t i = 0; i < count; ++i) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        unsigned bits;
        if (dtype == 1) bits = (seed & 0x807fffffU) | (((seed >> 24) % 16 + 119) << 23);
        else if (dtype == 3) bits = (seed & 0x807fU) | (((seed >> 24) % 16 + 119) << 7);
        else bits = (seed & 0x83ffU) | (((seed >> 24) % 8 + 10) << 10);
        if (pattern == 1 && i % 31 == 0) bits &= dtype == 1 ? 0x807fffffU : dtype == 3 ? 0x807fU : 0x83ffU;
        if (pattern == 2) {
            unsigned specials32[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345};
            unsigned specials16[] = {0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0x7f81};
            unsigned half16[] = {0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0x7c01};
            // Keep each full row uniform to include signed-zero and infinity rows.
            unsigned index = (i / 2816) % 6;
            bits = dtype == 1 ? specials32[index] : dtype == 3 ? specials16[index] : half16[index];
        }
        std::memcpy(result.data() + i * bytes, &bits, bytes);
    }
    return result;
}

static void prove(unsigned dtype, unsigned outtype, unsigned weighttype, unsigned width,
                  unsigned rows, bool misaligned, bool strided, unsigned pattern) {
    unsigned bytes = dtype == 1 ? 4 : 2, obytes = outtype == 1 ? 4 : 2;
    size_t count = size_t(rows) * width, source_count = strided ? count * 2 : count;
    auto hx = data(dtype, source_count + 8, 0x1234, pattern);
    auto hw = data(weighttype ? weighttype : 1, width, 0x5432, 0);
    void *x, *w, *reference, *candidate;
    check(cudaMalloc(&x, hx.size())); check(cudaMalloc(&w, hw.size()));
    check(cudaMalloc(&reference, count * obytes)); check(cudaMalloc(&candidate, count * obytes));
    check(cudaMemcpy(x, hx.data(), hx.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CudaKernelArgs a{}; a.inputs[0] = (et_u64)x + (misaligned ? bytes : 0);
    a.inputs[1] = weighttype ? (et_u64)w : 0;
    a.input_dtypes[0] = dtype; a.input_dtypes[1] = weighttype; a.output_dtype = outtype;
    a.elements = count; a.integers[0] = width; a.scalars[0] = 1e-6;
    if (strided) { a.integers[1] = 1; a.integers[2] = 1; a.integers[3] = rows; a.integers[4] = 2; }
    a.output = (et_u64)reference; launch_variant(a, rows, 0); check(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<unsigned char> expected(count * obytes), actual(count * obytes);
    check(cudaMemcpy(expected.data(), reference, expected.size(), cudaMemcpyDeviceToHost));
    for (unsigned mode = 1; mode < 9; ++mode) {
        a.output = (et_u64)candidate; launch_variant(a, rows, mode); check(cudaStreamSynchronize(cudaStreamPerThread));
        check(cudaMemcpy(actual.data(), candidate, actual.size(), cudaMemcpyDeviceToHost));
        size_t mismatches = 0;
        for (size_t i = 0; i < count; ++i) mismatches += std::memcmp(actual.data() + i * obytes, expected.data() + i * obytes, obytes) != 0;
        std::printf("{\"kind\":\"proof\",\"dtype\":%u,\"outtype\":%u,\"weighttype\":%u,\"width\":%u,\"rows\":%u,\"misaligned\":%u,\"strided\":%u,\"pattern\":%u,\"mode\":%u,\"mismatches\":%zu}\n", dtype, outtype, weighttype, width, rows, misaligned, strided, pattern, mode, mismatches);
        if (mismatches) std::exit(2);
    }
    check(cudaFree(x)); check(cudaFree(w)); check(cudaFree(reference)); check(cudaFree(candidate));
}

static void time_variants(unsigned dtype, unsigned outtype, unsigned weighttype, unsigned banks) {
    unsigned rows = 256, width = 2816, bytes = dtype == 1 ? 4 : 2, obytes = outtype == 1 ? 4 : 2;
    size_t count = size_t(rows) * width;
    auto hx = data(dtype, count, 0x1234, 0), hw = data(weighttype ? weighttype : 1, width, 0x5432, 0);
    void *x, *w, *output;
    check(cudaMalloc(&x, hx.size() * banks)); check(cudaMalloc(&w, hw.size()));
    check(cudaMalloc(&output, count * obytes * banks));
    for (unsigned b = 0; b < banks; ++b) check(cudaMemcpy((char*)x + b * hx.size(), hx.data(), hx.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CudaKernelArgs a{}; a.inputs[1] = weighttype ? (et_u64)w : 0;
    a.input_dtypes[0] = dtype; a.input_dtypes[1] = weighttype; a.output_dtype = outtype;
    a.elements = count; a.integers[0] = width; a.scalars[0] = 1e-6;
    cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
    for (unsigned mode = 0; mode < 9; ++mode) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal));
        for (unsigned b = 0; b < banks; ++b) {
            a.inputs[0] = (et_u64)x + b * count * bytes;
            a.output = (et_u64)output + b * count * obytes;
            launch_variant(a, rows, mode);
        }
        check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        std::vector<float> times;
        for (int repeat = -5; repeat < 31; ++repeat) {
            check(cudaEventRecord(start, cudaStreamPerThread));
            for (unsigned i = 0; i < 10; ++i) check(cudaGraphLaunch(exec, cudaStreamPerThread));
            check(cudaEventRecord(end, cudaStreamPerThread)); check(cudaEventSynchronize(end));
            float ms; check(cudaEventElapsedTime(&ms, start, end)); if (repeat >= 0) times.push_back(ms / (banks * 10));
        }
        std::sort(times.begin(), times.end());
        std::printf("{\"kind\":\"timing\",\"dtype\":%u,\"outtype\":%u,\"weighttype\":%u,\"banks\":%u,\"mode\":%u,\"gpuMedianMs\":%.9f}\n", dtype, outtype, weighttype, banks, mode, times[15]);
        check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    check(cudaFree(x)); check(cudaFree(w)); check(cudaFree(output));
}

int main() {
    for (unsigned dtype : {1U, 2U, 3U}) for (unsigned outtype : {1U, 2U, 3U})
        for (unsigned weighttype : {0U, 1U, 3U}) for (unsigned pattern : {0U, 1U, 2U})
            for (bool misaligned : {false, true}) for (bool strided : {false, true})
                prove(dtype, outtype, weighttype, 2816, 7, misaligned, strided, pattern);
    for (unsigned width : {1024U, 1028U, 2818U, 4096U})
        for (unsigned dtype : {1U, 2U, 3U}) prove(dtype, dtype, 1, width, 7, true, true, 1);
    for (unsigned dtype : {1U, 3U}) for (unsigned outtype : {1U, 3U})
        for (unsigned weighttype : {0U, 1U, 3U}) for (unsigned banks : {1U, 64U})
            time_variants(dtype, outtype, weighttype, banks);
}
