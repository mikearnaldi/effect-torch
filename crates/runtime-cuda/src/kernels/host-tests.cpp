// Host execution verifies scalar arithmetic and descriptor interpretation.
// It cannot verify CUDA scheduling, PTX intrinsics or device memory behavior.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

using std::min;
using std::max;
using std::isnan;
using std::isfinite;
#define ET_HOST_TEST
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#define __launch_bounds__(...)
struct Dim { unsigned int x = 0, y = 0, z = 0; };
static Dim threadIdx, blockIdx, blockDim{256, 1, 1}, gridDim{1, 1, 1};
static void __syncthreads() {}
static void __syncwarp() {}
template<class T> T __shfl_down_sync(unsigned int, T value, unsigned int) { return value; }
template<class T> T __shfl_sync(unsigned int, T value, unsigned int) { return value; }
static unsigned int atomicCAS(unsigned int *p, unsigned int expected, unsigned int value) {
    unsigned int old = *p; if (old == expected) *p = value; return old;
}
static unsigned int __float_as_uint(float value) { unsigned int bits; memcpy(&bits, &value, 4); return bits; }
static float __uint_as_float(unsigned int bits) { float value; memcpy(&value, &bits, 4); return value; }
static unsigned long long __double_as_longlong(double value) { unsigned long long bits; memcpy(&bits, &value, 8); return bits; }
static int __clzll(unsigned long long value) { return __builtin_clzll(value); }
static int __float2int_rn(float value) { return (int)nearbyintf(value); }

#include "typed.cuh"
#include "typed.cu"
#include "quantized.cu"
#include "cache.cu"

#ifdef ET_TEST_F32
using Compute = float;
#define double float
#define fabs fabsf
#define sqrt sqrtf
#define exp expf
#define log logf
#define sin sinf
#define cos cosf
#define pow powf
#define fmax fmaxf
#define fmin fminf
#else
using Compute = double;
#endif
#include "common.cuh"
#include "pointwise.cu"
#include "tensor.cu"
#include "linalg.cu"
#include "neural.cu"
#include "stateful.cu"
#define ET_POINTWISE
#define ET_TENSOR
#define ET_LINALG
#define ET_NEURAL
#define ET_STATEFUL
#include "compute.cu"
#undef double
#undef fabs
#undef sqrt
#undef exp
#undef log
#undef sin
#undef cos
#undef pow
#undef fmax
#undef fmin

template<class T> et_u64 address(T *p) { return (et_u64)p; }
static std::vector<et_u64> metadata(std::vector<et_u64> out, std::vector<std::vector<et_u64>> inputs) {
    inputs.resize(8);
    std::vector<et_u64> result{out.size()};
    for (auto &shape : inputs) result.push_back(shape.size());
    result.insert(result.end(), out.begin(), out.end());
    for (auto &shape : inputs) result.insert(result.end(), shape.begin(), shape.end());
    return result;
}
static void run(void (*kernel)(CudaKernelArgs), CudaKernelArgs a) {
    for (et_u64 i = 0; i < a.elements; ++i) {
        blockIdx.x = i / 256; threadIdx.x = i % 256; kernel(a);
    }
    blockIdx.x = threadIdx.x = 0;
}
static void casts() {
    static_assert(sizeof(CudaKernelArgs) == 360);
    static_assert(offsetof(CudaKernelArgs, scalars) == 248);
    static_assert(offsetof(CudaKernelArgs, input_dtypes) == 312);
    static_assert(offsetof(CudaKernelArgs, output_dtype) == 344);
    for (unsigned int bits = 0; bits < 65536; ++bits) {
        float value = et_half_float(bits);
        if (isnan(value)) { assert((et_to16(value, false) & 0x7fff) > 0x7c00); continue; }
        assert(et_to16(value, false) == bits);
        if (bits < 0x7bff) {
            double midpoint = ((double)value + et_half_float(bits + 1)) / 2;
            for (double probe : {std::nextafter(midpoint, -INFINITY), midpoint, std::nextafter(midpoint, INFINITY)}) {
                _Float16 reference = (_Float16)probe; unsigned short expected; memcpy(&expected, &reference, 2);
                assert(et_to16(probe, false) == expected);
            }
        }
        float bf = et_bfloat_float(bits);
        if (!isnan(bf)) assert(et_to16(bf, true) == bits);
    }
    assert(et_to16(70000.0f, false) == 0x7c00);
    assert(et_to16(-70000.0f, false) == 0xfc00);
    assert((et_to16(std::numeric_limits<float>::quiet_NaN(), true) & 0x7fff) > 0x7f80);
    et_i64 tricky = (1LL << 55) + (1LL << 47) + 1;
    assert(et_to16(tricky, true) == 0x5b01);
    assert(et_to16((double)tricky, true) == 0x5b00);
    assert(et_to16(1.0 + ldexp(1.0, -8) + ldexp(1.0, -40), true) == 0x3f81);
    et_i64 input[] = {(1LL << 60) + (1LL << 36) + 1, (1LL << 53) + 1, -1, (-0x7fffffffffffffffLL - 1)};
    float floats[4]; unsigned int unsigneds[4]; et_i64 copy[4];
    CudaKernelArgs a{}; a.elements = 4; a.inputs[0] = address(input); a.input_dtypes[0] = 4; a.output = address(floats); a.output_dtype = 1;
    run(et_convert, a); assert(__float_as_uint(floats[0]) == __float_as_uint((float)(1LL << 60)) + 1);
    a.output = address(unsigneds); a.output_dtype = 5; run(et_convert, a);
    assert(unsigneds[0] == 1 && unsigneds[1] == 1 && unsigneds[2] == 0xffffffffU && unsigneds[3] == 0);
    a.output = address(copy); a.output_dtype = 4; run(et_convert, a); assert(memcmp(copy, input, sizeof(input)) == 0);
    // Every source/destination pair writes the advertised width, with guards.
    for (unsigned int src = 0; src < 7; ++src) for (unsigned int dst = 0; dst < 7; ++dst) {
        alignas(8) unsigned char source[8]{}, output[24]; memset(output, 0xa5, sizeof(output));
        et_store(address(source), src, 0, 42U); a.elements = 1; a.inputs[0] = address(source); a.input_dtypes[0] = src;
        a.output = address(output + 8); a.output_dtype = dst; run(et_convert, a);
        assert(et_load<double>(a.output, dst, 0) == 42);
        for (int j = 0; j < 8; ++j) assert(output[j] == 0xa5);
        for (unsigned int j = 8 + et_bytes(dst); j < sizeof(output); ++j) assert(output[j] == 0xa5);
    }
}
static void integers_and_roles() {
    et_i64 x[] = {(1LL << 53) + 1, (1LL << 53) + 2}, y[] = {1}, result[2]; unsigned char masks[2];
    auto m = metadata({2}, {{2}, {1}}); unsigned int status = 0;
    CudaKernelArgs a{}; a.elements = 2; a.metadata = address(m.data()); a.scratch[3] = address(&status);
    a.inputs[0] = address(x); a.inputs[1] = address(y); a.input_dtypes[0] = a.input_dtypes[1] = a.output_dtype = a.compute_dtype = 4; a.output = address(result);
    run(et_binary, a); assert(result[0] == x[0] + 1 && result[1] == x[1] + 1);
    a.inputs[1] = address(x + 1); a.operation = 8; a.output_dtype = 6; a.output = address(masks); run(et_binary, a);
    assert(masks[0] == 1 && masks[1] == 0);
    auto wm = metadata({2}, {{2}, {2}, {1}}); a.metadata = address(wm.data()); a.inputs[0] = address(masks); a.input_dtypes[0] = 6;
    a.inputs[1] = address(x); a.input_dtypes[1] = 4; a.inputs[2] = address(y); a.input_dtypes[2] = 4; a.output = address(result); a.output_dtype = 4;
    run(et_where, a); assert(result[0] == x[0] && result[1] == 1);
    unsigned int indices[] = {1, 0}; auto im = metadata({2}, {{2}, {2}}); a.metadata = address(im.data());
    a.inputs[0] = address(x); a.input_dtypes[0] = 4; a.inputs[1] = address(indices); a.input_dtypes[1] = 5; a.operation = 3;
    run(et_index, a); assert(result[0] == x[1] && result[1] == x[0]);
    et_i64 invalid[] = {(1LL << 53) + 1}; a.inputs[1] = address(invalid); a.input_dtypes[1] = 4; a.elements = 1;
    run(et_index, a); assert(status == 1);
    for (unsigned int dtype : {4U, 5U}) for (unsigned int operation : {0U, 1U}) {
        alignas(8) unsigned char guarded[24]; memset(guarded, 0xa5, sizeof(guarded));
        auto am = metadata({}, {{2}}); a.metadata = address(am.data()); a.inputs[0] = address(x);
        a.input_dtypes[0] = 4; a.output = address(guarded + 8); a.output_dtype = dtype; a.operation = operation;
        run(et_index, a); assert(et_load<et_i64>(a.output, dtype, 0) == (operation == 0 ? 1 : 0));
        for (int j = 0; j < 8; ++j) assert(guarded[j] == 0xa5);
        for (unsigned int j = 8 + et_bytes(dtype); j < sizeof(guarded); ++j) assert(guarded[j] == 0xa5);
    }
    // Exact transport preserves NaN payloads and signed zero in every storage width.
    for (unsigned int dtype = 0; dtype < 7; ++dtype) {
        alignas(8) et_u64 source[] = {0x7ff8000000000001ULL, 0x8000000000000000ULL}, target[2]{};
        auto tm = metadata({2}, {{2}}); a.metadata = address(tm.data()); a.inputs[0] = address(source); a.output = address(target);
        a.output_dtype = a.input_dtypes[0] = dtype; a.elements = 2; a.operation = 0; run(et_reindex, a);
        assert(memcmp(source, target, 2 * et_bytes(dtype)) == 0);
    }
}
static void scalar_coercion() {
    for (unsigned int dtype : {2U, 3U}) for (int scalar_role = 0; scalar_role < 2; ++scalar_role) for (bool promoted : {false, true}) {
        float scalar = 1.0f + ldexpf(1.0f, dtype == 2 ? -11 : -8), tensor = 3, output = 0;
        unsigned short half_tensor = et_to16(tensor, dtype == 3);
        std::vector<std::vector<et_u64>> shapes(2); shapes[1 - scalar_role] = {1};
        auto m = metadata({1}, shapes); unsigned int status = 0;
        CudaKernelArgs a{}; a.elements = 1; a.operation = 2; a.compute_dtype = 1;
        a.metadata = address(m.data()); a.scratch[3] = address(&status);
        a.inputs[scalar_role] = address(&scalar); a.input_dtypes[scalar_role] = 1; a.integers[scalar_role] = dtype + 1;
        a.inputs[1 - scalar_role] = promoted ? address(&tensor) : address(&half_tensor); a.input_dtypes[1 - scalar_role] = promoted ? 1 : dtype;
        a.output = address(&output); a.output_dtype = 1;
        run(et_binary, a); assert(output == 3 && status == 0);
    }
}
static void compute_roles() {
    Compute logits[] = {1, 2, 3, 3, 2, 1}, output[6]{};
    unsigned int targets[] = {2, 0}, status = 0;
    auto m = metadata({}, {{2, 3}, {2}}); CudaKernelArgs a{};
    a.elements = 1; a.inputs[0] = address(logits); a.inputs[1] = address(targets); a.input_dtypes[1] = 5;
    a.output = address(output); a.metadata = address(m.data()); a.scratch[3] = address(&status); a.integers[0] = (et_u64)-1;
    run(et_cross_entropy, a); assert(fabs((double)output[0] - log(1 + exp(-1.0) + exp(-2.0))) < 1e-6 && status == 0);
    et_i64 ignored[] = {-1, -1}; a.inputs[1] = address(ignored); a.input_dtypes[1] = 4; run(et_cross_entropy, a); assert(status == 2);
    status = 0; ignored[0] = (1LL << 53) + 1; run(et_cross_entropy, a); assert(status == 1);
    // F64 module still receives U8 optimizer flags and F32 scalar operands.
    Compute param = 10, grad = 2, velocity = 50, updated = 0; float lr = .25f; unsigned char first = 1;
    a = {}; a.elements = 1; a.operation = 1; a.compute_dtype = sizeof(Compute) == 8 ? 0 : 1; a.output_dtype = a.compute_dtype;
    a.inputs[0] = address(&param); a.inputs[1] = address(&grad); a.inputs[2] = address(&velocity); a.inputs[3] = address(&first); a.inputs[4] = address(&lr);
    a.input_dtypes[0] = a.input_dtypes[1] = a.input_dtypes[2] = a.compute_dtype; a.input_dtypes[3] = 6; a.input_dtypes[4] = 1;
    a.output = address(&updated); a.scalars[0] = .9; run(et_optimizer, a); assert(updated == 9.5);

    Compute q[] = {2, 3}, k[] = {.5, .25}, v[] = {4, 8}, decay[] = {0, 0}, beta[] = {1, 1}, kda_grad[] = {1, 1};
    Compute history[3]{}, grad_state[2]{}, kda_output[2]{};
    float persisted[2] = {0, 12345}; unsigned int valid = 2;
    auto km = metadata({1, 1, 2, 1}, {{1, 1, 2, 1}, {1, 1, 2, 1}, {1, 1, 2, 1}, {1, 1, 2, 1}, {1, 1, 2}, {1, 1, 2, 1}});
    a = {}; a.elements = 2; a.metadata = address(km.data()); a.output = address(kda_output); a.scalars[0] = 1;
    a.inputs[0] = address(q); a.inputs[1] = address(k); a.inputs[2] = address(v); a.inputs[3] = address(decay); a.inputs[4] = address(beta); a.inputs[5] = address(kda_grad);
    a.scratch[0] = address(persisted); a.scratch[2] = address(&valid); a.integers[1] = 1;
    run(et_kda, a); assert(kda_output[0] == 4 && kda_output[1] == 11.625 && persisted[0] == 3.875f && persisted[1] == 12345);
    a.integers[0] = 1; a.integers[1] = 0; a.scratch[0] = address(history); a.scratch[1] = address(grad_state);
    const Compute expected[5][2] = {{2, 3.875}, {19.25, 21}, {2.40625, .75}, {0, 5.625}, {9.625, 5.625}};
    for (unsigned int role = 0; role < 5; ++role) {
        a.operation = role; run(et_kda, a);
        assert(kda_output[0] == expected[role][0] && kda_output[1] == expected[role][1]);
    }
}
static void packed_fixtures(const char *path) {
    if (!path) return;
    std::ifstream file(path, std::ios::binary); assert(file.good());
    unsigned int codec;
    while (file.read((char *)&codec, 4)) {
        unsigned int bytes = kquant_block_bytes(codec); std::vector<unsigned char> block(bytes);
        float expected[256], decoded[256], input[256], output;
        file.read((char *)block.data(), bytes); file.read((char *)expected, sizeof(expected)); assert(file.good());
        for (unsigned int i = 0; i < 256; ++i) {
            decoded[i] = kquant_value(block.data(), i, codec);
            assert(__float_as_uint(decoded[i]) == __float_as_uint(expected[i]));
            input[i] = sinf(i * .1234f) + i * .00031f;
        }
        CudaKernelArgs a{}; a.elements = 1; a.inputs[0] = address(input); a.inputs[1] = address(block.data()); a.output = address(&output);
        a.integers[0] = codec; a.integers[1] = 1; a.integers[2] = 256; a.integers[3] = bytes;
        run(et_quantized_linear, a); float reference = 0; for (int i = 0; i < 256; ++i) reference += input[i] * expected[i];
        assert(output == reference);
        et_i64 index = 0; a.inputs[0] = address(&index); a.input_dtypes[0] = 4; a.elements = 256; a.output = address(decoded);
        run(et_quantized_embedding, a); assert(memcmp(expected, decoded, sizeof(expected)) == 0);
        unsigned int uindex = 0; a.inputs[0] = address(&uindex); a.input_dtypes[0] = 5;
        run(et_quantized_embedding, a); assert(memcmp(expected, decoded, sizeof(expected)) == 0);
    }
}
static void cache_storage() {
    float q[] = {0, 0}, k[] = {1, -2}, v[] = {3, -4}, output[2]{};
    unsigned int valid = 1, cursor = 0, status = 0;
    auto m = metadata({1, 1, 1, 2}, {{1, 1, 1, 2}, {1, 1, 1, 2}, {1, 1, 1, 2}});
    for (unsigned int dtype : {1U, 2U, 3U, 6U}) {
        alignas(8) unsigned char keys[16], values[16]; memset(keys, 0xcd, sizeof(keys)); memset(values, 0xcd, sizeof(values));
        float ks = 0, vs = 0; CudaKernelArgs a{}; a.elements = 2; a.metadata = address(m.data()); a.output = address(output); a.output_dtype = 1;
        a.inputs[0] = address(q); a.inputs[1] = address(k); a.inputs[2] = address(v); a.inputs[3] = address(keys); a.inputs[4] = address(values);
        a.inputs[5] = address(&ks); a.inputs[6] = address(&vs); a.inputs[7] = address(&cursor); a.scratch[0] = address(&valid); a.scratch[3] = address(&status);
        a.input_dtypes[0] = a.input_dtypes[1] = a.input_dtypes[2] = 1;
        a.integers[0] = 1; a.integers[1] = dtype; a.integers[2] = 1; a.integers[5] = 1; a.scalars[0] = 1;
        // Host shims cannot emulate warp synchronization. Check the scalar
        // storage helpers here; cache-tests.py executes attention on the GPU.
        et_cache_append(a, 0, 0, 1, 1, 2);
        for (unsigned int d = 0; d < 2; ++d) output[d] = et_cache_load(a, 4, 0, d, 2);
        assert(status == 0);
        for (unsigned int i = 2 * et_bytes(dtype); i < sizeof(keys); ++i) assert(keys[i] == 0xcd && values[i] == 0xcd);
        if (dtype == 6) {
            assert(keys[0] == 192 && keys[1] == 1 && ks == 2.0f / 127 && vs == 4.0f / 127);
            assert(fabsf(output[0] - 3) <= vs && output[1] == -4);
        } else assert(output[0] == 3 && output[1] == -4);
    }
}
static void linear_bias_rounding() {
    float accumulator[] = {257, 257};
    unsigned short bias[] = {0x3f40, 0xc380}; // 0.75, -256 in BF16.
    unsigned short output[] = {0xabcd, 0, 0, 0xabcd};
    CudaKernelArgs a{}; a.elements = 2; a.integers[0] = 2;
    a.inputs[0] = address(accumulator); a.inputs[1] = address(bias); a.output = address(output + 1);
    a.input_dtypes[0] = 1; a.input_dtypes[1] = 3; a.output_dtype = 3;
    run(et_linear_bias, a);
    assert(output[0] == 0xabcd && output[3] == 0xabcd);
    assert(output[1] == 0x4381 && output[2] == 0x3f80); // 258, 1.
}
int main(int argc, char **argv) {
    casts(); integers_and_roles(); scalar_coercion(); compute_roles(); cache_storage(); linear_bias_rounding(); packed_fixtures(argc > 1 ? argv[1] : nullptr);
    puts("CUDA host scalar/ABI tests passed");
}
