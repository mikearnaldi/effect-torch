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
// Compile-only stubs. Grouping scheduling is covered by the CUDA hardware tests.
static unsigned int __ballot_sync(unsigned int, bool selected) { return selected ? 1U : 0U; }
static unsigned int __popc(unsigned int value) { return __builtin_popcount(value); }
static unsigned int atomicAdd(unsigned int *p, unsigned int value) { unsigned int old = *p; *p += value; return old; }
static unsigned int atomicExch(unsigned int *p, unsigned int value) { unsigned int old = *p; *p = value; return old; }
template<class T> T atomicCAS(T *p, T expected, T value) {
    T old = *p; if (old == expected) *p = value; return old;
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
    auto m = metadata({2}, {{2}, {1}}); et_u64 status = 0;
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
static void half_scatter_updates() {
    for (unsigned int dtype : {2U, 3U}) {
        float base = dtype == 2 ? 2048.0f : 256.0f;
        unsigned short input[] = {et_to16(base, dtype == 3), et_to16(-base, dtype == 3), et_to16(7.0f, dtype == 3)};
        unsigned short source[] = {et_to16(1.0f, dtype == 3), et_to16(-1.0f, dtype == 3), et_to16(1.0f, dtype == 3), et_to16(-1.0f, dtype == 3)};
        unsigned int indices[] = {0, 1, 0, 1}; et_u64 status = 0;
        unsigned short guarded[] = {0xdead, 0, 0, 0, 0xbeef};
        auto m = metadata({3}, {{3}, {4}, {4}});
        CudaKernelArgs a{}; a.elements = 3; a.operation = 5;
        a.inputs[0] = address(input); a.inputs[1] = address(indices); a.inputs[2] = address(source);
        a.input_dtypes[0] = a.input_dtypes[2] = a.output_dtype = dtype; a.input_dtypes[1] = 5;
        a.output = address(guarded + 1); a.metadata = address(m.data()); a.scratch[3] = address(&status);
        run(et_index, a);
        // Each unit is a half-ULP update and ties back to the base. A widened
        // reduction would instead produce base+2 and -base-2.
        assert(status == 0 && guarded[0] == 0xdead && guarded[4] == 0xbeef);
        assert(et_load<float>(a.output, dtype, 0) == base);
        assert(et_load<float>(a.output, dtype, 1) == -base);
        assert(et_load<float>(a.output, dtype, 2) == 7);
        indices[3] = 0xffffffffU; run(et_index, a); assert(status == 1);
        status = 0; indices[3] = 1; run(et_index, a); assert(status == 0);
        assert(memcmp(guarded + 1, input, sizeof(input)) == 0);
    }
}
static void scalar_coercion() {
    for (unsigned int dtype : {2U, 3U}) for (int scalar_role = 0; scalar_role < 2; ++scalar_role) for (bool promoted : {false, true}) {
        float scalar = 1.0f + ldexpf(1.0f, dtype == 2 ? -11 : -8), tensor = 3, output = 0;
        unsigned short half_tensor = et_to16(tensor, dtype == 3);
        std::vector<std::vector<et_u64>> shapes(2); shapes[1 - scalar_role] = {1};
        auto m = metadata({1}, shapes); et_u64 status = 0;
        CudaKernelArgs a{}; a.elements = 1; a.operation = 2; a.compute_dtype = 1;
        a.metadata = address(m.data()); a.scratch[3] = address(&status);
        a.inputs[scalar_role] = address(&scalar); a.input_dtypes[scalar_role] = 1; a.integers[scalar_role] = dtype + 1;
        a.inputs[1 - scalar_role] = promoted ? address(&tensor) : address(&half_tensor); a.input_dtypes[1 - scalar_role] = promoted ? 1 : dtype;
        a.output = address(&output); a.output_dtype = 1;
        run(et_binary, a); assert(output == 3 && status == 0);
    }
}
static void binary_broadcast_indexing() {
    for (et_u64 width : {7ULL, 8ULL}) {
        std::vector<float> x(6 * width), y(6), baseline(x.size()), optimized(x.size());
        for (et_u64 i = 0; i < x.size(); ++i) x[i] = (int(i) - 17) * 0.03125f;
        for (et_u64 i = 0; i < y.size(); ++i) y[i] = (i + 1) * 0.125f;
        auto m = metadata({2, 3, width}, {{2, 3, width}, {2, 3, 1}});
        CudaKernelArgs a{}; a.elements = x.size(); a.operation = 3;
        a.metadata = address(m.data()); a.input_dtypes[0] = a.input_dtypes[1] = 1;
        a.output_dtype = a.compute_dtype = 1;
        a.inputs[0] = address(x.data()); a.inputs[1] = address(y.data());
        a.output = address(baseline.data()); run(et_binary, a);
        a.integers[2] = 1; a.integers[3] = width == 8 ? 4 : 3;
        a.integers[5] = width == 8 ? 3 : width;
        a.output = address(optimized.data()); run(et_binary, a);
        assert(memcmp(baseline.data(), optimized.data(), x.size() * sizeof(float)) == 0);
        for (et_u64 i = 0; i < x.size(); ++i) assert(optimized[i] == x[i] / y[i / width]);
    }
    for (unsigned int dtype : {2U, 3U}) for (int scalar_role = 0; scalar_role < 2; ++scalar_role) {
        float scalar = 1.0f + ldexpf(1.0f, dtype == 2 ? -11 : -8), output[2];
        unsigned short tensor[] = {et_to16(3.0f, dtype == 3), et_to16(-5.0f, dtype == 3)};
        std::vector<std::vector<et_u64>> shapes(2); shapes[1 - scalar_role] = {2};
        auto m = metadata({2}, shapes); CudaKernelArgs a{};
        a.elements = 2; a.operation = 2; a.compute_dtype = a.output_dtype = 1;
        a.metadata = address(m.data()); a.output = address(output);
        a.inputs[scalar_role] = address(&scalar); a.input_dtypes[scalar_role] = 1;
        a.integers[scalar_role] = dtype + 1; a.integers[2 + scalar_role] = 2;
        a.inputs[1 - scalar_role] = address(tensor); a.input_dtypes[1 - scalar_role] = dtype;
        a.integers[3 - scalar_role] = 1; run(et_binary, a);
        assert(output[0] == 3.0f && output[1] == -5.0f);
    }
}
static void compute_roles() {
    Compute logits[] = {1, 2, 3, 3, 2, 1}, output[6]{};
    unsigned int targets[] = {2, 0}; et_u64 status = 0;
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
    unsigned int valid = 1, cursor = 0; et_u64 status = 0;
    auto m = metadata({1, 1, 1, 2}, {{1, 1, 1, 2}, {1, 1, 1, 2}, {1, 1, 1, 2}});
    for (unsigned int dtype : {1U, 2U, 3U, 6U}) {
        alignas(8) unsigned char keys[16], values[16]; memset(keys, 0xcd, sizeof(keys)); memset(values, 0xcd, sizeof(values));
        float ks = 0, vs = 0; CudaKernelArgs a{}; a.elements = 2; a.metadata = address(m.data()); a.output = address(output); a.output_dtype = 1;
        a.inputs[0] = address(q); a.inputs[1] = address(k); a.inputs[2] = address(v); a.inputs[3] = address(keys); a.inputs[4] = address(values);
        et_u64 table[] = {0, 0, 1, 4, address(keys), address(values), address(&ks), address(&vs)};
        a.inputs[3] = address(table); a.inputs[7] = address(&cursor); a.scratch[0] = address(&valid); a.scratch[3] = address(&status);
        a.input_dtypes[0] = a.input_dtypes[1] = a.input_dtypes[2] = 1;
        a.integers[0] = 1; a.integers[1] = dtype; a.integers[2] = 1; a.integers[5] = 1; a.scalars[0] = 1;
        // Host shims cannot emulate warp synchronization. Check the scalar
        // storage helpers here; cache-tests.py executes attention on the GPU.
        et_cache_append(a, 0, 0, 1, 1, 2);
        for (unsigned int d = 0; d < 2; ++d) output[d] = et_cache_load(a, 1, 0, 0, 0, d, 2);
        assert(status == 0);
        for (unsigned int i = 2 * et_bytes(dtype); i < sizeof(keys); ++i) assert(keys[i] == 0xcd && values[i] == 0xcd);
        if (dtype == 6) {
            assert(keys[0] == 192 && keys[1] == 1 && ks == 2.0f / 127 && vs == 4.0f / 127);
            assert(fabsf(output[0] - 3) <= vs && output[1] == -4);
        } else assert(output[0] == 3 && output[1] == -4);
    }
}
// Executes the production scalar math with a single-lane host reduction. CUDA
// scheduling and shuffle behavior still require the separate GPU fixture.
static void cache_prefix_canvas() {
    for (unsigned int dtype : {1U, 2U, 3U}) for (bool causal : {false, true}) for (unsigned int retained : {0U, 2U}) {
        const unsigned int heads = 2, dim = 3, tokens = 3, count = 2, cursor = 44;
        unsigned int valid = count; et_u64 status = 0;
        auto m = metadata({1, heads, tokens, dim}, {{1, heads, tokens, dim}, {1, 1, tokens, dim}, {1, 1, tokens, dim}});
        alignas(8) unsigned char q[72]{}, k[36]{}, v[36]{}, prefix_k[24]{}, prefix_v[24]{}, tail_k[24]{}, tail_v[24]{};
        float out[heads * tokens * dim]{};
        for (unsigned int i = 0; i < heads * tokens * dim; ++i) et_store(address(q), dtype, i, sinf(i * .739f) * 3.7f);
        for (unsigned int i = 0; i < tokens * dim; ++i) {
            et_store(address(k), dtype, i, cosf(i * .313f) * 2.9f);
            et_store(address(v), dtype, i, sinf(i * 1.137f) * 5.3f);
        }
        for (unsigned int i = 0; i < 2 * dim; ++i) {
            et_store(address(prefix_k), dtype, i, sinf(i * .47f) * 1.3f);
            et_store(address(prefix_v), dtype, i, cosf(i * .81f) * 4.7f);
        }
        unsigned char before_k[sizeof(prefix_k)], before_v[sizeof(prefix_v)];
        memcpy(before_k, prefix_k, sizeof(prefix_k)); memcpy(before_v, prefix_v, sizeof(prefix_v));
        et_u64 table[20] = {cursor - retained, cursor, cursor + count, 4};
        for (unsigned int row = 0; row < retained + count; ++row) {
            bool prefix = row < retained;
            unsigned int source = prefix ? row + 2 - retained : row - retained;
            table[4 + row * 4] = address(prefix ? prefix_k : tail_k) + source * dim * et_bytes(dtype);
            table[5 + row * 4] = address(prefix ? prefix_v : tail_v) + source * dim * et_bytes(dtype);
        }
        CudaKernelArgs a{}; a.elements = heads * tokens * dim; a.compute_dtype = a.output_dtype = 1;
        a.inputs[0] = address(q); a.inputs[1] = address(k); a.inputs[2] = address(v); a.inputs[3] = address(table); a.inputs[7] = address(&cursor);
        a.input_dtypes[0] = a.input_dtypes[1] = a.input_dtypes[2] = dtype;
        a.scratch[0] = address(&valid); a.scratch[3] = address(&status); a.metadata = address(m.data()); a.output = address(out);
        a.integers[1] = dtype; a.integers[2] = a.integers[5] = 1; a.integers[4] = !causal; a.integers[6] = dtype != 1; a.integers[8] = dtype;
        a.scalars[0] = .30157f; blockIdx.x = threadIdx.x = 0;
        float probabilities[heads * tokens * 4]{};
        a.inputs[4] = address(probabilities); a.integers[9] = 4;
        for (unsigned int repetition = 0; repetition < 2; ++repetition) {
            for (unsigned int t = 0; t < tokens; ++t) { threadIdx.x = t; et_kv_store(a); }
            for (unsigned int query = 0; query < heads * tokens; ++query) { threadIdx.x = query * 32; et_kv_attention(a); }
            assert(status == 0 && memcmp(prefix_k, before_k, sizeof(prefix_k)) == 0 && memcmp(prefix_v, before_v, sizeof(prefix_v)) == 0);
            auto round = [dtype](float x) { unsigned short bits = et_to16(x, dtype == 3); return dtype == 1 ? x : dtype == 2 ? et_half_float(bits) : et_bfloat_float(bits); };
            for (unsigned int head = 0; head < heads; ++head) for (unsigned int t = 0; t < tokens; ++t) {
                unsigned int base = (head * tokens + t) * dim;
                if (t >= count) { for (unsigned int d = 0; d < dim; ++d) assert(out[base + d] == 0); continue; }
                unsigned int n = retained + (causal ? t + 1 : count);
                float scores[4]{}, weights[4]{}, maximum = -INFINITY, total = 0;
                for (unsigned int row = 0; row < n; ++row) {
                    bool prefix = row < retained; unsigned int source = prefix ? row + 2 - retained : row - retained;
                    float dot = 0;
                    for (unsigned int d = 0; d < dim; ++d) dot += et_load<float>(address(q), dtype, base + d) * et_load<float>(address(prefix ? prefix_k : k), dtype, source * dim + d);
                    scores[row] = round(round(dot) * float(a.scalars[0])); maximum = fmaxf(maximum, scores[row]);
                }
                for (unsigned int row = 0; row < n; ++row) { weights[row] = expf(scores[row] - maximum); total += weights[row]; }
                for (unsigned int d = 0; d < dim; ++d) {
                    float expected = 0;
                    for (unsigned int row = 0; row < n; ++row) {
                        bool prefix = row < retained; unsigned int source = prefix ? row + 2 - retained : row - retained;
                        expected += round(weights[row] / total) * et_load<float>(address(prefix ? prefix_v : v), dtype, source * dim + d);
                    }
                    expected = round(expected);
                    assert(fabsf(out[base + d] - expected) < 1e-6f);
                }
            }
            // A second invocation changes its canvas while sharing the same prefix.
            et_store(address(v), dtype, 0, 17.0f);
        }
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
// CUDA barriers cannot execute on the host. Run each disjoint network stage
// using the production comparator, then compare with an independent stable sort.
static void top_k_bitonic_network() {
    unsigned int state = 0x71ac4935U;
    for (unsigned int width : {128U, 256U}) for (unsigned int trial = 0; trial < 128; ++trial) {
        std::vector<float> values(width);
        std::vector<unsigned int> orders(width), indices(width), expected(width);
        for (unsigned int index = 0; index < width; ++index) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            unsigned int bits = state;
            if ((bits & 0x7f800000U) == 0x7f800000U) bits ^= 0x00800000U;
            values[index] = trial % 3 == 0 ? float(int(state % 9) - 4) : __uint_as_float(bits);
            indices[index] = expected[index] = index;
        }
        values[0] = -0.0f; values[1] = 0.0f;
        values[2] = INFINITY; values[3] = -INFINITY; values[4] = INFINITY;
        values[5] = std::numeric_limits<float>::denorm_min(); values[6] = -values[5];
        if (trial == 0) std::fill(values.begin(), values.end(), 0.0f);
        for (unsigned int index = 0; index < width; ++index) orders[index] = et_top_k_order(values[index]);
        for (unsigned int size = 2; size <= width; size <<= 1)
            for (unsigned int stride = size >> 1; stride; stride >>= 1)
                for (unsigned int index = 0; index < width; ++index) {
                    unsigned int other = index ^ stride;
                    if (index < other) et_top_k_pair(orders.data(), indices.data(), index, other, (index & size) == 0);
                }
        std::stable_sort(expected.begin(), expected.end(), [&](unsigned int a, unsigned int b) { return values[a] > values[b]; });
        assert(indices == expected);
    }
}
static void top_k_indices() {
    constexpr unsigned int width = 128, rows = 6;
    std::vector<float> scores(width * rows);
    for (unsigned int i = 0; i < scores.size(); ++i) scores[i] = int((i * 73 + 19) % 137) - 68;
    scores[0] = -0.0f; scores[1] = 0.0f;
    scores[2] = std::numeric_limits<float>::denorm_min();
    scores[3] = -scores[2];
    scores[4] = INFINITY; scores[5] = -INFINITY; scores[6] = INFINITY;
    for (unsigned int k : {1U, 8U, width}) {
        std::vector<unsigned int> output(rows * k + 2, 0xdeadbeefU);
        et_u64 status = 0;
        CudaKernelArgs a{}; a.elements = rows * k;
        a.inputs[0] = address(scores.data()); a.output = address(output.data() + 1);
        a.scratch[3] = address(&status); a.integers[0] = k; a.integers[1] = width;
        for (unsigned int row = 0; row < rows; ++row) {
            blockIdx.x = row; threadIdx.x = 0; et_top_k_indices(a);
            std::vector<unsigned int> expected(width);
            for (unsigned int i = 0; i < width; ++i) expected[i] = i;
            std::stable_sort(expected.begin(), expected.end(), [&](unsigned int i, unsigned int j) {
                return scores[row * width + i] > scores[row * width + j];
            });
            for (unsigned int i = 0; i < k; ++i) assert(output[1 + row * k + i] == expected[i]);
        }
        assert(status == 0 && output.front() == 0xdeadbeefU && output.back() == 0xdeadbeefU);
        float saved = scores.back(); scores.back() = NAN;
        et_top_k_indices(a); assert(status == 5); scores.back() = saved;
    }
    blockIdx.x = threadIdx.x = 0;
}
// Checks the actual fallback store, both layouts, inactive-row holes and
// redzones. Compare direct BF16 to the original F32 result plus boundary cast.
static void attention82_round() {
    constexpr unsigned int heads=2, tokens=3, declared=5, dim=8, active=heads*tokens*dim, extent=heads*declared*dim;
    for (unsigned int layout=0; layout<2; ++layout) for (unsigned int base=0; base<65536; base+=active) {
        float source[active]; float reference[extent+2]{};
        unsigned short direct[extent+2]{};
        for (unsigned int i=0; i<active; ++i) source[i]=__uint_as_float(((base+i)&65535U)<<16 | 0x8000U);
        direct[0]=direct[extent+1]=0xdead;
        reference[0]=reference[extent+1]=17.0f;
        CudaKernelArgs a{}; a.inputs[2]=address(source); a.elements=active;
        a.integers[7]=declared; a.integers[10]=dim; a.integers[11]=heads; a.integers[15]=tokens;
        a.operation=layout;
        for (unsigned int i=0; i<active; ++i) {
            blockIdx.x=i/256; threadIdx.x=i%256;
            a.output=address(reference+1); a.output_dtype=1; et_kv_gemm_round(a);
            a.output=address(direct+1); a.output_dtype=3; et_kv_gemm_round(a);
        }
        for (unsigned int i=0; i<extent; ++i) assert(direct[i+1]==et_to16(reference[i+1],true));
        assert(direct[0]==0xdead && direct[extent+1]==0xdead);
        assert(reference[0]==17.0f && reference[extent+1]==17.0f);
    }
    blockIdx.x=threadIdx.x=0;
}
int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--attention82-round") == 0) { attention82_round(); puts("attention82 fallback round passed"); return 0; }
    if (argc > 1 && strcmp(argv[1], "--top-k") == 0) {
        top_k_indices();
        top_k_bitonic_network();
        puts("CUDA host stable top-k tests passed");
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--binary-broadcast") == 0) {
        binary_broadcast_indexing();
        scalar_coercion();
        puts("CUDA host binary broadcasting tests passed");
        return 0;
    }
    top_k_indices();
    half_scatter_updates();
    binary_broadcast_indexing();
    casts(); integers_and_roles(); scalar_coercion(); compute_roles(); cache_storage(); cache_prefix_canvas(); linear_bias_rounding(); packed_fixtures(argc > 1 ? argv[1] : nullptr);
    puts("CUDA host scalar/ABI tests passed");
}
