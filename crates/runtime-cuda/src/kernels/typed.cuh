// This ABI is shared with executable::CudaKernelArgs. Never compile it under
// the compute module's double-to-float macro. Addresses are CUDA device pointers.
typedef unsigned long long et_u64;
typedef long long et_i64;
struct CudaKernelArgs {
    et_u64 inputs[8];
    et_u64 output;
    et_u64 scratch[4];
    et_u64 metadata;
    et_u64 elements;
    et_u64 integers[16];
    double scalars[8];
    unsigned int input_dtypes[8];
    unsigned int output_dtype;
    unsigned int compute_dtype;
    unsigned int operation;
    unsigned int reserved;
};
static_assert(sizeof(CudaKernelArgs) == 360, "CUDA descriptor ABI size");

__device__ et_u64 et_thread() { return (et_u64)blockIdx.x * blockDim.x + threadIdx.x; }
__device__ const et_u64 *et_meta(const CudaKernelArgs &a) { return (const et_u64 *)a.metadata; }
__device__ const et_u64 *et_shape(const CudaKernelArgs &a, int role = -1) {
    const et_u64 *m = et_meta(a);
    et_u64 offset = 9;
    for (int i = -1; i < role; ++i) offset += m[i + 1];
    return m + offset;
}
__device__ const et_u64 *et_tail(const CudaKernelArgs &a) { return et_shape(a, 7) + et_meta(a)[8]; }
__device__ et_u64 et_numel(const CudaKernelArgs &a, int role) {
    et_u64 n = 1;
    const et_u64 *s = et_shape(a, role);
    for (et_u64 d = 0; d < et_meta(a)[role + 1]; ++d) n *= s[d];
    return n;
}
__device__ et_u64 et_broadcast(const CudaKernelArgs &a, et_u64 index, int role) {
    const et_u64 *out = et_shape(a), *in = et_shape(a, role);
    int rank = et_meta(a)[0], irank = et_meta(a)[role + 1];
    et_u64 result = 0, stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
        et_u64 c = index % out[d]; index /= out[d];
        int id = d - rank + irank;
        if (id >= 0) { if (in[id] != 1) result += c * stride; stride *= in[id]; }
    }
    return result;
}
__device__ void et_error(const CudaKernelArgs &a, unsigned int code) {
    if (a.scratch[3]) atomicCAS((unsigned int *)a.scratch[3], 0U, code);
}
__device__ unsigned int et_bytes(unsigned int dtype) {
    return dtype == 0 || dtype == 4 ? 8 : dtype == 1 || dtype == 5 ? 4 : dtype == 6 ? 1 : 2;
}
__device__ float et_half_float(unsigned short bits) {
#ifdef ET_HOST_TEST
    _Float16 value;
    memcpy(&value, &bits, sizeof(bits));
    return (float)value;
#else
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
#endif
}
__device__ float et_bfloat_float(unsigned short bits) { return __uint_as_float((unsigned int)bits << 16); }

// Round an exact nonnegative significand * 2^power to a 16-bit float.
// This also handles integer and F64 casts without an F32/F64 intermediate.
__device__ et_u64 et_round_shift(et_u64 v, int shift) {
    if (shift <= 0) return v << -shift;
    if (shift > 64) return 0;
    if (shift == 64) return v > (1ULL << 63);
    et_u64 q = v >> shift, mask = (1ULL << shift) - 1, half = 1ULL << (shift - 1);
    return q + ((v & mask) > half || ((v & mask) == half && (q & 1)));
}
__device__ unsigned short et_pack16(et_u64 sig, int power, unsigned int sign, bool bf) {
    if (!sig) return sign;
    int mantissa = bf ? 7 : 10, bias = bf ? 127 : 15;
    int top = 63 - __clzll(sig), exponent = top + power;
    int minimum = 1 - bias, maximum = bias;
    unsigned int inf = bf ? 0x7f80U : 0x7c00U;
    if (exponent > maximum) return sign | inf;
    if (exponent < minimum) {
        et_u64 fraction = et_round_shift(sig, minimum - mantissa - power);
        return sign | (unsigned int)fraction;
    }
    et_u64 fraction = et_round_shift(sig, top - mantissa);
    if (fraction == (1ULL << (mantissa + 1))) { fraction >>= 1; ++exponent; }
    if (exponent > maximum) return sign | inf;
    return sign | ((exponent + bias) << mantissa) | ((unsigned int)fraction & ((1U << mantissa) - 1));
}
__device__ unsigned short et_to16(double x, bool bf) {
    et_u64 bits = __double_as_longlong(x);
    unsigned int sign = (bits >> 48) & 0x8000;
    unsigned int exp = (bits >> 52) & 2047;
    et_u64 sig = bits & 0xfffffffffffffULL;
    if (exp == 2047) return sign | (bf ? 0x7f80 : 0x7c00) | (sig ? (bf ? 0x40 : 0x200) : 0);
    return et_pack16(exp ? sig | (1ULL << 52) : sig, exp ? (int)exp - 1075 : -1074, sign, bf);
}
__device__ unsigned short et_to16(et_i64 x, bool bf) {
    et_u64 magnitude = x < 0 ? 0ULL - (et_u64)x : (et_u64)x;
    return et_pack16(magnitude, 0, x < 0 ? 0x8000 : 0, bf);
}
__device__ unsigned short et_to16(unsigned int x, bool bf) { return et_pack16(x, 0, 0, bf); }
__device__ unsigned short et_to16(unsigned char x, bool bf) { return et_pack16(x, 0, 0, bf); }
__device__ unsigned short et_to16(float x, bool bf) { return et_to16((double)x, bf); }

template<class T> __device__ T et_load(et_u64 p, unsigned int dtype, et_u64 i) {
    switch (dtype) {
        case 0: return (T)((const double *)p)[i];
        case 1: return (T)((const float *)p)[i];
        case 2: return (T)et_half_float(((const unsigned short *)p)[i]);
        case 3: return (T)et_bfloat_float(((const unsigned short *)p)[i]);
        case 4: return (T)((const et_i64 *)p)[i];
        case 5: return (T)((const unsigned int *)p)[i];
        default: return (T)((const unsigned char *)p)[i];
    }
}
__device__ et_i64 et_int(double x) {
    if (isnan(x)) return 0;
    if (x >= 9223372036854775808.0) return 0x7fffffffffffffffLL;
    if (x <= -9223372036854775808.0) return (-0x7fffffffffffffffLL - 1);
    return (et_i64)x;
}
template<class T> __device__ et_i64 et_int(T x) { return (et_i64)x; }
__device__ et_i64 et_int(float x) { return et_int((double)x); }
template<class T> __device__ unsigned int et_uint(T x, unsigned int maximum) { return (unsigned int)x & maximum; }
__device__ unsigned int et_uint(double x, unsigned int maximum) {
    return isnan(x) || x <= 0 ? 0 : x >= maximum ? maximum : (unsigned int)x;
}
__device__ unsigned int et_uint(float x, unsigned int maximum) { return et_uint((double)x, maximum); }
template<class T> __device__ void et_store(et_u64 p, unsigned int dtype, et_u64 i, T x) {
    switch (dtype) {
        case 0: ((double *)p)[i] = (double)x; break;
        case 1: ((float *)p)[i] = (float)x; break;
        case 2: ((unsigned short *)p)[i] = et_to16(x, false); break;
        case 3: ((unsigned short *)p)[i] = et_to16(x, true); break;
        case 4: ((et_i64 *)p)[i] = et_int(x); break;
        case 5: ((unsigned int *)p)[i] = et_uint(x, 0xffffffffU); break;
        case 6: ((unsigned char *)p)[i] = et_uint(x, 255U); break;
    }
}
__device__ void et_copy(et_u64 from, et_u64 to, unsigned int dtype, et_u64 si, et_u64 di) {
    switch (et_bytes(dtype)) {
        case 8: ((et_u64 *)to)[di] = ((const et_u64 *)from)[si]; break;
        case 4: ((unsigned int *)to)[di] = ((const unsigned int *)from)[si]; break;
        case 2: ((unsigned short *)to)[di] = ((const unsigned short *)from)[si]; break;
        default: ((unsigned char *)to)[di] = ((const unsigned char *)from)[si]; break;
    }
}
__device__ bool et_selected(const CudaKernelArgs &a, int role, et_u64 i, et_u64 bound, et_u64 *out) {
    et_i64 v = et_load<et_i64>(a.inputs[role], a.input_dtypes[role], i);
    if (v < 0 || (et_u64)v >= bound) { et_error(a, 1); return false; }
    *out = (et_u64)v; return true;
}
