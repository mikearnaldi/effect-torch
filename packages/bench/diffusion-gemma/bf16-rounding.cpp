// Exhaustive CPU proof: all 2^32 F32 encodings versus the unchanged F64 packer.
// c++ -O3 -std=c++17 -ffp-contract=off -pthread bf16-rounding.cpp -o /tmp/bf16-rounding
// /tmp/bf16-rounding [threads, default 8] [encoding limit, default 4294967296]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
using std::min;
using std::max;
using std::isnan;
using std::isfinite;
#define ET_HOST_TEST
#define ET_CUDA_F32_BF16_BITS 1
#define __device__
struct Dim { unsigned int x = 0, y = 0, z = 0; };
static Dim threadIdx, blockIdx, blockDim{256, 1, 1}, gridDim{1, 1, 1};
static unsigned int __float_as_uint(float value) { unsigned int bits; std::memcpy(&bits, &value, 4); return bits; }
static float __uint_as_float(unsigned int bits) { float value; std::memcpy(&value, &bits, 4); return value; }
static unsigned long long __double_as_longlong(double value) { unsigned long long bits; std::memcpy(&bits, &value, 8); return bits; }
static int __clzll(unsigned long long value) { return __builtin_clzll(value); }
template<class T> T atomicCAS(T *p, T expected, T value) { T old = *p; if (old == expected) *p = value; return old; }
#include "../../../crates/runtime-cuda/src/kernels/typed.cuh"

int main(int argc, char **argv) {
    const unsigned int threads = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 8;
    const unsigned long long limit = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : (1ULL << 32);
    if (!threads || threads > 256 || limit > (1ULL << 32)) return 2;
    std::atomic<unsigned long long> mismatches{0}, checked{0};
    std::vector<std::thread> workers;
    auto started = std::chrono::steady_clock::now();
    for (unsigned int worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&, worker] {
            unsigned long long bad = 0;
            const auto begin = limit * worker / threads, end = limit * (worker + 1) / threads;
            for (auto bits = begin; bits < end; ++bits) {
                float value = __uint_as_float((unsigned int)bits);
                unsigned short old = et_to16((double)value, true);
                unsigned short fast = et_to16(value, true);
                if (old != fast) {
                    if (bad < 4) std::fprintf(stderr, "bits=%08x old=%04x new=%04x\n", (unsigned int)bits, old, fast);
                    ++bad;
                }
            }
            mismatches += bad; checked += end - begin;
        });
    }
    for (auto& worker : workers) worker.join();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("{\"encodings\":%llu,\"mismatches\":%llu,\"threads\":%u,\"seconds\":%.6f,\"exhaustive\":%s}\n",
        checked.load(), mismatches.load(), threads, seconds, checked == (1ULL << 32) ? "true" : "false");
    return mismatches ? 1 : 0;
}
