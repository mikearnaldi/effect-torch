// Isolated reconstruction of cuBLAS's BF16 split-K partial-rounding schedule.
// Compile: nvcc -O3 -arch=sm_120 -lcublas -lcuda expert-splitk.cu -o expert-splitk
// This is a diagnostic experiment, not a production inference kernel.
#define main reference_benchmark_main
#include "expert-gemm-batching.cu"
#undef main
#include <mma.h>
#include <cstring>

using namespace nvcuda;

// The reference transposed GEMM is W[N,K] * X^T[K,M] -> C[N,M],
// with row-major A, column-major B and column-major C. Preserve that orientation
// and the Tensor Core reduction order, then round every split to BF16.
__global__ void partial_gemm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* partials, int rows, int columns, int inner, int slices, int slice_width) {
  __shared__ __align__(16) __nv_bfloat16 a[32 * 16];
  __shared__ __align__(16) __nv_bfloat16 b[32 * 16];
  __shared__ __align__(16) float c[32 * 32];
  int lane = threadIdx.x, warp = lane / 32;
  int nr = blockIdx.x * 32, mr = blockIdx.y * 32, split = blockIdx.z;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  int begin = split * slice_width, end = min(inner, begin + slice_width);
  for (int start = begin; start < end; start += 16) {
    for (int index = lane; index < 32 * 16; index += blockDim.x) {
      int outer = index / 16, reduced = start + index % 16;
      a[index] = nr + outer < columns && reduced < end ? weight[(nr + outer) * inner + reduced] : __float2bfloat16(0);
      b[index] = mr + outer < rows && reduced < end ? x[(mr + outer) * inner + reduced] : __float2bfloat16(0);
    }
    __syncthreads();
    wmma::load_matrix_sync(af, a + (warp % 2) * 16 * 16, 16);
    wmma::load_matrix_sync(bf, b + (warp / 2) * 16 * 16, 16);
    wmma::mma_sync(acc, af, bf, acc);
    __syncthreads();
  }
  wmma::store_matrix_sync(c + (warp % 2) * 16 + (warp / 2) * 16 * 32, acc, 32, wmma::mem_col_major);
  __syncthreads();
  for (int index = lane; index < 32 * 32; index += blockDim.x) {
    int n = nr + index % 32, m = mr + index / 32;
    if (n < columns && m < rows) {
      partials[(size_t(split) * rows + m) * columns + n] = __float2bfloat16_rn(c[index]);
    }
  }
}

__global__ void reduce_partials(const __nv_bfloat16* partials, __nv_bfloat16* out,
    size_t count, int slices, int order) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
    float value = 0.0f;
    if (order == 0) {
      for (int split = 0; split < slices; ++split) value += __bfloat162float(partials[size_t(split) * count + index]);
    } else {
      for (int split = slices - 1; split >= 0; --split) value += __bfloat162float(partials[size_t(split) * count + index]);
    }
    out[index] = __float2bfloat16_rn(value);
  }
}

void describe(cublasHandle_t handle, cudaStream_t stream, int rows, int columns, int inner,
    void* x, void* weight, void* output) {
  float alpha = 1, beta = 0;
  auto gemm = [&]() {
    BLAS(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, inner,
      &alpha, weight, CUDA_R_16BF, inner, 0, x, CUDA_R_16BF, inner, size_t(rows) * inner,
      &beta, output, CUDA_R_16BF, columns, size_t(rows) * columns, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  gemm(); CUDA(cudaStreamSynchronize(stream));
  CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); gemm();
  cudaGraph_t graph; CUDA(cudaStreamEndCapture(stream, &graph));
  size_t count = 0; CUDA(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count); CUDA(cudaGraphGetNodes(graph, nodes.data(), &count));
  for (auto node : nodes) {
    cudaGraphNodeType type; CUDA(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) continue;
    CUDA_KERNEL_NODE_PARAMS params;
    if (cuGraphKernelNodeGetParams(node, &params) != CUDA_SUCCESS) std::exit(3);
    const char* name = "unknown"; cuFuncGetName(&name, params.func);
    if (!params.kernelParams) {
      std::fprintf(stderr, "kernel %s uses packed launch arguments; cannot dump argument array\n", name);
      continue;
    }
    std::printf("{\"kernelName\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"arguments\":[",
      name, rows, columns, inner, params.gridDimX, params.gridDimY, params.gridDimZ, params.blockDimX, params.blockDimY, params.blockDimZ);
    for (size_t index = 0; index < 64; ++index) {
      size_t offset, length; if (cuFuncGetParamInfo(params.func, index, &offset, &length) != CUDA_SUCCESS) break;
      std::printf("%s{\"bytes\":%zu,\"u32\":[", index ? "," : "", length);
      for (size_t word = 0; word < length / 4; ++word) {
        unsigned value; std::memcpy(&value, static_cast<char*>(params.kernelParams[index]) + word * 4, 4);
        std::printf("%s%u", word ? "," : "", value);
      }
      std::printf("]}");
    }
    std::printf("]}\n");
  }
  CUDA(cudaGraphDestroy(graph));
}

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: %s expert-fixture-directory\n", argv[0]); return 1; }
  cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle; BLAS(cublasCreate(&handle)); BLAS(cublasSetStream(handle, stream));
  void* workspace; CUDA(cudaMalloc(&workspace, workspace_bytes));
  BLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)); BLAS(cublasSetWorkspace(handle, workspace, workspace_bytes));
  if (std::string(argv[1]) == "--map") {
    for (int projection : {0, 1}) {
      int columns = projection ? 2816 : 1408, inner = projection ? 704 : 2816;
      auto x = dense(size_t(512) * inner, 17), weight = dense(size_t(columns) * inner, 29);
      __nv_bfloat16 *dx, *dw, *out;
      CUDA(cudaMalloc(&dx, x.size() * 2)); CUDA(cudaMalloc(&dw, weight.size() * 2));
      CUDA(cudaMalloc(&out, size_t(512) * columns * 2));
      CUDA(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      CUDA(cudaMemcpy(dw, weight.data(), weight.size() * 2, cudaMemcpyHostToDevice));
      for (int rows = 2; rows <= 512; ++rows) describe(handle, stream, rows, columns, inner, dx, dw, out);
      CUDA(cudaFree(dx)); CUDA(cudaFree(dw)); CUDA(cudaFree(out));
    }
    BLAS(cublasDestroy(handle)); CUDA(cudaFree(workspace)); CUDA(cudaStreamDestroy(stream)); return 0;
  }
  for (const char* name : {"decoder-2", "decoder-0", "encoder-1"}) {
    int rows = name[0] == 'e' ? 262 : name[8] == '2' ? 17 : 149;
    int columns = name[0] == 'e' ? 2816 : 1408, inner = name[0] == 'e' ? 704 : 2816;
    auto x = load(std::string(argv[1]) + "/" + name + ".input.bf16", size_t(rows) * inner);
    auto weight = load(std::string(argv[1]) + "/" + name + ".weight.bf16", size_t(columns) * inner);
    auto reference = load(std::string(argv[1]) + "/" + name + ".official.bf16", size_t(rows) * columns);
    __nv_bfloat16 *dx, *dw, *out, *partials;
    CUDA(cudaMalloc(&dx, x.size() * 2)); CUDA(cudaMalloc(&dw, weight.size() * 2));
    CUDA(cudaMalloc(&out, reference.size() * 2)); CUDA(cudaMalloc(&partials, reference.size() * 2 * 32));
    CUDA(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dw, weight.data(), weight.size() * 2, cudaMemcpyHostToDevice));
    describe(handle, stream, rows, columns, inner, dx, dw, out);
    std::vector<unsigned short> actual(reference.size());
    for (int slices : {2, 3, 4, 5, 6, 7, 8, 11, 16, 22}) {
      for (int grain : {16, 32, 64, 128}) {
        int width = ((inner + slices * grain - 1) / (slices * grain)) * grain;
        partial_gemm<<<dim3((columns + 31) / 32, (rows + 31) / 32, slices), 128, 0, stream>>>(dx, dw, partials, rows, columns, inner, slices, width);
        CUDA(cudaGetLastError());
        for (int order : {0, 1}) {
          reduce_partials<<<128, 256, 0, stream>>>(partials, out, reference.size(), slices, order);
          CUDA(cudaGetLastError()); CUDA(cudaStreamSynchronize(stream));
          CUDA(cudaMemcpy(actual.data(), out, actual.size() * 2, cudaMemcpyDeviceToHost));
          size_t exact = 0; for (size_t index = 0; index < actual.size(); ++index) exact += actual[index] == reference[index];
          std::printf("{\"case\":\"%s\",\"slices\":%d,\"width\":%d,\"grain\":%d,\"reverseReduction\":%s,\"exact\":%zu,\"elements\":%zu}\n",
            name, slices, width, grain, order ? "true" : "false", exact, actual.size());
        }
      }
    }
    CUDA(cudaFree(dx)); CUDA(cudaFree(dw)); CUDA(cudaFree(out)); CUDA(cudaFree(partials));
  }
  BLAS(cublasDestroy(handle)); CUDA(cudaFree(workspace)); CUDA(cudaStreamDestroy(stream));
  return 0;
}
