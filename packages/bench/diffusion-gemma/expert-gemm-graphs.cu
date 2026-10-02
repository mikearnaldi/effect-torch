// Isolate the benefit of replaying the exact existing expert cuBLAS calls.
// This is a fixed-routing upper bound, not a full-model latency measurement.
// Compile in the pinned CUDA shell with nvcc and link -lcublas.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA(call) do { auto status = (call); if (status != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); std::exit(1); } } while (0)
#define BLAS(call) do { auto status = (call); if (status != CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr, "%s: status %d\n", #call, int(status)); std::exit(1); } } while (0)

constexpr int experts = 128, workers = 32;
constexpr size_t workspace_bytes = 32 << 20;

__global__ void initialize(__nv_bfloat16* values, size_t length, unsigned seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x) {
    unsigned bits = unsigned(i) + seed;
    bits = (bits ^ (bits >> 16)) * 0x7feb352du;
    bits = (bits ^ (bits >> 15)) * 0x846ca68bu;
    values[i] = __float2bfloat16(float(int((bits ^ (bits >> 16)) & 1023) - 512) / 512.0f);
  }
}

struct Worker {
  cudaStream_t stream;
  cublasHandle_t handle;
  cudaEvent_t done;
  void* workspace;
};

void benchmark(int rows, int columns, int inner) {
  __nv_bfloat16 *x, *weight, *output;
  size_t x_elements = size_t(experts) * rows * inner;
  size_t weight_elements = size_t(experts) * columns * inner;
  size_t output_elements = size_t(experts) * rows * columns;
  CUDA(cudaMalloc(&x, x_elements * 2));
  CUDA(cudaMalloc(&weight, weight_elements * 2));
  CUDA(cudaMalloc(&output, output_elements * 2));
  initialize<<<1024, 256>>>(x, x_elements, 123);
  initialize<<<1024, 256>>>(weight, weight_elements, 456);
  CUDA(cudaGetLastError());
  CUDA(cudaDeviceSynchronize());
  Worker worker[workers];
  for (auto& w : worker) {
    CUDA(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    CUDA(cudaEventCreateWithFlags(&w.done, cudaEventDisableTiming));
    CUDA(cudaMalloc(&w.workspace, workspace_bytes));
    BLAS(cublasCreate(&w.handle));
    BLAS(cublasSetStream(w.handle, w.stream));
    BLAS(cublasSetMathMode(w.handle, CUBLAS_DEFAULT_MATH));
    BLAS(cublasSetWorkspace(w.handle, w.workspace, workspace_bytes));
  }
  auto gemm = [&](int expert) {
    float alpha = 1.0f, beta = 0.0f;
    BLAS(cublasGemmStridedBatchedEx(worker[expert % workers].handle,
      CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, inner, &alpha,
      weight + size_t(expert) * columns * inner, CUDA_R_16BF, inner, 0,
      x + size_t(expert) * rows * inner, CUDA_R_16BF, inner, size_t(rows) * inner,
      &beta, output + size_t(expert) * rows * columns, CUDA_R_16BF,
      columns, size_t(rows) * columns, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  // Warm all handles before capture so startup cannot enter a graph.
  for (int i = 0; i < experts; ++i) gemm(i);
  CUDA(cudaDeviceSynchronize());
  std::vector<unsigned short> reference(output_elements), actual(output_elements);
  CUDA(cudaMemcpy(reference.data(), output, output_elements * 2, cudaMemcpyDeviceToHost));
  cudaGraphExec_t expert_graph[experts], stream_graph[workers];
  size_t expert_nodes = 0, stream_nodes = 0;
  auto capture = [&](int index, bool grouped) {
    int wi = index % workers;
    CUDA(cudaStreamBeginCapture(worker[wi].stream, cudaStreamCaptureModeThreadLocal));
    if (grouped) {
      for (int e = wi; e < experts; e += workers) gemm(e);
    } else gemm(index);
    cudaGraph_t graph;
    CUDA(cudaStreamEndCapture(worker[wi].stream, &graph));
    size_t nodes = 0;
    CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
    auto& executable = grouped ? stream_graph[index] : expert_graph[index];
    CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA(cudaGraphDestroy(graph));
    (grouped ? stream_nodes : expert_nodes) += nodes;
  };
  for (int i = 0; i < experts; ++i) capture(i, false);
  for (int i = 0; i < workers; ++i) capture(i, true);
  cudaStream_t primary;
  cudaEvent_t start, end, ready;
  CUDA(cudaStreamCreateWithFlags(&primary, cudaStreamNonBlocking));
  CUDA(cudaEventCreate(&start));
  CUDA(cudaEventCreate(&end));
  CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
  // Fork/join every expert stream into one graph with unchanged GEMMs.
  CUDA(cudaStreamBeginCapture(primary, cudaStreamCaptureModeThreadLocal));
  CUDA(cudaEventRecord(ready, primary));
  for (auto& w : worker) CUDA(cudaStreamWaitEvent(w.stream, ready, 0));
  for (int wi = 0; wi < workers; ++wi) {
    for (int e = wi; e < experts; e += workers) gemm(e);
    CUDA(cudaEventRecord(worker[wi].done, worker[wi].stream));
    CUDA(cudaStreamWaitEvent(primary, worker[wi].done, 0));
  }
  cudaGraph_t joined_capture;
  cudaGraphExec_t joined_graph;
  CUDA(cudaStreamEndCapture(primary, &joined_capture));
  size_t joined_nodes = 0;
  CUDA(cudaGraphGetNodes(joined_capture, nullptr, &joined_nodes));
  CUDA(cudaGraphInstantiate(&joined_graph, joined_capture, nullptr, nullptr, 0));
  CUDA(cudaGraphDestroy(joined_capture));
  auto run = [&](int mode) {
    CUDA(cudaEventRecord(start, primary));
    if (mode == 3) {
      CUDA(cudaGraphLaunch(joined_graph, primary));
    } else {
      for (auto& w : worker) CUDA(cudaStreamWaitEvent(w.stream, start, 0));
      for (int wi = 0; wi < workers; ++wi) {
        if (mode == 2) {
          CUDA(cudaGraphLaunch(stream_graph[wi], worker[wi].stream));
        } else {
          for (int e = wi; e < experts; e += workers) {
            if (mode == 1) CUDA(cudaGraphLaunch(expert_graph[e], worker[wi].stream));
            else gemm(e);
          }
        }
      }
      for (auto& w : worker) {
        CUDA(cudaEventRecord(w.done, w.stream));
        CUDA(cudaStreamWaitEvent(primary, w.done, 0));
      }
    }
    CUDA(cudaEventRecord(end, primary));
    CUDA(cudaEventSynchronize(end));
    float ms;
    CUDA(cudaEventElapsedTime(&ms, start, end));
    return ms;
  };
  std::vector<double> gpu[4], wall[4];
  for (int iteration = -5; iteration < 25; ++iteration) {
    // Rotate ordering to avoid giving one mode consistently warmer clocks.
    for (int offset = 0; offset < 4; ++offset) {
      int mode = ((iteration + 6) + offset) % 4;
      auto begin = std::chrono::steady_clock::now();
      float ms = run(mode);
      double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
      if (iteration >= 0) { gpu[mode].push_back(ms); wall[mode].push_back(elapsed); }
    }
  }
  for (int mode = 0; mode < 4; ++mode) {
    CUDA(cudaMemset(output, 0xff, output_elements * 2));
    CUDA(cudaDeviceSynchronize());
    run(mode);
    CUDA(cudaMemcpy(actual.data(), output, output_elements * 2, cudaMemcpyDeviceToHost));
    size_t exact = 0;
    for (size_t i = 0; i < output_elements; ++i) exact += actual[i] == reference[i];
    std::sort(gpu[mode].begin(), gpu[mode].end());
    std::sort(wall[mode].begin(), wall[mode].end());
    std::printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"mode\":\"%s\",\"gpuMedianMs\":%.6f,"
      "\"wallMedianMs\":%.6f,\"exact\":%zu,\"elements\":%zu,\"graphNodes\":%zu}\n",
      rows, columns, inner, mode == 0 ? "cublas" : mode == 1 ? "expert-graphs" : mode == 2 ? "stream-graphs" : "joined-graph",
      gpu[mode][12], wall[mode][12], exact, output_elements,
      mode == 0 ? 0 : mode == 1 ? expert_nodes : mode == 2 ? stream_nodes : joined_nodes);
    if (exact != output_elements) std::exit(2);
  }
  CUDA(cudaGraphExecDestroy(joined_graph));
  for (auto graph : expert_graph) CUDA(cudaGraphExecDestroy(graph));
  for (auto graph : stream_graph) CUDA(cudaGraphExecDestroy(graph));
  CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end)); CUDA(cudaStreamDestroy(primary));
  CUDA(cudaEventDestroy(ready));
  for (auto& w : worker) {
    BLAS(cublasDestroy(w.handle)); CUDA(cudaFree(w.workspace));
    CUDA(cudaEventDestroy(w.done)); CUDA(cudaStreamDestroy(w.stream));
  }
  CUDA(cudaFree(x)); CUDA(cudaFree(weight)); CUDA(cudaFree(output));
}

int main() {
  for (int m : {8, 16, 32}) {
    benchmark(m, 1408, 2816);
    benchmark(m, 2816, 704);
  }
}
