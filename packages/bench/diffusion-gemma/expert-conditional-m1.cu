// Standalone CUDA 12.9 conditional-node feasibility and complete M1-route timing.
// No runtime integration. Arguments: projection (0/1), experts (2..128), pattern (0..3).
// nvcc -O3 -std=c++17 -arch=sm_120 expert-conditional-m1.cu -lcublas -lcuda -o expert-conditional-m1
#define main unused_batching_main
#include "expert-gemm-batching.cu"
#undef main

template<class T> T* alloc_m1(size_t count) {
  T* p; CUDA(cudaMalloc(&p, count * sizeof(T))); return p;
}

__global__ void prepare_m1(const int* counts, const int* rows,
                          const cudaGraphConditionalHandle* handles,
                          const unsigned short* input, unsigned short* staged,
                          int experts, int k) {
  int e = blockIdx.x;
  if (threadIdx.x == 0) cudaGraphSetConditional(handles[e], counts[e] == 1);
  if (counts[e] == 1)
    for (int i = threadIdx.x; i < k; i += blockDim.x)
      staged[e * k + i] = input[rows[e] * k + i];
}

__global__ void finish_m1(const int* counts, const int* rows,
                         const unsigned short* staged, unsigned short* output,
                         int n) {
  int e = blockIdx.x;
  if (counts[e] == 1)
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      output[rows[e] * n + i] = staged[e * n + i];
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  int projection = std::atoi(argv[1]), experts = std::atoi(argv[2]);
  int pattern = std::atoi(argv[3]);
  bool fanout = std::getenv("ET_CONDITIONAL_FANOUT") != nullptr;
  bool zero_only = std::getenv("ET_CONDITIONAL_ZERO_ONLY") != nullptr;
  if (projection < 0 || projection > 1 || experts < 2 || experts > 128 || pattern < 0 || pattern > 3) return 2;
  int n = projection ? 2816 : 1408, k = projection ? 704 : 2816;
  size_t wc = size_t(n) * k, output_count = size_t(experts) * n;
  auto input = alloc_m1<unsigned short>(size_t(experts) * k);
  auto weight = alloc_m1<unsigned short>(size_t(experts) * wc);
  auto staged_input = alloc_m1<unsigned short>(size_t(experts) * k);
  auto staged_output = alloc_m1<unsigned short>(output_count);
  auto output = alloc_m1<unsigned short>(output_count);
  auto counts = alloc_m1<int>(experts), rows = alloc_m1<int>(experts);
  auto handles_device = alloc_m1<cudaGraphConditionalHandle>(experts);
  auto hx = dense(size_t(experts) * k, 117);
  if (pattern == 1) std::fill(hx.begin(), hx.end(), 0x8000);
  if (pattern >= 2) std::fill(hx.begin(), hx.end(), 0x3f80);
  CUDA(cudaMemcpy(input, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  for (int e = 0; e < experts; ++e) {
    auto hw = dense(wc, 129 + e * 43);
    if (pattern >= 2) for (size_t j = 0; j < wc; ++j) {
      const unsigned short inf[] = {0x7f80, 0xff80, 0x3f80, 0xbf80};
      const unsigned short nan[] = {0x7fc1, 0xffc1, 0x7f81, 0xff81};
      hw[j] = (pattern == 2 ? inf : nan)[(j / k) % 4];
    }
    CUDA(cudaMemcpy(weight + e * wc, hw.data(), wc * 2, cudaMemcpyHostToDevice));
  }
  cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t blas; BLAS(cublasCreate(&blas)); BLAS(cublasSetStream(blas, stream));
  void* workspace; CUDA(cudaMalloc(&workspace, workspace_bytes));
  BLAS(cublasSetWorkspace(blas, workspace, workspace_bytes));
  BLAS(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
  std::vector<cublasHandle_t> body_blas(experts, blas);
  std::vector<void*> body_workspace;
  if (fanout) for (int e = 0; e < experts; ++e) {
    BLAS(cublasCreate(&body_blas[e]));
    BLAS(cublasSetStream(body_blas[e], stream));
    void* p; CUDA(cudaMalloc(&p, 1 << 20)); body_workspace.push_back(p);
    BLAS(cublasSetWorkspace(body_blas[e], p, 1 << 20));
    BLAS(cublasSetMathMode(body_blas[e], CUBLAS_DEFAULT_MATH));
  }
  float one = 1, zero = 0;
  auto gemm = [&](int e, const unsigned short* x, unsigned short* y, bool reference = false) {
    BLAS(cublasGemmStridedBatchedEx(reference ? blas : body_blas[e], CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k,
      &one, weight + e * wc, CUDA_R_16BF, k, 0, x, CUDA_R_16BF, k, k,
      &zero, y, CUDA_R_16BF, n, n, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  // Warm actual fixed-pointer GEMMs before capture so cuBLAS initializes outside bodies.
  CUDA(cudaMemcpyAsync(staged_input, input, hx.size() * 2, cudaMemcpyDeviceToDevice, stream));
  for (int e = 0; e < experts; ++e) gemm(e, staged_input + e * k, staged_output + e * n);
  CUDA(cudaStreamSynchronize(stream));
  cudaGraph_t graph; CUDA(cudaGraphCreate(&graph, 0));
  std::vector<cudaGraphConditionalHandle> handles(experts);
  for (auto& h : handles) CUDA(cudaGraphConditionalHandleCreate(&h, graph, 0, cudaGraphCondAssignDefault));
  CUDA(cudaMemcpy(handles_device, handles.data(), experts * sizeof(handles[0]), cudaMemcpyHostToDevice));
  // First capture initializes every condition and gathers only active M1 rows.
  CUDA(cudaStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
  prepare_m1<<<experts, 256, 0, stream>>>(counts, rows, handles_device, input, staged_input, experts, k);
  cudaGraph_t captured; CUDA(cudaStreamEndCapture(stream, &captured));
  size_t roots_count = 1; cudaGraphNode_t previous;
  CUDA(cudaGraphGetNodes(graph, &previous, &roots_count));
  cudaGraphNode_t prepare = previous;
  std::vector<cudaGraphNode_t> conditional_nodes;
  size_t body_nodes = 0;
  for (int e = 0; e < experts; ++e) {
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handles[e];
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t node;
    auto dependency = fanout ? prepare : previous;
    CUDA(cudaGraphAddNode(&node, graph, &dependency, 1, &params));
    conditional_nodes.push_back(node);
    auto body = params.conditional.phGraph_out[0];
    CUDA(cudaStreamBeginCaptureToGraph(stream, body, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    gemm(e, staged_input + e * k, staged_output + e * n);
    CUDA(cudaStreamEndCapture(stream, &captured));
    size_t count = 0; CUDA(cudaGraphGetNodes(body, nullptr, &count)); body_nodes += count;
    std::vector<cudaGraphNode_t> nodes(count); CUDA(cudaGraphGetNodes(body, nodes.data(), &count));
    for (auto child : nodes) {
      cudaGraphNodeType type; CUDA(cudaGraphNodeGetType(child, &type));
      if (type == cudaGraphNodeTypeKernel && e == 0) {
        CUDA_KERNEL_NODE_PARAMS driver_params{};
        if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(child), &driver_params) != CUDA_SUCCESS) return 5;
        const char* name = nullptr;
        if (cuFuncGetName(&name, driver_params.func) != CUDA_SUCCESS) return 5;
        std::fprintf(stderr, "captured M1 kernel: %s\n", name);
      }
      if (type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeMemcpy && type != cudaGraphNodeTypeMemset && type != cudaGraphNodeTypeEmpty) {
        std::fprintf(stderr, "unexpected captured cuBLAS node type %d\n", int(type)); return 3;
      }
    }
    // Serial bodies safely reuse the same cuBLAS workspace and establish a fair
    // first proof against serial eager cuBLAS. Parallel worker banks are separate work.
    previous = node;
  }
  CUDA(cudaStreamBeginCaptureToGraph(stream, graph, fanout ? conditional_nodes.data() : &previous,
    nullptr, fanout ? conditional_nodes.size() : 1, cudaStreamCaptureModeGlobal));
  finish_m1<<<experts, 256, 0, stream>>>(counts, rows, staged_output, output, n);
  CUDA(cudaStreamEndCapture(stream, &captured));
  cudaGraphExec_t exec; CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cudaEvent_t start, end; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
  std::vector<int> hc(experts), hr(experts);
  std::vector<unsigned short> expected(output_count), actual(output_count);
  for (int route = 0; route < (zero_only ? 1 : 6); ++route) {
    int active = 0;
    for (int e = 0; e < experts; ++e) {
      hc[e] = route == 0 ? 0 : route == 1 ? 1 : route == 2 ? 2 : ((e + route) % 3);
      hr[e] = (e + route) % experts;
      active += hc[e] == 1;
    }
    CUDA(cudaMemcpyAsync(counts, hc.data(), experts * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA(cudaMemcpyAsync(rows, hr.data(), experts * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA(cudaMemsetAsync(output, 0xa5, output_count * 2, stream));
    for (int e = 0; e < experts; ++e) if (hc[e] == 1) gemm(e, input + hr[e] * k, output + hr[e] * n, true);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(expected.data(), output, output_count * 2, cudaMemcpyDeviceToHost));
    CUDA(cudaMemsetAsync(output, 0xa5, output_count * 2, stream));
    CUDA(cudaGraphLaunch(exec, stream)); CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(actual.data(), output, output_count * 2, cudaMemcpyDeviceToHost));
    size_t exact = 0; for (size_t i = 0; i < output_count; ++i) exact += actual[i] == expected[i];
    if (exact != output_count) { std::fprintf(stderr, "route %d exact %zu/%zu\n", route, exact, output_count); return 4; }
    for (int mode = 0; mode < 2; ++mode) {
      std::vector<float> times; std::vector<double> walls;
      for (int repeat = -3; repeat < 21; ++repeat) {
        CUDA(cudaEventRecord(start, stream)); auto begin = std::chrono::steady_clock::now();
        if (mode) CUDA(cudaGraphLaunch(exec, stream));
        else for (int e = 0; e < experts; ++e) if (hc[e] == 1) gemm(e, input + hr[e] * k, output + hr[e] * n, true);
        CUDA(cudaEventRecord(end, stream)); CUDA(cudaEventSynchronize(end));
        float ms; CUDA(cudaEventElapsedTime(&ms, start, end));
        double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (repeat >= 0) { times.push_back(ms); walls.push_back(wall); }
      }
      std::sort(times.begin(), times.end()); std::sort(walls.begin(), walls.end());
      std::printf("{\"projection\":%d,\"experts\":%d,\"pattern\":%d,\"route\":%d,\"active\":%d,\"mode\":\"%s\",\"exact\":%zu,\"elements\":%zu,\"bodyNodes\":%zu,\"gpuMedianMs\":%.6f,\"wallMedianMs\":%.6f}\n", projection, experts, pattern, route, active, mode ? "conditional" : "eager", exact, output_count, body_nodes, times[10], walls[10]);
    }
  }
  CUDA(cudaGraphExecDestroy(exec)); CUDA(cudaGraphDestroy(graph));
  CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end)); BLAS(cublasDestroy(blas));
  if (fanout) for (auto h : body_blas) BLAS(cublasDestroy(h));
  for (auto p : body_workspace) CUDA(cudaFree(p));
  CUDA(cudaFree(workspace)); CUDA(cudaStreamDestroy(stream));
  for (void* p : {static_cast<void*>(input), static_cast<void*>(weight), static_cast<void*>(staged_input), static_cast<void*>(staged_output), static_cast<void*>(output), static_cast<void*>(counts), static_cast<void*>(rows), static_cast<void*>(handles_device)}) CUDA(cudaFree(p));
}
