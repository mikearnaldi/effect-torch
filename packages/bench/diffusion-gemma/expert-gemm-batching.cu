// Exact-size batch submission experiment against fresh eager expert fixtures.
// Repeated weights are separate allocations in the bank so timings include
// reading every expert's weights. This is a component upper bound, not model
// performance. Compile with nvcc -O3 -lcublas -lcuda; pass fixture directory argv[1].
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA(call) do { auto status = (call); if (status != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); std::exit(1); } } while (0)
#define BLAS(call) do { auto status = (call); if (status != CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr, "%s: status %d\n", #call, int(status)); std::exit(1); } } while (0)

constexpr int workers = 32;
constexpr size_t workspace_bytes = 32 << 20;

std::vector<unsigned short> load(const std::string& path, size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() != std::streamoff(count * 2)) {
    std::fprintf(stderr, "unexpected size or absent fixture: %s\n", path.c_str());
    std::exit(1);
  }
  input.seekg(0);
  std::vector<unsigned short> result(count);
  input.read(reinterpret_cast<char*>(result.data()), count * 2);
  return result;
}

struct Worker { cudaStream_t stream; cublasHandle_t handle; cudaEvent_t done; void* workspace; };

std::vector<unsigned short> dense(size_t count, unsigned seed) {
  std::vector<unsigned short> result(count);
  for (size_t index = 0; index < count; ++index) {
    unsigned h = unsigned(index) + seed;
    h = (h ^ (h >> 16)) * 0x7feb352du;
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    h ^= h >> 16;
    result[index] = ((h >> 16) & 0x8000) | ((((h >> 24) % 9) + 120) << 7) | ((h >> 8) & 127);
  }
  return result;
}

void benchmark(const std::string& directory, const std::string& name, int rows, int columns,
    int inner, int batch, Worker* worker, cudaStream_t primary, int seed = -1) {
  const size_t xc = size_t(rows) * inner, wc = size_t(columns) * inner, oc = size_t(rows) * columns;
  auto input = seed < 0 ? load(directory + "/" + name + ".input.bf16", xc) : dense(xc, 17 + seed * 113);
  auto weight = seed < 0 ? load(directory + "/" + name + ".weight.bf16", wc) : dense(wc, 29 + seed * 127);
  auto reference = seed < 0 ? load(directory + "/" + name + ".official.bf16", oc) : std::vector<unsigned short>(oc);
  unsigned short *x, *w, *out;
  CUDA(cudaMalloc(&x, batch * xc * 2)); CUDA(cudaMalloc(&w, batch * wc * 2));
  CUDA(cudaMalloc(&out, batch * oc * 2));
  CUDA(cudaMemcpy(x, input.data(), xc * 2, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(w, weight.data(), wc * 2, cudaMemcpyHostToDevice));
  std::vector<const void*> xp(batch), wp(batch);
  std::vector<void*> op(batch);
  for (int i = 0; i < batch; ++i) {
    xp[i] = x + i * xc; wp[i] = w + i * wc; op[i] = out + i * oc;
    if (i) {
      CUDA(cudaMemcpy(x + i * xc, x, xc * 2, cudaMemcpyDeviceToDevice));
      CUDA(cudaMemcpy(w + i * wc, w, wc * 2, cudaMemcpyDeviceToDevice));
    }
  }
  const void **dx, **dw; void** dout;
  CUDA(cudaMalloc(&dx, batch * sizeof(void*))); CUDA(cudaMalloc(&dw, batch * sizeof(void*)));
  CUDA(cudaMalloc(&dout, batch * sizeof(void*)));
  CUDA(cudaMemcpy(dx, xp.data(), batch * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dw, wp.data(), batch * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dout, op.data(), batch * sizeof(void*), cudaMemcpyHostToDevice));
  cudaEvent_t start, end;
  CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
  float alpha = 1, beta = 0;
  auto submit = [&](int mode) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    if (mode == 0) {
      for (int wi = 0; wi < workers; ++wi) {
        for (int i = wi; i < batch; i += workers) {
          status = cublasGemmStridedBatchedEx(worker[wi].handle, CUBLAS_OP_T, CUBLAS_OP_N,
            columns, rows, inner, &alpha, wp[i], CUDA_R_16BF, inner, 0,
            xp[i], CUDA_R_16BF, inner, xc, &beta, op[i], CUDA_R_16BF,
            columns, oc, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
          if (status != CUBLAS_STATUS_SUCCESS) return status;
        }
      }
    } else if (mode == 1) {
      status = cublasGemmBatchedEx(worker[0].handle, CUBLAS_OP_T, CUBLAS_OP_N,
        columns, rows, inner, &alpha, dw, CUDA_R_16BF, inner, dx, CUDA_R_16BF,
        inner, &beta, dout, CUDA_R_16BF, columns, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    } else if (mode == 2) {
      status = cublasGemmStridedBatchedEx(worker[0].handle, CUBLAS_OP_T, CUBLAS_OP_N,
        columns, rows, inner, &alpha, w, CUDA_R_16BF, inner, wc, x, CUDA_R_16BF,
        inner, xc, &beta, out, CUDA_R_16BF, columns, oc, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    } else {
      cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;
      status = cublasGemmGroupedBatchedEx(worker[0].handle, &transa, &transb,
        &columns, &rows, &inner, &alpha, dw, CUDA_R_16BF, &inner, dx, CUDA_R_16BF,
        &inner, &beta, dout, CUDA_R_16BF, &columns, 1, &batch, CUBLAS_COMPUTE_32F);
    }
    return status;
  };
  auto run = [&](int mode, float* elapsed) {
    CUDA(cudaEventRecord(start, primary));
    for (int wi = 0; wi < workers; ++wi) CUDA(cudaStreamWaitEvent(worker[wi].stream, start, 0));
    auto status = submit(mode);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    for (int wi = 0; wi < workers; ++wi) {
      CUDA(cudaEventRecord(worker[wi].done, worker[wi].stream));
      CUDA(cudaStreamWaitEvent(primary, worker[wi].done, 0));
    }
    CUDA(cudaEventRecord(end, primary)); CUDA(cudaEventSynchronize(end));
    CUDA(cudaEventElapsedTime(elapsed, start, end));
    return status;
  };
  const char* modes[] = {"ordinary-32streams", "pointer-batched", "strided-batched", "grouped-batched"};
  if (seed >= 0) {
    float elapsed; BLAS(run(0, &elapsed));
    CUDA(cudaMemcpy(reference.data(), out, oc * 2, cudaMemcpyDeviceToHost));
  }
  std::vector<double> gpu[4], wall[4];
  bool supported[4] = {true, true, true, true};
  for (int iteration = -3; iteration < 11; ++iteration) {
    for (int offset = 0; offset < 4; ++offset) {
      int mode = (iteration + 4 + offset) % 4;
      if (!supported[mode]) continue;
      auto begin = std::chrono::steady_clock::now();
      float ms;
      auto status = run(mode, &ms);
      if (status != CUBLAS_STATUS_SUCCESS) {
        supported[mode] = false;
        std::printf("{\"case\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,\"mode\":\"%s\",\"status\":%d}\n",
          name.c_str(), rows, columns, inner, batch, modes[mode], int(status));
        CUDA(cudaDeviceSynchronize()); continue;
      }
      double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
      if (iteration >= 0) { gpu[mode].push_back(ms); wall[mode].push_back(elapsed); }
    }
  }
  std::vector<unsigned short> actual(batch * oc);
  for (int mode = 0; mode < 4; ++mode) {
    if (!supported[mode]) continue;
    CUDA(cudaMemset(out, 0xff, batch * oc * 2)); CUDA(cudaDeviceSynchronize());
    float ms; BLAS(run(mode, &ms));
    CUDA(cudaMemcpy(actual.data(), out, batch * oc * 2, cudaMemcpyDeviceToHost));
    size_t exact = 0;
    for (size_t i = 0; i < actual.size(); ++i) exact += actual[i] == reference[i % oc];
    std::sort(gpu[mode].begin(), gpu[mode].end()); std::sort(wall[mode].begin(), wall[mode].end());
    std::printf("{\"case\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,\"mode\":\"%s\",\"gpuMedianMs\":%.6f,\"wallMedianMs\":%.6f,\"exact\":%zu,\"elements\":%zu}\n",
      name.c_str(), rows, columns, inner, batch, modes[mode], gpu[mode][5], wall[mode][5], exact, actual.size());
    std::fflush(stdout);
    if (mode == 0 && exact != actual.size()) std::exit(2);
  }
  // cuBLAS kernels are driver functions. Graph handles interoperate between
  // runtime and driver APIs; querying driver params provides names and ABI
  // geometry without parsing opaque cuBLAS launch parameters.
  if (batch == 8 && seed < 0) {
    for (int mode : {0, 1, 3}) {
      if (!supported[mode]) continue;
      CUDA(cudaStreamBeginCapture(worker[0].stream, cudaStreamCaptureModeThreadLocal));
      if (mode == 0) {
        BLAS(cublasGemmStridedBatchedEx(worker[0].handle, CUBLAS_OP_T, CUBLAS_OP_N,
          columns, rows, inner, &alpha, w, CUDA_R_16BF, inner, 0, x, CUDA_R_16BF,
          inner, xc, &beta, out, CUDA_R_16BF, columns, oc, 1,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else BLAS(submit(mode));
      cudaGraph_t graph; CUDA(cudaStreamEndCapture(worker[0].stream, &graph));
      size_t count = 0; CUDA(cudaGraphGetNodes(graph, nullptr, &count));
      std::vector<cudaGraphNode_t> nodes(count); CUDA(cudaGraphGetNodes(graph, nodes.data(), &count));
      for (auto node : nodes) {
        cudaGraphNodeType type; CUDA(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel) continue;
        CUDA_KERNEL_NODE_PARAMS params;
        auto driver_status = cuGraphKernelNodeGetParams(node, &params);
        if (driver_status != CUDA_SUCCESS) {
          std::fprintf(stderr, "cuGraphKernelNodeGetParams: %d\n", int(driver_status)); std::exit(1);
        }
        const char* kernel_name = "unknown";
        cuFuncGetName(&kernel_name, params.func);
        std::printf("{\"case\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,\"mode\":\"%s\",\"kernelName\":\"%s\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"sharedBytes\":%u,\"params\":[",
          name.c_str(), rows, columns, inner, mode == 0 ? 1 : batch, modes[mode], kernel_name,
          params.gridDimX, params.gridDimY, params.gridDimZ, params.blockDimX, params.blockDimY, params.blockDimZ, params.sharedMemBytes);
        for (size_t index = 0; index < 64; ++index) {
          size_t offset, length;
          if (cuFuncGetParamInfo(params.func, index, &offset, &length) != CUDA_SUCCESS) break;
          std::printf("%s[%zu,%zu]", index ? "," : "", offset, length);
        }
        std::printf("]}\n");
      }
      CUDA(cudaGraphDestroy(graph));
    }
  }
  CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
  CUDA(cudaFree(dx)); CUDA(cudaFree(dw)); CUDA(cudaFree(dout));
  CUDA(cudaFree(x)); CUDA(cudaFree(w)); CUDA(cudaFree(out));
}

void heterogeneous(int projection, int seed, Worker* worker, cudaStream_t primary) {
  constexpr int count = 128;
  const int columns = projection ? 2816 : 1408, inner = projection ? 704 : 2816;
  const size_t wc = size_t(columns) * inner;
  int m[count], n[count], k[count], sizes[count];
  cublasOperation_t ta[count], tb[count];
  float alpha[count], beta[count];
  std::vector<size_t> x_offset(count), out_offset(count);
  size_t xc = 0, oc = 0;
  for (int i = 0; i < count; ++i) {
    m[i] = columns; n[i] = 2 + i % (projection ? 127 : 15); k[i] = inner;
    sizes[i] = 1; ta[i] = CUBLAS_OP_T; tb[i] = CUBLAS_OP_N; alpha[i] = 1; beta[i] = 0;
    x_offset[i] = xc; out_offset[i] = oc; xc += size_t(n[i]) * inner; oc += size_t(n[i]) * columns;
  }
  unsigned short *x, *w, *out;
  CUDA(cudaMalloc(&x, xc * 2)); CUDA(cudaMalloc(&w, count * wc * 2)); CUDA(cudaMalloc(&out, oc * 2));
  std::vector<const void*> xp(count), wp(count); std::vector<void*> op(count);
  for (int i = 0; i < count; ++i) {
    auto input = dense(size_t(n[i]) * inner, 17 + seed * 113 + i * 37);
    auto weight = dense(wc, 29 + seed * 127 + i * 43);
    xp[i] = x + x_offset[i]; wp[i] = w + i * wc; op[i] = out + out_offset[i];
    CUDA(cudaMemcpy(x + x_offset[i], input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(w + i * wc, weight.data(), wc * 2, cudaMemcpyHostToDevice));
  }
  const void **dx, **dw; void** dout;
  CUDA(cudaMalloc(&dx, count * sizeof(void*))); CUDA(cudaMalloc(&dw, count * sizeof(void*)));
  CUDA(cudaMalloc(&dout, count * sizeof(void*)));
  CUDA(cudaMemcpy(dx, xp.data(), count * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dw, wp.data(), count * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dout, op.data(), count * sizeof(void*), cudaMemcpyHostToDevice));
  cudaEvent_t start, end; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
  auto run = [&](bool grouped) {
    CUDA(cudaEventRecord(start, primary));
    for (int wi = 0; wi < workers; ++wi) CUDA(cudaStreamWaitEvent(worker[wi].stream, start, 0));
    if (grouped) {
      BLAS(cublasGemmGroupedBatchedEx(worker[0].handle, ta, tb, m, n, k, alpha,
        dw, CUDA_R_16BF, k, dx, CUDA_R_16BF, k, beta, dout, CUDA_R_16BF, m, count, sizes, CUBLAS_COMPUTE_32F));
    } else {
      for (int wi = 0; wi < workers; ++wi) {
        for (int i = wi; i < count; i += workers) {
          BLAS(cublasGemmStridedBatchedEx(worker[wi].handle, CUBLAS_OP_T, CUBLAS_OP_N,
            columns, n[i], inner, &alpha[i], wp[i], CUDA_R_16BF, inner, 0,
            xp[i], CUDA_R_16BF, inner, size_t(n[i]) * inner, &beta[i], op[i], CUDA_R_16BF,
            columns, size_t(n[i]) * columns, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
      }
    }
    for (int wi = 0; wi < workers; ++wi) {
      CUDA(cudaEventRecord(worker[wi].done, worker[wi].stream)); CUDA(cudaStreamWaitEvent(primary, worker[wi].done, 0));
    }
    CUDA(cudaEventRecord(end, primary)); CUDA(cudaEventSynchronize(end));
    float elapsed; CUDA(cudaEventElapsedTime(&elapsed, start, end)); return elapsed;
  };
  run(false);
  std::vector<unsigned short> reference(oc), actual(oc);
  CUDA(cudaMemcpy(reference.data(), out, oc * 2, cudaMemcpyDeviceToHost));
  std::vector<double> gpu[2], wall[2];
  for (int iteration = -3; iteration < 11; ++iteration) {
    for (int offset = 0; offset < 2; ++offset) {
      int mode = (iteration + 4 + offset) % 2;
      auto begin = std::chrono::steady_clock::now(); float elapsed = run(mode);
      auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
      if (iteration >= 0) { gpu[mode].push_back(elapsed); wall[mode].push_back(ms); }
    }
  }
  for (int mode = 0; mode < 2; ++mode) {
    CUDA(cudaMemset(out, 0xff, oc * 2)); CUDA(cudaDeviceSynchronize()); run(mode);
    CUDA(cudaMemcpy(actual.data(), out, oc * 2, cudaMemcpyDeviceToHost));
    size_t exact = 0; for (size_t i = 0; i < oc; ++i) exact += actual[i] == reference[i];
    std::sort(gpu[mode].begin(), gpu[mode].end()); std::sort(wall[mode].begin(), wall[mode].end());
    std::printf("{\"case\":\"heterogeneous-%d-%d\",\"minRows\":2,\"maxRows\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,\"mode\":\"%s\",\"gpuMedianMs\":%.6f,\"wallMedianMs\":%.6f,\"exact\":%zu,\"elements\":%zu}\n",
      seed, projection, projection ? 128 : 16, columns, inner, count, mode ? "grouped-batched" : "ordinary-32streams", gpu[mode][5], wall[mode][5], exact, oc);
    std::fflush(stdout);
  }
  CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end)); CUDA(cudaFree(dx)); CUDA(cudaFree(dw));
  CUDA(cudaFree(dout)); CUDA(cudaFree(x)); CUDA(cudaFree(w)); CUDA(cudaFree(out));
}

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: %s expert-fixture-directory|--stress\n", argv[0]); return 1; }
  Worker worker[workers]; cudaStream_t primary;
  CUDA(cudaStreamCreateWithFlags(&primary, cudaStreamNonBlocking));
  for (auto& item : worker) {
    CUDA(cudaStreamCreateWithFlags(&item.stream, cudaStreamNonBlocking));
    CUDA(cudaEventCreateWithFlags(&item.done, cudaEventDisableTiming));
    CUDA(cudaMalloc(&item.workspace, workspace_bytes));
    BLAS(cublasCreate(&item.handle)); BLAS(cublasSetStream(item.handle, item.stream));
    BLAS(cublasSetMathMode(item.handle, CUBLAS_DEFAULT_MATH));
    BLAS(cublasSetWorkspace(item.handle, item.workspace, workspace_bytes));
  }
  const int encoder_rows[] = {262, 262, 6, 6, 1, 1, 1, 1, 4, 4};
  const int decoder_rows[] = {149, 149, 17, 17, 8, 8, 3, 3, 6, 6};
  if (std::string(argv[1]) == "--heterogeneous") {
    for (int seed : {0, 1, 2}) for (int projection : {0, 1}) heterogeneous(projection, seed, worker, primary);
  } else if (std::string(argv[1]) == "--stress") {
    for (int seed : {0, 1, 2}) {
      for (int m : {1, 2, 3, 4, 6, 8, 15, 16, 17, 31, 32, 33, 64, 128, 129, 149, 262}) {
        for (int projection : {0, 1}) {
          benchmark("", "stress-" + std::to_string(seed) + "-" + std::to_string(m) + "-" + std::to_string(projection),
            m, projection ? 2816 : 1408, projection ? 704 : 2816, 8, worker, primary, seed);
        }
      }
    }
  } else for (const auto* phase : {"encoder", "decoder"}) {
    for (int i = 0; i < 10; ++i) {
      int m = phase[0] == 'e' ? encoder_rows[i] : decoder_rows[i];
      int n = i % 2 ? 2816 : 1408, k = i % 2 ? 704 : 2816;
      for (int batch : {2, 8, 32, 128}) {
        benchmark(argv[1], std::string(phase) + "-" + std::to_string(i), m, n, k, batch, worker, primary);
      }
    }
  }
  for (auto& item : worker) {
    BLAS(cublasDestroy(item.handle)); CUDA(cudaFree(item.workspace));
    CUDA(cudaEventDestroy(item.done)); CUDA(cudaStreamDestroy(item.stream));
  }
  CUDA(cudaStreamDestroy(primary));
  return 0;
}
