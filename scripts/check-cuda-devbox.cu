#include <cublasLt.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

__global__ void increment(int* value) {
  *value += 1;
}

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  errno = 0;
  char* end = nullptr;
  const long parsed_device = argc > 1 ? std::strtol(argv[1], &end, 10) : 0;
  if (
    errno != 0 || parsed_device < 0 ||
    parsed_device > std::numeric_limits<int>::max() ||
    (argc > 1 && (end == argv[1] || *end != '\0'))
  ) {
    std::fprintf(stderr, "invalid CUDA device ordinal: %s\n", argc > 1 ? argv[1] : "");
    return 1;
  }
  const int device = static_cast<int>(parsed_device);

  int device_count = 0;
  if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
    return 1;
  }
  if (device >= device_count) {
    std::fprintf(stderr, "CUDA device %d is not in the visible range 0..%d\n", device, device_count - 1);
    return 1;
  }
  if (!check_cuda(cudaSetDevice(device), "cudaSetDevice")) {
    return 1;
  }

  cudaDeviceProp properties{};
  if (!check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties")) {
    return 1;
  }

  int* device_value = nullptr;
  int value = 0;
  if (!check_cuda(cudaMalloc(&device_value, sizeof(int)), "cudaMalloc")) {
    return 1;
  }
  if (!check_cuda(cudaMemset(device_value, 0, sizeof(int)), "cudaMemset")) {
    cudaFree(device_value);
    return 1;
  }
  increment<<<1, 1>>>(device_value);
  if (
    !check_cuda(cudaGetLastError(), "increment launch") ||
    !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
    !check_cuda(cudaMemcpy(&value, device_value, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy")
  ) {
    cudaFree(device_value);
    return 1;
  }
  cudaFree(device_value);
  if (value != 1) {
    std::fprintf(stderr, "CUDA kernel returned %d instead of 1\n", value);
    return 1;
  }

  int nvrtc_major = 0;
  int nvrtc_minor = 0;
  const nvrtcResult nvrtc_status = nvrtcVersion(&nvrtc_major, &nvrtc_minor);
  if (nvrtc_status != NVRTC_SUCCESS) {
    std::fprintf(stderr, "nvrtcVersion: %s\n", nvrtcGetErrorString(nvrtc_status));
    return 1;
  }

  cublasLtHandle_t cublas = nullptr;
  const cublasStatus_t cublas_status = cublasLtCreate(&cublas);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasLtCreate failed with status %d\n", static_cast<int>(cublas_status));
    return 1;
  }
  cublasLtDestroy(cublas);

  std::printf(
    "CUDA check passed: device=%d name=\"%s\" cc=%d.%d memory=%zu MiB nvrtc=%d.%d\n",
    device,
    properties.name,
    properties.major,
    properties.minor,
    properties.totalGlobalMem / (1024 * 1024),
    nvrtc_major,
    nvrtc_minor
  );
  return 0;
}
