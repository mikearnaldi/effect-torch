// Standalone CUPTI injection profiler; never enables the serial KERNEL activity.
// Build against matching CUPTI headers/library. Set CUDA_INJECTION64_PATH and
// EFFECT_TORCH_CUPTI_OUTPUT (a new path). All recording is asynchronous; no CUDA
// synchronization is inserted. JSON batches are retained until the exit flush.
#include <cupti.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sstream>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr size_t buffer_bytes = 8 * 1024 * 1024;
constexpr size_t retained_limit = 1024ULL * 1024 * 1024;
struct State {
  std::mutex mutex;
  std::vector<std::string> batches;
  size_t retained = 0;
  std::atomic<uint64_t> dropped{0}, omitted{0}, errors{0};
  std::atomic<bool> flushed{false};
  int output = -1;
};
// Deliberately process-lifetime storage: CUDA may deliver callbacks at teardown.
State& state() { static auto* value = new State; return *value; }
std::string quoted(const char* s) {
  std::string out = "\"";
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s ? s : ""); *p; ++p) {
    if (*p == '"' || *p == '\\') { out += '\\'; out += char(*p); }
    else if (*p < 32) { char escaped[7]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", *p); out += escaped; }
    else out += char(*p);
  }
  return out + '"';
}
bool check(CUptiResult result, const char* operation) {
  if (result == CUPTI_SUCCESS) return true;
  const char* message = nullptr;
  cuptiGetResultString(result, &message);
  std::fprintf(stderr, "CUPTI timeline: %s: %s\n", operation, message ? message : "unknown error");
  ++state().errors;
  return false;
}
void CUPTIAPI requested(uint8_t** buffer, size_t* size, size_t* max_records) {
  *buffer = static_cast<uint8_t*>(std::malloc(buffer_bytes));
  *size = *buffer ? buffer_bytes : 0;
  *max_records = 0;
  if (!*buffer) ++state().errors;
}
void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t* buffer, size_t, size_t valid) {
  std::ostringstream out;
  CUpti_Activity* record = nullptr;
  CUptiResult result;
  uint64_t records = 0;
  while ((result = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
    ++records;
    switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const auto& k = *reinterpret_cast<const CUpti_ActivityKernel9*>(record);
        out << "{\"kind\":\"kernel\",\"start\":" << k.start << ",\"end\":" << k.end
            << ",\"device\":" << k.deviceId << ",\"context\":" << k.contextId
            << ",\"stream\":" << k.streamId << ",\"correlation\":" << k.correlationId
            << ",\"name\":" << quoted(k.name) << ",\"grid\":[" << k.gridX << ',' << k.gridY << ',' << k.gridZ
            << "],\"block\":[" << k.blockX << ',' << k.blockY << ',' << k.blockZ
            << "],\"registers\":" << k.registersPerThread << ",\"sharedBytes\":" << k.staticSharedMemory + k.dynamicSharedMemory
            << ",\"graph\":" << k.graphId << "}\n";
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const auto& m = *reinterpret_cast<const CUpti_ActivityMemcpy6*>(record);
        out << "{\"kind\":\"memcpy\",\"start\":" << m.start << ",\"end\":" << m.end
            << ",\"device\":" << m.deviceId << ",\"context\":" << m.contextId
            << ",\"stream\":" << m.streamId << ",\"correlation\":" << m.correlationId
            << ",\"runtimeCorrelation\":" << m.runtimeCorrelationId
            << ",\"copyKind\":" << unsigned(m.copyKind) << ",\"bytes\":" << m.bytes << "}\n";
        break;
      }
      case CUPTI_ACTIVITY_KIND_DRIVER:
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        const auto& a = *reinterpret_cast<const CUpti_ActivityAPI*>(record);
        const bool driver = record->kind == CUPTI_ACTIVITY_KIND_DRIVER;
        const char* name = nullptr;
        cuptiGetCallbackName(driver ? CUPTI_CB_DOMAIN_DRIVER_API : CUPTI_CB_DOMAIN_RUNTIME_API, a.cbid, &name);
        out << "{\"kind\":" << quoted(driver ? "driver" : "runtime") << ",\"start\":" << a.start << ",\"end\":" << a.end
            << ",\"correlation\":" << a.correlationId << ",\"thread\":" << a.threadId
            << ",\"cbid\":" << a.cbid << ",\"name\":" << quoted(name) << ",\"result\":" << a.returnValue << "}\n";
        break;
      }
      default: break;
    }
  }
  if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) check(result, "next record");
  size_t dropped = 0;
  if (check(cuptiActivityGetNumDroppedRecords(context, stream, &dropped), "dropped records")) state().dropped += dropped;
  auto batch = out.str();
  {
    std::lock_guard<std::mutex> lock(state().mutex);
    if (batch.size() <= retained_limit - state().retained) {
      state().retained += batch.size();
      state().batches.push_back(std::move(batch));
    } else state().omitted += records;
  }
  std::free(buffer);
}
void finish() {
  if (state().output < 0 || state().flushed.exchange(true)) return;
  check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "exit flush");
  std::lock_guard<std::mutex> lock(state().mutex);
  FILE* output = fdopen(state().output, "w");
  if (!output) { std::perror("CUPTI timeline fdopen"); return; }
  bool failed = false;
  for (const auto& batch : state().batches) failed |= std::fwrite(batch.data(), 1, batch.size(), output) != batch.size();
  std::fprintf(output, "{\"kind\":\"summary\",\"droppedRecords\":%llu,\"omittedRecords\":%llu,\"errors\":%llu,\"retainedBytes\":%zu}\n",
      static_cast<unsigned long long>(state().dropped.load()), static_cast<unsigned long long>(state().omitted.load()),
      static_cast<unsigned long long>(state().errors.load()), state().retained);
  failed |= std::fclose(output) != 0;
  if (failed) std::fprintf(stderr, "CUPTI timeline: output write failed\n");
}
}
// Multiprocessing workers may use os._exit and bypass atexit. A diagnostic-only
// multiprocessing finalizer can call this after its last measured CUDA work.
extern "C" __attribute__((visibility("default"))) void EffectTorchCuptiFlush() {
  finish();
}
extern "C" __attribute__((visibility("default"))) int InitializeInjection() {
  static std::once_flag once;
  static int initialized = 0;
  std::call_once(once, [] {
    const char* path = std::getenv("EFFECT_TORCH_CUPTI_OUTPUT");
    if (!path || !*path) { std::fprintf(stderr, "CUPTI timeline: EFFECT_TORCH_CUPTI_OUTPUT required\n"); return; }
    std::string output_path(path);
    const std::string pid = std::to_string(getpid());
    for (size_t at = 0; (at = output_path.find("%p", at)) != std::string::npos; at += pid.size()) {
      output_path.replace(at, 2, pid);
    }
    state().output = open(output_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (state().output < 0) { std::perror("CUPTI timeline output"); return; }
    uint32_t version = 0;
    uint64_t timestamp = 0;
    timespec wall{};
    check(cuptiGetVersion(&version), "version");
    check(cuptiGetTimestamp(&timestamp), "timestamp");
    clock_gettime(CLOCK_REALTIME, &wall);
    std::ostringstream metadata;
    metadata << "{\"kind\":\"metadata\",\"cuptiVersion\":" << version << ",\"timestamp\":" << timestamp
             << ",\"wallNs\":" << uint64_t(wall.tv_sec) * 1000000000ULL + wall.tv_nsec << ",\"pid\":" << getpid() << "}\n";
    state().batches.push_back(metadata.str());
    std::atexit(finish);
    if (!check(cuptiActivityRegisterCallbacks(requested, completed), "register callbacks")) return;
    for (auto kind : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_RUNTIME}) {
      if (!check(cuptiActivityEnable(kind), "enable activity")) return;
    }
    initialized = 1;
  });
  return initialized;
}
