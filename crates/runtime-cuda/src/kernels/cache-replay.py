"""Replay a pinned layer_probe.py capture without loading checkpoint weights.

python3 crates/runtime-cuda/src/kernels/check-nvrtc.py \
    --cache-replay /path/to/capture \
    --cache-replay-output /path/to/new-output-directory

Uses the runner's CUDA ABI and allocation helpers. Safetensors payloads contain
contiguous logical values; original strides describe the tensors before capture
and must not be reapplied to the serialized payload. Dtypes remain unchanged.
Production cache.cu is compiled verbatim. Diagnostic entry points only expose
its QK calculation or reset softmax/PV to exact official intermediate inputs.
"""
import argparse
import hashlib
import json


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache-replay", required=True, type=Path)
parser.add_argument("--cache-replay-output", required=True, type=Path)
parser.add_argument("--cache-replay-effect", type=Path)
parser.add_argument("--attention-stages", action="store_true")
parser.add_argument("--attention-torch", action="store_true")
parser.add_argument("--attention-padding", action="store_true")
options = parser.parse_args()
directory, output_directory = options.cache_replay, options.cache_replay_output
output_directory.mkdir(parents=True, exist_ok=False)
for source_name in ["cache.cu", "typed.cuh", "cache-replay.py", "check-nvrtc.py"]:
    (output_directory / source_name).write_bytes((ROOT / source_name).read_bytes())
manifest = json.loads((directory / "manifest.json").read_text())
calls = json.loads((directory / "calls.json").read_text())
assert manifest["status"] == "passed" and manifest["archived_answer_exact"] and manifest["prefix_unchanged"]
archive = directory / "layer0.safetensors"
with archive.open("rb") as source:
    digest = hashlib.file_digest(source, "sha256").hexdigest()
assert digest == manifest["tensor_file_sha256"], "capture archive hash differs"
with archive.open("rb") as source:
    header_length = struct.unpack("<Q", source.read(8))[0]
    assert 0 < header_length <= 16 * 1024 * 1024
    tensor_header = json.loads(source.read(header_length))
payload_start = 8 + header_length
types = {"F64": (0, 8, "float64"), "F32": (1, 4, "float32"),
         "F16": (2, 2, "float16"), "BF16": (3, 2, "bfloat16")}


def tensor(name):
    metadata = tensor_header[name]
    dtype, width, torch_dtype = types[metadata["dtype"]]
    shape = metadata["shape"]
    original = manifest["original_metadata"][name]
    assert original["shape"] == shape and original["dtype"] == "torch." + torch_dtype
    assert len(original["stride"]) == len(shape) and all(n >= 0 for n in original["stride"])
    start, end = metadata["data_offsets"]
    assert 0 <= start <= end and end - start == math.prod(shape) * width
    with archive.open("rb") as source:
        source.seek(payload_start + start)
        data = source.read(end - start)
    assert len(data) == end - start
    return dict(name=name, shape=shape, dtype=dtype, width=width, data=data, original=original)


def numbers(data, dtype):
    if dtype == 3:
        return [struct.unpack("<f", struct.pack("<I", bits << 16))[0]
                for (bits,) in struct.iter_unpack("<H", data)]
    return [v for (v,) in struct.iter_unpack("<" + {0: "d", 1: "f", 2: "e"}[dtype], data)]


def comparison(actual, expected, dtype):
    assert len(actual) == len(expected)
    exact = 0
    maximum = squared_error = squared_reference = 0.0
    beyond = 0
    failures = []
    max_storage_steps = 0
    storage_steps = {"0": 0, "1": 0, "2": 0, "3-4": 0, "5-16": 0, ">16": 0}
    for index, (a, b) in enumerate(zip(numbers(actual, dtype), numbers(expected, dtype))):
        assert math.isfinite(a) and math.isfinite(b), (index, a, b)
        error = abs(a - b)
        maximum = max(maximum, error)
        squared_error += error * error
        squared_reference += b * b
        exact += a == b
        if dtype in (2, 3):
            mantissa, minimum = (10, -24) if dtype == 2 else (7, -133)
            ulp = 2 ** max(minimum, math.floor(math.log2(abs(b))) - mantissa) if b else 2 ** minimum
        else:
            ulp = 2e-5 + 2e-5 * abs(b)
        beyond += error > ulp + 2e-6
        if dtype == 3:
            bits_a = struct.unpack_from("<H", actual, index * 2)[0]
            bits_b = struct.unpack_from("<H", expected, index * 2)[0]
            ordered = lambda bits: 32768 - (bits & 32767) if bits & 32768 else 32768 + bits
            steps = abs(ordered(bits_a) - ordered(bits_b))
            max_storage_steps = max(max_storage_steps, steps)
            bucket = str(steps) if steps < 3 else "3-4" if steps <= 4 else "5-16" if steps <= 16 else ">16"
            storage_steps[bucket] += 1
        if a != b and len(failures) < 12:
            failures.append(dict(index=index, actual=a, expected=b, absoluteError=error, allowance=ulp))
    return dict(elements=len(actual) // types[{0: "F64", 1: "F32", 2: "F16", 3: "BF16"}[dtype]][1],
                exact=exact, byteExact=actual == expected, maxAbsoluteError=maximum,
                relativeL2=math.sqrt(squared_error / max(squared_reference, 1e-30)),
                beyondAllowance=beyond, firstDifferences=failures,
                bf16StorageSteps=storage_steps if dtype == 3 else None,
                maxBf16StorageSteps=max_storage_steps if dtype == 3 else None)


q, k, v = [tensor("decoder.attention." + name) for name in ("query", "key", "value")]
prefix = [tensor("encoder.prefix.layer0." + name) for name in ("keys", "values")]
context = tensor(calls["decoder.attention"]["output"][0]["tensor"])
probabilities = tensor("decoder.attention.probabilities")
rounded_qk = tensor("decoder.attention.rounded_qk")
masked_scores = tensor("decoder.attention.masked_scores")
assert len(q["shape"]) == 4 and len(k["shape"]) == 4
batch, qheads, tokens, dim = q["shape"]
kb, kheads, positions, kd = k["shape"]
cursor = positions - tokens
assert batch == kb and dim == kd and qheads % kheads == 0 and cursor >= 0
assert min(batch, qheads, kheads, tokens, dim) > 0
assert v["shape"] == k["shape"] and q["dtype"] == k["dtype"] == v["dtype"]
assert all(p["shape"] == [batch, kheads, cursor, dim] and p["dtype"] == q["dtype"] for p in prefix)
assert context["shape"] == [batch, tokens, qheads, dim] and context["dtype"] == q["dtype"]
assert all(t["shape"] == [batch, qheads, tokens, positions] and t["dtype"] == q["dtype"]
           for t in (probabilities, rounded_qk, masked_scores))
assert calls["decoder.attention"]["kwargs"]["scaling"] == 1.0, "replay expects the pinned scale=1 boundary"
mask_ref = calls["decoder.attention"]["mask"]
if mask_ref is not None:
    mask = tensor(mask_ref["tensor"])
    assert all(value == 0 for value in numbers(mask["data"], mask["dtype"])), "capture has an attention mask not represented by this ReadOnly replay"
assert rounded_qk["data"] == masked_scores["data"], "capture has another score transform"


def token_first(data, heads, time, width):
    # [B,H,T,D] -> [B,T,H,D], preserving storage bytes.
    row_bytes = dim * width
    return b"".join(data[((b * heads + h) * time + t) * row_bytes:
                        ((b * heads + h) * time + t + 1) * row_bytes]
                    for b in range(batch) for t in range(time) for h in range(heads))


def current_rows(t):
    row_bytes = dim * t["width"]
    return b"".join(t["data"][((b * kheads + h) * positions + cursor) * row_bytes:
                              ((b * kheads + h) * positions + positions) * row_bytes]
                    for b in range(batch) for h in range(kheads))


a = Args()
a.elements = batch * qheads * tokens * dim
a.compute_dtype = 0 if q["dtype"] == 0 else 1
a.output_dtype = q["dtype"]
a.inputs[0] = upload(q["data"])
current = [current_rows(t) for t in (k, v)]
for role in range(3):
    a.input_dtypes[role] = q["dtype"]
for role in range(2):
    a.inputs[role + 1] = upload(current[role])
guards = bytes([165]) * 16
guarded = []


def protected(data):
    ptr = upload(guards + data + guards) + 16
    guarded.append((ptr, len(data)))
    return ptr


prefix_data = [token_first(t["data"], kheads, cursor, t["width"]) for t in prefix]
prefix_ptrs = [protected(data) for data in prefix_data]
row_bytes = kheads * dim * q["width"]
current_ptrs = [protected(bytes([165]) * (batch * tokens * row_bytes)) for _ in range(2)]
table = [0] * (batch * 4)
for b in range(batch):
    table[b * 4:b * 4 + 4] = [0, cursor, positions, len(table)]
    for p in range(positions):
        pointers = prefix_ptrs if p < cursor else current_ptrs
        offset = b * (cursor if p < cursor else tokens) + (p if p < cursor else p - cursor)
        table.extend([pointers[0] + offset * row_bytes, pointers[1] + offset * row_bytes, 0, 0])
    for role in range(2):
        full = token_first((k, v)[role]["data"], kheads, positions, q["width"])
        assert full[b * positions * row_bytes:(b * positions + cursor) * row_bytes] == prefix_data[role][b * cursor * row_bytes:(b + 1) * cursor * row_bytes]
a.inputs[3] = upload(struct.pack("<" + "Q" * len(table), *table))
capacity = max(544, cursor)
stride = capacity + tokens
score_width = 4  # Stepwise attention uses F32 scratch even for F64 operands.
a.inputs[4] = protected(bytes(batch * qheads * tokens * stride * score_width))
a.inputs[7] = upload(struct.pack("<" + "I" * batch, *([cursor] * batch)))
a.scratch[0] = upload(struct.pack("<" + "I" * batch, *([tokens] * batch)))
a.scratch[3] = upload(bytes(4))
a.output = protected(bytes(a.elements * q["width"]))
kvshape = [batch, kheads, tokens, dim]
metadata = [4, 4, 4, 4, 0, 0, 0, 0, 0] + q["shape"] * 2 + kvshape * 2
a.metadata = upload(struct.pack("<" + "Q" * len(metadata), *metadata))
a.integers[0], a.integers[1], a.integers[2] = capacity, q["dtype"], 1
a.integers[3], a.integers[4], a.integers[5] = 0, 1, batch
a.integers[6], a.integers[7], a.integers[8] = 1, tokens, q["dtype"]
a.integers[9], a.integers[10], a.integers[11], a.integers[13] = stride, dim, qheads, positions
a.scalars[0] = 1.0

report = dict(status="running", diagnostic=True, captureSha256=digest,
              productionSourceSha256=hashlib.sha256((header + text("cache.cu")).encode()).hexdigest(),
              geometry=dict(batch=batch, queryHeads=qheads, kvHeads=kheads, headDim=dim,
                            cursor=cursor, tokens=tokens, validLengths=[tokens] * batch,
                            capacity=capacity, access="ReadOnly", queryWindow=None, scale=1.0),
              serializedLayout="contiguous logical values; original strides not reapplied",
              originalMetadata={t["name"]: t["original"] for t in (q, k, v, context, probabilities, rounded_qk)},
              stages={})


def persist():
    (output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def record(name, actual, expected):
    result = comparison(actual, expected["data"], expected["dtype"])
    result["shape"] = expected["shape"]
    result["dtype"] = {0: "F64", 1: "F32", 2: "F16", 3: "BF16"}[expected["dtype"]]
    result["sha256"] = hashlib.sha256(actual).hexdigest()
    for difference in result["firstDifferences"]:
        index = difference["index"]
        coordinates = []
        for extent in reversed(expected["shape"]):
            coordinates.insert(0, index % extent)
            index //= extent
        difference["coordinates"] = coordinates
    (output_directory / (name + ".bin")).write_bytes(actual)
    report["stages"][name] = result
    persist()
    print(json.dumps({"stage": name, **result}), flush=True)


persist()
launch("cache", "et_kv_attention", a)
assert read(a.scratch[3], 4) == bytes(4), "production attention status failure"
actual_context = read(a.output, a.elements * q["width"])
if options.cache_replay_effect:
    assert q["dtype"] == 3
    expanded = b"".join(struct.pack("<I", bits << 16) for (bits,) in struct.iter_unpack("<H", actual_context))
    actual_effect = (options.cache_replay_effect / "production.layer0.attention.context.f32").read_bytes()
    assert expanded == actual_effect, "unchanged attention must reproduce actual production context"
    report["productionReproducesEffect"] = True
    report["effectContextSha256"] = hashlib.sha256(actual_effect).hexdigest()
record("production-context", token_first(actual_context, qheads, tokens, q["width"]), context)
# Capture normalized, storage-rounded probabilities from planned F32 scratch.
scratch = read(a.inputs[4], batch * qheads * tokens * stride * 4)
packed_probabilities = bytearray()
for row in range(batch * qheads * tokens):
    data = scratch[row * stride * 4:(row * stride + positions) * 4]
    for (value,) in struct.iter_unpack("<f", data):
        if q["dtype"] == 3:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            assert bits & 65535 == 0, "stepwise probability was not BF16 rounded"
            packed_probabilities.extend(struct.pack("<H", bits >> 16))
        else:
            packed_probabilities.extend(struct.pack("<" + {0: "d", 1: "f", 2: "e"}[q["dtype"]], value))
record("production-probabilities", bytes(packed_probabilities), probabilities)
launch("cache", "et_kv_attention", a)
assert read(a.output, len(actual_context)) == actual_context, "production replay is not deterministic"

diagnostic_source = r"""
extern "C" __global__ void et_replay_qk(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    et_u64 positions = a.integers[13], item = i / positions;
    et_u64 tokens = a.integers[7], qheads = a.integers[11], dim = a.integers[10];
    et_u64 sequence = item / (tokens * qheads), head = (item / tokens) % qheads;
    et_u64 kheads = et_shape(a, 1)[1];
    et_store(a.output, a.output_dtype, i,
        et_kv_score<float>(a, item * dim, sequence, i % positions, head * kheads / qheads, dim));
}
extern "C" __global__ void et_replay_softmax(CudaKernelArgs a) {
    et_u64 positions = a.integers[13], item = et_thread() / 32;
    if (item >= a.elements / positions) return;
    unsigned int lane = threadIdx.x & 31U;
    float maximum = -1.0f / 0.0f;
    for (et_u64 p = lane; p < positions; p += 32)
        maximum = fmaxf(maximum, et_load<float>(a.inputs[6], a.input_dtypes[6], item * positions + p));
    maximum = et_kv_warp_max(maximum);
    if (a.operation && lane == 0) ((float*)a.inputs[5])[item] = maximum;
    float total = 0;
    for (et_u64 p = lane; p < positions; p += 32) {
        float value = exp(et_load<float>(a.inputs[6], a.input_dtypes[6], item * positions + p) - maximum);
        if (a.operation) ((float*)a.scratch[1])[item*positions+p] = value;
        total += value;
    }
    total = et_kv_warp_sum(total);
    if (a.operation && lane == 0) ((float*)a.scratch[2])[item] = total;
    for (et_u64 p = lane; p < positions; p += 32) {
        if (a.operation) ((float*)a.inputs[7])[item*positions+p] = exp(et_load<float>(a.inputs[6], a.input_dtypes[6], item*positions+p)-maximum)/total;
        et_store(a.output, a.output_dtype, item * positions + p,
            et_kv_round(exp(et_load<float>(a.inputs[6], a.input_dtypes[6], item * positions + p) - maximum) / total, a.integers[8]));
    }
}
extern "C" __global__ void et_replay_pv(CudaKernelArgs a) {
    et_u64 item = et_thread() / 32, dim = a.integers[10];
    if (item >= a.elements / dim) return;
    et_u64 tokens = a.integers[7], qheads = a.integers[11], positions = a.integers[13];
    et_u64 sequence = item / (tokens * qheads), head = (item / tokens) % qheads;
    et_u64 kheads = et_shape(a, 1)[1];
    unsigned int lane = threadIdx.x & 31U;
    for (et_u64 d = 0; d < dim; ++d) {
        float sum = 0;
        for (et_u64 p = lane; p < positions; p += 32)
            sum += et_load<float>(a.inputs[5], a.input_dtypes[5], item * positions + p)
                * et_cache_load(a, 1, sequence, p, head * kheads / qheads, d, dim);
        float value = et_kv_warp_sum(sum);
        if (lane == 0) et_store(a.output, a.output_dtype, item * dim + d, et_kv_round(value, a.integers[8]));
    }
}
"""
modules["cache_replay"] = compile_module("cache_replay", header + text("cache.cu") + diagnostic_source)
(output_directory / "diagnostic.cu").write_text(diagnostic_source)
report["diagnosticSourceSha256"] = hashlib.sha256(diagnostic_source.encode()).hexdigest()


def launch_diagnostic(name, args, work):
    function = c.c_void_p()
    checked(driver.cuModuleGetFunction(c.byref(function), modules["cache_replay"], name.encode()))
    params = (c.c_void_p * 1)(c.addressof(args))
    checked(driver.cuLaunchKernel(function, (work + 255) // 256, 1, 1, 256, 1, 1, 0, None, params, None))
    checked(driver.cuCtxSynchronize())
    assert read(args.scratch[3], 4) == bytes(4), name


scores_args = Args.from_buffer_copy(a)
scores_args.elements = batch * qheads * tokens * positions
scores_args.output = protected(bytes(len(rounded_qk["data"])))
launch_diagnostic("et_replay_qk", scores_args, scores_args.elements)
record("native-rounded-qk", read(scores_args.output, len(rounded_qk["data"])), rounded_qk)
scores_args.inputs[6] = upload(masked_scores["data"])
scores_args.input_dtypes[6] = masked_scores["dtype"]
launch_diagnostic("et_replay_softmax", scores_args, batch * qheads * tokens * 32)
record("official-scores-native-softmax", read(scores_args.output, len(probabilities["data"])), probabilities)
for label, source in [("official", masked_scores["data"]), ("native", (output_directory / "native-rounded-qk.bin").read_bytes())]:
    stages = Args.from_buffer_copy(scores_args)
    stages.operation = 1
    stages.inputs[6] = upload(source)
    rows = batch * qheads * tokens
    stages.inputs[5] = protected(bytes(rows * 4))
    stages.scratch[1] = protected(bytes(rows * positions * 4))
    stages.scratch[2] = protected(bytes(rows * 4))
    stages.inputs[7] = protected(bytes(rows * positions * 4))
    launch_diagnostic("et_replay_softmax", stages, rows * 32)
    for name, ptr, count in [("maximum", stages.inputs[5], rows), ("exp", stages.scratch[1], rows * positions),
                             ("denominator", stages.scratch[2], rows), ("probabilities", stages.inputs[7], rows * positions)]:
        (output_directory / (label + "-scores-softmax-" + name + ".f32")).write_bytes(read(ptr, count * 4))
    assert read(stages.output, len(probabilities["data"])) == (probabilities["data"] if label == "official" else bytes(packed_probabilities))
pv_args = Args.from_buffer_copy(a)
pv_args.inputs[5] = upload(probabilities["data"])
pv_args.input_dtypes[5] = probabilities["dtype"]
launch_diagnostic("et_replay_pv", pv_args, batch * qheads * tokens * 32)
record("official-probabilities-native-pv", token_first(read(pv_args.output, len(actual_context)), qheads, tokens, q["width"]), context)
for role in range(2):
    assert read(prefix_ptrs[role], len(prefix_data[role])) == prefix_data[role], "immutable prefix changed"
    expected = token_first(current[role], kheads, tokens, q["width"])
    assert read(current_ptrs[role], len(expected)) == expected, "private current storage differs"
for ptr, size in guarded:
    assert read(ptr - 16, 16) == guards and read(ptr + size, 16) == guards, "allocation guard changed"
report["prefixUnchanged"] = report["currentStorageExact"] = report["storageGuardsIntact"] = report["deterministicReplay"] = True
report["status"] = "complete"
report["allStagesByteExact"] = all(stage["byteExact"] for stage in report["stages"].values())
persist()
print(json.dumps({"report": str(output_directory / "report.json"), "allStagesByteExact": report["allStagesByteExact"]}), flush=True)
if options.attention_stages:
    stages_path = ROOT / "attention-stages.py"
    (output_directory / stages_path.name).write_bytes(stages_path.read_bytes())
    exec(compile(stages_path.read_text(), str(stages_path), "exec"))
