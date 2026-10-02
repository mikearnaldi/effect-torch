"""Bounded captured expert-dot replay, dispatched by check-nvrtc.py.

No production arithmetic is changed. The production kernel's F32-output call
exposes its pre-store sum. F32 cuBLAS results are preserved before a host-only
BF16 comparison conversion; these diagnostics never feed model execution.
"""
import array
import hashlib
import json
import runpy

helpers = runpy.run_path(str(ROOT / "cublas-replay.py"))
compare_bf16 = helpers["compare"]
DEFAULT_MATH = helpers["DEFAULT_MATH"]
STRICT_MATH = helpers["DISALLOW_REDUCED_PRECISION_REDUCTION"]


def option(name):
    return Path(sys.argv[sys.argv.index(name) + 1])


class Archive:
    def __init__(self, path):
        self.path = path
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            self.header = json.loads(stream.read(size))
            self.base = 8 + size

    def get(self, name, expert=None):
        item = self.header[name]
        start, end = item["data_offsets"]
        shape = item["shape"]
        if expert is not None:
            assert item["dtype"] == "BF16" and len(shape) == 3
            assert 0 <= expert < shape[0]
            size = math.prod(shape[1:]) * 2
            start += expert * size
            end = start + size
            shape = shape[1:]
        with self.path.open("rb") as stream:
            stream.seek(self.base + start)
            data = stream.read(end - start)
        assert len(data) == end - start
        return shape, item["dtype"], data


def floats(data):
    result = array.array("f")
    result.frombytes(data)
    return result


def bf16_floats(data):
    return [struct.unpack("<f", struct.pack("<I", bits[0] << 16))[0]
            for bits in struct.iter_unpack("<H", data)]


def narrow(data):
    bits = array.array("I")
    bits.frombytes(data)
    result = array.array("H", [((b + 0x7fff + ((b >> 16) & 1)) >> 16) & 65535 for b in bits])
    assert all(math.isfinite(v) for v in floats(data))
    return result.tobytes()


def rounding_interval(bits):
    ordered = 32768 - (bits & 32767) if bits & 32768 else 32768 + bits
    def value(order):
        raw = 0x8000 | (32768 - order) if order < 32768 else order - 32768
        return struct.unpack("<f", struct.pack("<I", raw << 16))[0]
    center = value(ordered)
    return (value(ordered - 1) + center) / 2, (value(ordered + 1) + center) / 2


def dot_evidence(x, w, expected, native, strict, default, shape, columns):
    actual, reference = bf16_floats(native), bf16_floats(expected)
    different = [i for i, (a, b) in enumerate(zip(actual, reference)) if a != b]
    selected = list(dict.fromkeys(different[:8] + sorted(different, key=lambda i: abs(actual[i] - reference[i]), reverse=True)[:24]))
    xv, wv = bf16_floats(x), bf16_floats(w)
    strict_values, default_values = floats(strict), floats(default)
    native_values = floats(unrounded_native)
    inner = shape[1]
    # Any order/tree of at most K-1 F32 additions with exact F32 products has
    # this standard worst-case absolute error bound. Include gradual-underflow
    # and F64 reference rounding allowances. This does not assume a GEMM tree.
    u = 2 ** -24
    gamma = (inner - 1) * u / (1 - (inner - 1) * u)
    results = []
    for index in selected:
        row, column = divmod(index, columns)
        products = [xv[row * inner + k] * wv[column * inner + k] for k in range(inner)]
        exact_products = all(math.isfinite(p) and struct.unpack("<f", struct.pack("<f", p))[0] == p for p in products)
        dot = math.fsum(products)
        absolute_sum = math.fsum(abs(p) for p in products)
        bound = gamma * absolute_sum + inner * 2 ** -149 + abs(dot) * 2 ** -52
        bits = struct.unpack_from("<H", expected, index * 2)[0]
        low, high = rounding_interval(bits)
        distance = max(low - dot, dot - high, 0)
        results.append(dict(index=index, coordinates=[row, column], official=reference[index],
                            native=actual[index], nativeUnrounded=native_values[index],
                            strictCublasUnrounded=strict_values[index], defaultCublasUnrounded=default_values[index],
                            highPrecisionDot=dot, sumAbsoluteProducts=absolute_sum,
                            allProductsExactlyF32=exact_products, anyF32ReductionAbsoluteErrorBound=bound,
                            officialRoundingInterval=[low, high], distanceToOfficialRoundingInterval=distance,
                            officialOutsideAnyF32Reduction=exact_products and distance > bound))
    return results


capture = option("--expert-replay")
checkpoint = option("--expert-replay-checkpoint")
effect = option("--expert-replay-effect")
output = option("--expert-replay-output")
output.mkdir(parents=True, exist_ok=False)
manifest = json.loads((capture / "manifest.json").read_text())
archive = Archive(capture / "experts.safetensors")
assert manifest["status"] == "passed"
assert hashlib.sha256(archive.path.read_bytes()).hexdigest() == manifest["tensor_file_sha256"]
assert all(phase["captured_output_exact"] for phase in manifest["phases"].values())
index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
source = (ROOT / "expert-replay.py").read_bytes()
(output / "replay.py").write_bytes(source)
(output / "typed.cu").write_text(header + text("typed.cu"))
report = dict(status="running", diagnostic=True, captureSha256=manifest["tensor_file_sha256"],
              sourceSha256=hashlib.sha256(source).hexdigest(), cases=[])
blas = library("cublas")
handle = c.c_void_p()
checked(blas.cublasCreate_v2(c.byref(handle)))
version = c.c_int()
checked(blas.cublasGetVersion_v2(handle, c.byref(version)))
report["cublasVersion"] = version.value
workspace = upload(b"\0" * (32 << 20))


def persist():
    path = output / "report.json.next"
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.replace(output / "report.json")


try:
    # Include large/mid-size encoder groups with substantial drift and small
    # decoder control groups. Both projections reset to their official input.
    for phase, expert in [("encoder", 102), ("encoder", 47), ("encoder", 99), ("decoder", 0), ("decoder", 79)]:
        prefix = f"{phase}.expert.{expert}."
        row_shape, _, rows_data = archive.get(prefix + "rows")
        _, _, slots_data = archive.get(prefix + "slots")
        for projection, input_name, result_name in [("gate_up_proj", "input", "gate_up"), ("down_proj", "product", "down")]:
            allocation_start = len(allocations)
            try:
                xs, xd, x = archive.get(prefix + input_name)
                ys, yd, expected = archive.get(prefix + result_name)
                weight_name = "model.decoder.layers.0.experts." + projection
                ws, wd, w = Archive(checkpoint / index[weight_name]).get(weight_name, expert)
                assert xd == yd == wd == "BF16" and xs[0] == row_shape[0]
                m, k, n = xs[0], xs[1], ws[0]
                assert ws[1] == k and ys == [m, n]
                name = prefix + projection
                entry = dict(name=name, shape=xs, weightShape=ws, outputShape=ys, weightExpert=expert,
                             inputSha256=hashlib.sha256(x).hexdigest(), weightSha256=hashlib.sha256(w).hexdigest(),
                             rows=list(struct.unpack("<" + "q" * m, rows_data)),
                             slots=list(struct.unpack("<" + "q" * m, slots_data)), results=[])
                report["cases"].append(entry)
                for label, data in [("input.bf16", x), ("weight.bf16", w), ("official.bf16", expected)]:
                    (output / (name + "." + label)).write_bytes(data)
                guards = []
                def guarded(data):
                    base = upload(b"\xa5" * 256 + data + b"\xa5" * 256)
                    guards.append((base, len(data)))
                    return base + 256
                xp, wp = guarded(x), guarded(w)
                status = guarded(b"\0" * 4)
                args = Args()
                args.inputs[0], args.inputs[1], args.inputs[2] = xp, wp, guarded(b"\0" * (4 * m))
                args.input_dtypes[0], args.input_dtypes[1], args.input_dtypes[2] = 3, 3, 5
                args.integers[0], args.integers[1], args.integers[2], args.integers[3] = m, n, k, 1
                args.elements, args.compute_dtype, args.scratch[3] = m * n, 1, status
                results = {}
                def save(label, data, dtype, config):
                    (output / (name + "." + label + "." + dtype)).write_bytes(data)
                    results[label] = data
                    rounded = narrow(data) if dtype == "f32" else data
                    entry["results"].append(dict(label=label, dtype=dtype, config=config,
                                                  comparison=compare_bf16(rounded, expected)))
                    persist()
                function = c.c_void_p()
                checked(driver.cuModuleGetFunction(c.byref(function), modules["typed"], b"et_expert_linear_rows"))
                for dtype, code, width in [("bf16", 3, 2), ("f32", 1, 4)]:
                    args.output = guarded(b"\xa5" * (m * n * width))
                    args.output_dtype = code
                    parameters = (c.c_void_p * 1)(c.addressof(args))
                    checked(driver.cuLaunchKernel(function, min(65535, (m * n + 7) // 8), 1, 1, 256, 1, 1, 0, None, parameters, None))
                    checked(driver.cuCtxSynchronize())
                    assert read(status, 4) == b"\0" * 4
                    save("native-" + dtype, read(args.output, m * n * width), dtype, {"kernel": "et_expert_linear_rows", "outputDtype": code})
                unrounded_native = results["native-f32"]
                assert narrow(unrounded_native) == results["native-bf16"]
                prior = (effect / (name + ".expertLinear.f32")).read_bytes()
                assert narrow(prior) == results["native-bf16"], name + " did not reproduce component 20"
                for mode_name, mode in [("strict", STRICT_MATH), ("default", DEFAULT_MATH)]:
                    checked(blas.cublasSetMathMode(handle, mode))
                    actual_mode = c.c_int()
                    checked(blas.cublasGetMathMode(handle, c.byref(actual_mode)))
                    assert actual_mode.value == mode
                    checked(blas.cublasSetWorkspace_v2(handle, c.c_void_p(workspace), c.c_size_t(32 << 20)))
                    for dtype, ctype, width in [("bf16", 14, 2), ("f32", 0, 4)]:
                        yp = guarded(b"\xa5" * (m * n * width))
                        alpha, beta = c.c_float(1), c.c_float(0)
                        checked(blas.cublasGemmStridedBatchedEx(
                            handle, 1, 0, n, m, k, c.byref(alpha), c.c_void_p(wp), 14, k, c.c_longlong(n * k),
                            c.c_void_p(xp), 14, k, c.c_longlong(m * k), c.byref(beta), c.c_void_p(yp), ctype, n,
                            c.c_longlong(m * n), 1, 68, -1))
                        checked(driver.cuCtxSynchronize())
                        save(mode_name + "-" + dtype, read(yp, m * n * width), dtype,
                             dict(mathMode=mode, computeType=68, outputType=ctype, workspaceBytes=32 << 20))
                assert results["default-bf16"] == expected, name + " default grouped cuBLAS differs from oracle"
                entry["dotEvidence"] = dot_evidence(x, w, expected, results["native-bf16"], results["strict-f32"], results["default-f32"], xs, n)
                for base, size in guards:
                    assert read(base, 256) == read(base + 256 + size, 256) == b"\xa5" * 256
                assert read(xp, len(x)) == x and read(wp, len(w)) == w
                entry["guardsAndInputsUnchanged"] = True
                persist()
                print(json.dumps(dict(name=name, comparisons=[dict(label=r["label"], **r["comparison"]) for r in entry["results"]],
                                      outsideAnyF32=sum(r["officialOutsideAnyF32Reduction"] for r in entry["dotEvidence"]))), flush=True)
            finally:
                while len(allocations) > allocation_start:
                    checked(driver.cuMemFree_v2(allocations.pop()))
    report["status"] = "completed"
finally:
    checked(blas.cublasDestroy_v2(handle))
    persist()
