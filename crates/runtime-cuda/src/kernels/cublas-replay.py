"""Replay captured BF16 projections against cuBLAS without loading the model.

Use --torch-reference in the pinned reference venv to record its unchanged
flags and linear outputs. Otherwise only the Python standard library and the
CUDA driver/cuBLAS shared libraries are needed. Every run uses a new output
directory and preserves raw BF16 results.
"""

import argparse
import array
import ctypes as c
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import struct


# Values from the CUDA 12.9 cublas_api.h ABI, also verified by cuBLAS logs.
DEFAULT_MATH = 0
DISALLOW_REDUCED_PRECISION_REDUCTION = 16
COMPUTE_32F = 68
BF16 = 14


def tensor(path, name):
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        item = header[name]
        assert item["dtype"] == "BF16", (name, item)
        start, end = item["data_offsets"]
        stream.seek(8 + length + start)
        data = stream.read(end - start)
        assert len(data) == math.prod(item["shape"]) * 2
        return item["shape"], data


def compare(actual, expected):
    assert len(actual) == len(expected)
    result = {"elements": len(actual) // 2, "byteExact": actual == expected,
              "sha256": hashlib.sha256(actual).hexdigest()}
    if actual == expected:
        return dict(result, exact=len(actual) // 2, maxAbsoluteError=0,
                    relativeL2=0, beyondAllowance=0, maxBf16StorageSteps=0)
    a, b = array.array("H"), array.array("H")
    a.frombytes(actual)
    b.frombytes(expected)
    # A full BF16 decoding table keeps comparison independent of NumPy/Torch.
    values = [struct.unpack("<f", struct.pack("<I", bits << 16))[0] for bits in range(65536)]
    exact = beyond = maximum_steps = 0
    maximum = error_sum = reference_sum = 0.0
    histogram = {"0": 0, "1": 0, "2": 0, "3-4": 0, "5-16": 0, ">16": 0}
    first = []
    for index, (ab, bb) in enumerate(zip(a, b)):
        av, bv = values[ab], values[bb]
        assert math.isfinite(av) and math.isfinite(bv), (index, av, bv)
        error = abs(av - bv)
        exact += ab == bb
        maximum = max(maximum, error)
        error_sum += error * error
        reference_sum += bv * bv
        allowance = 2 ** max(-133, ((bb >> 7) & 255) - 134) + 2e-6
        beyond += error > allowance
        ao = 32768 - (ab & 32767) if ab & 32768 else 32768 + ab
        bo = 32768 - (bb & 32767) if bb & 32768 else 32768 + bb
        steps = abs(ao - bo)
        maximum_steps = max(maximum_steps, steps)
        bucket = str(steps) if steps < 3 else "3-4" if steps <= 4 else "5-16" if steps <= 16 else ">16"
        histogram[bucket] += 1
        if ab != bb and len(first) < 8:
            first.append(dict(index=index, actual=av, expected=bv, steps=steps))
    return dict(result, exact=exact, maxAbsoluteError=maximum,
                relativeL2=math.sqrt(error_sum / reference_sum), beyondAllowance=beyond,
                maxBf16StorageSteps=maximum_steps, bf16StorageSteps=histogram, firstDifferences=first)


def checked(status):
    if status:
        raise RuntimeError(f"CUDA/cuBLAS status {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-reference", action="store_true")
    parser.add_argument("--configuration", help="Run only this native configuration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output / "replay.py").write_bytes(source)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    archive = args.capture / "layer0.safetensors"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == manifest["tensor_file_sha256"]
    assert manifest["status"] == "passed" and manifest["archived_answer_exact"] and manifest["prefix_unchanged"]
    weights = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    report = dict(captureSha256=digest, replaySourceSha256=hashlib.sha256(source).hexdigest(), diagnostic=True, cases=[],
                  environment={name: os.environ.get(name) for name in
                               ["LD_LIBRARY_PATH", "CUBLAS_WORKSPACE_CONFIG", "CUBLAS_LOGINFO_DBG", "CUBLAS_LOGDEST_DBG"]})

    def persist():
        path = args.output / "report.json.next"
        path.write_text(json.dumps(report, indent=2) + "\n")
        path.replace(args.output / "report.json")

    if args.torch_reference:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["torch"] = dict(version=torch.__version__, cuda=torch.version.cuda,
                               allowTf32=torch.backends.cuda.matmul.allow_tf32,
                               allowBf16ReducedPrecisionReduction=str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
                               preferredBlas=str(torch.backends.cuda.preferred_blas_library()),
                               device=torch.cuda.get_device_name())
    else:
        driver = c.CDLL(ctypes.util.find_library("cuda") or "libcuda.so")
        blas = c.CDLL(ctypes.util.find_library("cublas") or "libcublas.so")
        checked(driver.cuInit(0))
        context = c.c_void_p()
        checked(driver.cuDevicePrimaryCtxRetain(c.byref(context), 0))
        checked(driver.cuCtxSetCurrent(context))
        handle = c.c_void_p()
        checked(blas.cublasCreate_v2(c.byref(handle)))
        version = c.c_int()
        checked(blas.cublasGetVersion_v2(handle, c.byref(version)))
        report["cublasVersion"] = version.value
        workspace = c.c_uint64()
        checked(driver.cuMemAlloc_v2(c.byref(workspace), c.c_size_t(32 << 20)))

        def upload(data):
            ptr = c.c_uint64()
            checked(driver.cuMemAlloc_v2(c.byref(ptr), c.c_size_t(len(data))))
            checked(driver.cuMemcpyHtoD_v2(ptr, data, c.c_size_t(len(data))))
            return ptr

    try:
        for phase in ["decoder", "encoder"]:
            for projection in ["router.proj", "mlp.down_proj", "self_attn.o_proj", "self_attn.q_proj"]:
                name = phase + ".layer0." + projection
                xs, x = tensor(archive, name + ".args.0")
                ys, y = tensor(archive, name + ".output")
                weight_name = "model.decoder.layers.0." + projection + ".weight"
                ws, w = tensor(args.checkpoint / weights[weight_name], weight_name)
                m, k, n = math.prod(xs[:-1]), xs[-1], ws[0]
                assert ws[1] == k and math.prod(ys) == m * n
                entry = dict(name=name, shape=xs, weightShape=ws, outputShape=ys,
                             originalMetadata=manifest["original_metadata"][name + ".args.0"],
                             xSha256=hashlib.sha256(x).hexdigest(), weightSha256=hashlib.sha256(w).hexdigest(), results=[])
                report["cases"].append(entry)

                def save(label, data, config):
                    (args.output / (name + "." + label + ".bf16")).write_bytes(data)
                    result = dict(label=label, config=config, comparison=compare(data, y))
                    entry["results"].append(result)
                    persist()
                    print(json.dumps(dict(name=name, **result)), flush=True)

                if args.torch_reference:
                    xt = torch.frombuffer(bytearray(x), dtype=torch.bfloat16).reshape(xs).cuda()
                    wt = torch.frombuffer(bytearray(w), dtype=torch.bfloat16).reshape(ws).cuda()
                    with torch.no_grad():
                        yt = torch.nn.functional.linear(xt, wt)
                    torch.cuda.synchronize()
                    save("torch", yt.cpu().contiguous().view(torch.uint16).numpy().tobytes(), report["torch"])
                    del xt, wt, yt
                else:
                    xp, wp, yp = upload(x), upload(w), upload(b"\xa5" * len(y))
                    try:
                        configs = [
                            ("native", True, DISALLOW_REDUCED_PRECISION_REDUCTION, 1, COMPUTE_32F, -1),
                            ("batched-default", True, DEFAULT_MATH, 1, COMPUTE_32F, -1),
                            ("gemm-disallow", False, DISALLOW_REDUCED_PRECISION_REDUCTION, 1, COMPUTE_32F, -1),
                            ("gemm-default", False, DEFAULT_MATH, 1, COMPUTE_32F, -1),
                            ("batched-disallow-32MiB", True, DISALLOW_REDUCED_PRECISION_REDUCTION, 32, COMPUTE_32F, -1),
                            ("batched-default-32MiB", True, DEFAULT_MATH, 32, COMPUTE_32F, -1),
                            ("gemm-default-32MiB", False, DEFAULT_MATH, 32, COMPUTE_32F, -1),
                            ("torch-call", False, 0, 32, 0, 99),
                            ("torch-call-1MiB", False, 0, 1, 0, 99),
                        ]
                        if args.configuration:
                            configs = [config for config in configs if config[0] == args.configuration]
                            assert configs, args.configuration
                        for label, batched, mode, workspace_mib, compute, algorithm in configs:
                            checked(blas.cublasSetMathMode(handle, mode))
                            actual_mode = c.c_int()
                            checked(blas.cublasGetMathMode(handle, c.byref(actual_mode)))
                            assert actual_mode.value == mode
                            checked(blas.cublasSetWorkspace_v2(handle, c.c_void_p(workspace.value), c.c_size_t(workspace_mib << 20)))
                            alpha, beta = c.c_float(1), c.c_float(0)
                            parameters = [handle, 1, 0, n, m, k, c.byref(alpha), c.c_void_p(wp.value), BF16, k]
                            if batched:
                                parameters += [c.c_longlong(n * k)]
                            parameters += [c.c_void_p(xp.value), BF16, k]
                            if batched:
                                parameters += [c.c_longlong(m * k)]
                            parameters += [c.byref(beta), c.c_void_p(yp.value), BF16, n]
                            if batched:
                                parameters += [c.c_longlong(m * n), 1]
                            parameters += [compute, algorithm]
                            function = blas.cublasGemmStridedBatchedEx if batched else blas.cublasGemmEx
                            checked(function(*parameters))
                            checked(driver.cuCtxSynchronize())
                            data = c.create_string_buffer(len(y))
                            checked(driver.cuMemcpyDtoH_v2(data, yp, c.c_size_t(len(y))))
                            save(label, data.raw, dict(batched=batched, mathMode=mode, workspaceMiB=workspace_mib,
                                                      computeType=compute, algorithm=algorithm, cType=BF16))
                    finally:
                        for ptr in [xp, wp, yp]:
                            checked(driver.cuMemFree_v2(ptr))
        report["status"] = "completed"
    finally:
        report["loadedLibraries"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                            if any(name in line for name in ["libcublas", "libcuda.so", "libcudart"])})
        if not args.torch_reference:
            checked(blas.cublasDestroy_v2(handle))
            checked(driver.cuMemFree_v2(workspace))
            checked(driver.cuDevicePrimaryCtxRelease_v2(0))
        persist()


if __name__ == "__main__":
    main()
