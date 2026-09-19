"""Compile every production module and check typed launch ABIs on a CUDA device.

Run with the CUDA toolkit libraries on LD_LIBRARY_PATH:
    python3 crates/runtime-cuda/src/kernels/check-nvrtc.py
No Python packages are required. The full graph integration tests live in Rust.
"""

import ctypes as c
import ctypes.util
import math
from pathlib import Path
import re
import struct
import sys


ROOT = Path(__file__).resolve().parent


class Args(c.Structure):
    _fields_ = [
        ("inputs", c.c_uint64 * 8), ("output", c.c_uint64),
        ("scratch", c.c_uint64 * 4), ("metadata", c.c_uint64),
        ("elements", c.c_uint64), ("integers", c.c_uint64 * 16),
        ("scalars", c.c_double * 8), ("input_dtypes", c.c_uint32 * 8),
        ("output_dtype", c.c_uint32), ("compute_dtype", c.c_uint32),
        ("operation", c.c_uint32), ("reserved", c.c_uint32),
    ]


def checked(code):
    if code:
        raise RuntimeError(f"CUDA API returned {code}")


def library(name):
    return c.CDLL(ctypes.util.find_library(name) or f"lib{name}.so")


nvrtc = library("nvrtc")
driver = library("cuda")
checked(driver.cuInit(0))
context = c.c_void_p()
checked(driver.cuDevicePrimaryCtxRetain(c.byref(context), 0))
checked(driver.cuCtxSetCurrent(context))
major, minor = c.c_int(), c.c_int()
checked(driver.cuDeviceGetAttribute(c.byref(major), 75, 0))
checked(driver.cuDeviceGetAttribute(c.byref(minor), 76, 0))


def compile_module(name, source):
    program = c.c_void_p()
    checked(nvrtc.nvrtcCreateProgram(c.byref(program), source.encode(), name.encode(), 0, None, None))
    options = (c.c_char_p * 3)(f"--gpu-architecture=compute_{major.value}{minor.value}".encode(), b"--std=c++17", b"--fmad=false")
    status = nvrtc.nvrtcCompileProgram(program, len(options), options)
    size = c.c_size_t()
    checked(nvrtc.nvrtcGetProgramLogSize(program, c.byref(size)))
    log = c.create_string_buffer(size.value)
    checked(nvrtc.nvrtcGetProgramLog(program, log))
    if status:
        raise RuntimeError(f"{name}: {log.value.decode()}")
    checked(nvrtc.nvrtcGetPTXSize(program, c.byref(size)))
    ptx = c.create_string_buffer(size.value)
    checked(nvrtc.nvrtcGetPTX(program, ptx))
    checked(nvrtc.nvrtcDestroyProgram(c.byref(program)))
    module = c.c_void_p()
    checked(driver.cuModuleLoadData(c.byref(module), ptx))
    print(f"compiled {name}", flush=True)
    return module


def text(name):
    return (ROOT / name).read_text()


header = text("typed.cuh")
prelude = re.search(r'const F32_PRELUDE: &str = r#"(.*?)"#;', (ROOT.parent / "device.rs").read_text(), re.S).group(1)
modules = {}
scalar_only = "--scalar-only" in sys.argv
cache_only = "--cache-only" in sys.argv
for name in (["cache"] if cache_only else ["typed"] if scalar_only else ["typed", "quantized", "cache"]):
    modules[name] = compile_module(name, header + text(f"{name}.cu"))
for name in ([] if scalar_only or cache_only else ["pointwise", "tensor", "linalg", "neural", "stateful"]):
    for dtype in ["f32", "f64"]:
        source = "\n".join([header, prelude if dtype == "f32" else "", text("common.cuh"), f"#define ET_{name.upper()}", text(f"{name}.cu"), text("compute.cu")])
        modules[f"{name}_{dtype}"] = compile_module(f"{name}_{dtype}", source)

allocations = []


def upload(data):
    ptr = c.c_uint64()
    checked(driver.cuMemAlloc_v2(c.byref(ptr), c.c_size_t(len(data))))
    allocations.append(ptr)
    checked(driver.cuMemcpyHtoD_v2(ptr, data, c.c_size_t(len(data))))
    return ptr.value


def read(ptr, size):
    out = c.create_string_buffer(size)
    checked(driver.cuMemcpyDtoH_v2(out, c.c_uint64(ptr), c.c_size_t(size)))
    return out.raw


def launch(module, name, args):
    function = c.c_void_p()
    checked(driver.cuModuleGetFunction(c.byref(function), modules[module], name.encode()))
    params = (c.c_void_p * 1)(c.addressof(args))
    work_items = args.integers[5] * 32 if name == "et_kv_attention" else args.elements
    checked(driver.cuLaunchKernel(function, (work_items + 255) // 256, 1, 1, 256, 1, 1, 0, None, params, None))
    checked(driver.cuCtxSynchronize())


def convert(data, src, dst, count, width):
    a = Args()
    a.inputs[0] = upload(data)
    a.input_dtypes[0] = src
    allocation = upload(b"\xa5" * (count * width + 16))
    a.output = allocation + 8
    a.output_dtype = dst
    a.elements = count
    launch("typed", "et_convert", a)
    result = read(allocation, count * width + 16)
    assert result[:8] == b"\xa5" * 8 and result[-8:] == b"\xa5" * 8
    return result[8:-8]


if cache_only:
    path = ROOT / "cache-tests.py"
    exec(compile(path.read_text(), str(path), "exec"))
    for ptr in allocations:
        checked(driver.cuMemFree_v2(ptr))
    checked(driver.cuModuleUnload(modules["cache"]))
    checked(driver.cuDevicePrimaryCtxRelease_v2(0))
    sys.exit(0)

assert c.sizeof(Args) == 360 and Args.scalars.offset == 248 and Args.input_dtypes.offset == 312
formats = ["d", "f", "e", "H", "q", "I", "B"]
for src, fmt in enumerate(formats):
    data = struct.pack("<" + fmt, 0x4228 if src == 3 else 42)
    for dst, outfmt in enumerate(formats):
        result = convert(data, src, dst, 1, struct.calcsize(outfmt))
        assert struct.unpack("<" + outfmt, result)[0] == (0x4228 if dst == 3 else 42)

for dtype in [2, 3]:
    bits = [i for i in range(65536) if (i & (0x7c00 if dtype == 2 else 0x7f80)) != (0x7c00 if dtype == 2 else 0x7f80)]
    payload = struct.pack("<" + "H" * len(bits), *bits)
    floats = convert(payload, dtype, 1, len(bits), 4)
    assert convert(floats, 1, dtype, len(bits), 2) == payload

tricky = (1 << 60) + (1 << 36) + 1
assert struct.unpack("<I", convert(struct.pack("<q", tricky), 4, 1, 1, 4))[0] == 0x5d800001
tricky = (1 << 55) + (1 << 47) + 1
assert convert(struct.pack("<q", tricky), 4, 3, 1, 2) == struct.pack("<H", 0x5b01)
assert convert(struct.pack("<q", (1 << 53) + 1), 4, 5, 1, 4) == struct.pack("<I", 1)
assert convert(struct.pack("<d", 1 + 2 ** -8 + 2 ** -40), 0, 3, 1, 2) == struct.pack("<H", 0x3f81)

# Scalar coercion is a semantic boundary before F32 arithmetic, independent
# of whether the tensor operand has already been promoted to F32 storage.
for dtype in [2, 3]:
    for scalar_role in [0, 1]:
        for promoted in [False, True]:
            a = Args()
            a.elements = 1
            a.operation = 2
            a.compute_dtype = a.output_dtype = 1
            a.inputs[scalar_role] = upload(struct.pack("<f", 1 + 2 ** (-11 if dtype == 2 else -8)))
            a.input_dtypes[scalar_role] = 1
            a.integers[scalar_role] = dtype + 1
            payload = struct.pack("<f", 3) if promoted else struct.pack("<H", 0x4200 if dtype == 2 else 0x4040)
            a.inputs[1 - scalar_role] = upload(payload)
            a.input_dtypes[1 - scalar_role] = 1 if promoted else dtype
            metadata = [1, 0 if scalar_role == 0 else 1, 0 if scalar_role == 1 else 1, 0, 0, 0, 0, 0, 0, 1, 1]
            a.metadata = upload(struct.pack("<" + "Q" * len(metadata), *metadata))
            a.output = upload(b"\0" * 4)
            a.scratch[3] = upload(b"\0" * 4)
            launch("typed", "et_binary", a)
            assert struct.unpack("<f", read(a.output, 4))[0] == 3
            assert read(a.scratch[3], 4) == b"\0" * 4

# Arg reductions use the descriptor's integer result dtype, independently of
# the input dtype. Poison the full allocation to catch a partial I64 write.
for output_dtype, fmt in [(4, "q"), (5, "I")]:
    for operation, expected in [(0, 1), (1, 2)]:
        a = Args()
        a.elements = 1
        a.operation = operation
        a.inputs[0] = upload(struct.pack("<qqq", 1 << 53, (1 << 53) + 1, -(1 << 53) - 1))
        a.input_dtypes[0] = 4
        a.output_dtype = output_dtype
        metadata = [0, 1, 0, 0, 0, 0, 0, 0, 0, 3]
        a.metadata = upload(struct.pack("<" + "Q" * len(metadata), *metadata))
        width = struct.calcsize(fmt)
        allocation = upload(b"\xa5" * (width + 16))
        a.output = allocation + 8
        launch("typed", "et_index", a)
        result = read(allocation, width + 16)
        assert result == b"\xa5" * 8 + struct.pack("<" + fmt, expected) + b"\xa5" * 8

if scalar_only:
    for ptr in allocations:
        checked(driver.cuMemFree_v2(ptr))
    checked(driver.cuModuleUnload(modules["typed"]))
    checked(driver.cuDevicePrimaryCtxRelease_v2(0))
    print("CUDA scalar rounding and descriptor-typed arg reduction checks passed")
    sys.exit(0)

probes = []
for bits in range(0x7bff):
    x = struct.unpack("<e", struct.pack("<H", bits))[0]
    y = struct.unpack("<e", struct.pack("<H", bits + 1))[0]
    midpoint = (x + y) / 2
    probes.extend([math.nextafter(midpoint, -math.inf), midpoint, math.nextafter(midpoint, math.inf)])
assert convert(struct.pack("<" + "d" * len(probes), *probes), 0, 2, len(probes), 2) == b"".join(struct.pack("<e", v) for v in probes)

# U32 CE targets must remain U32 in both compute modules.
for dtype, fmt in [(0, "d"), (1, "f")]:
    a = Args()
    a.inputs[0] = upload(struct.pack("<" + fmt * 6, 1, 2, 3, 3, 2, 1))
    a.inputs[1] = upload(struct.pack("<II", 2, 0))
    a.input_dtypes[1] = 5
    a.output = upload(b"\0" * 8)
    a.elements = 1
    a.metadata = upload(struct.pack("<" + "Q" * 12, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 3, 2))
    a.scratch[3] = upload(b"\0" * 4)
    a.integers[0] = (1 << 64) - 1
    a.compute_dtype = dtype
    launch(f"tensor_{'f64' if dtype == 0 else 'f32'}", "et_cross_entropy", a)
    result = struct.unpack("<" + fmt, read(a.output, struct.calcsize(fmt)))[0]
    assert abs(result - math.log(1 + math.exp(-1) + math.exp(-2))) < 1e-6
    assert read(a.scratch[3], 4) == b"\0" * 4

    # F32 persisted state has its own pointer type even in the F64 module.
    def numbers(values):
        return upload(struct.pack("<" + fmt * len(values), *values))

    a = Args()
    a.elements = 2
    for role, values in enumerate([[2, 3], [.5, .25], [4, 8], [0, 0], [1, 1], [1, 1]]):
        a.inputs[role] = numbers(values)
    shape = [1, 1, 2, 1]
    shapes = shape + shape * 4 + [1, 1, 2] + shape
    metadata = [4, 4, 4, 4, 4, 3, 4, 0, 0] + shapes
    a.metadata = upload(struct.pack("<" + "Q" * len(metadata), *metadata))
    a.output = numbers([0, 0])
    a.compute_dtype = dtype
    a.scratch[0] = upload(struct.pack("<ff", 0, 12345))
    a.scratch[2] = upload(struct.pack("<I", 2))
    a.scalars[0] = 1
    a.integers[1] = 1
    module = f"stateful_{'f64' if dtype == 0 else 'f32'}"
    launch(module, "et_kda", a)
    assert struct.unpack("<" + fmt * 2, read(a.output, 2 * struct.calcsize(fmt))) == (4, 11.625)
    assert struct.unpack("<ff", read(a.scratch[0], 8)) == (3.875, 12345)
    a.integers[0] = 1
    a.integers[1] = 0
    a.scratch[0] = numbers([0, 0, 0])
    a.scratch[1] = numbers([0, 0])
    for role, expected in enumerate([(2, 3.875), (19.25, 21), (2.40625, .75), (0, 5.625), (9.625, 5.625)]):
        a.operation = role
        launch(module, "et_kda", a)
        assert struct.unpack("<" + fmt * 2, read(a.output, 2 * struct.calcsize(fmt))) == expected

# Quantized KV cache stores bytes and independent F32 scales.
a = Args()
a.elements = 2
a.inputs[0] = upload(struct.pack("<ff", 0, 0))
a.inputs[1] = upload(struct.pack("<ff", 1, -2))
a.inputs[2] = upload(struct.pack("<ff", 3, -4))
a.inputs[3] = upload(b"\xa5" * 8)
a.inputs[4] = upload(b"\xa5" * 8)
a.inputs[5] = upload(b"\0" * 4)
a.inputs[6] = upload(b"\0" * 4)
a.inputs[7] = upload(b"\0" * 4)
a.scratch[0] = upload(struct.pack("<I", 1))
a.output = upload(b"\0" * 8)
a.output_dtype = a.compute_dtype = 1
for role in range(3):
    a.input_dtypes[role] = 1
metadata = [4, 4, 4, 4, 0, 0, 0, 0, 0] + [1, 1, 1, 2] * 4
a.metadata = upload(struct.pack("<" + "Q" * len(metadata), *metadata))
a.integers[0] = a.integers[2] = a.integers[5] = 1
a.integers[1] = 6
a.scalars[0] = 1
launch("cache", "et_kv_attention", a)
assert read(a.inputs[3], 8) == bytes([192, 1]) + b"\xa5" * 6
assert read(a.inputs[4], 8)[2:] == b"\xa5" * 6
assert struct.unpack("<f", read(a.inputs[5], 4))[0] == struct.unpack("<f", struct.pack("<f", 2 / 127))[0]
result = struct.unpack("<ff", read(a.output, 8))
assert abs(result[0] - 3) <= 4 / 127 and result[1] == -4

for ptr in allocations:
    checked(driver.cuMemFree_v2(ptr))
for module in modules.values():
    checked(driver.cuModuleUnload(module))
checked(driver.cuDevicePrimaryCtxRelease_v2(0))
print("CUDA NVRTC compilation and device cast/CE/KDA/cache ABI checks passed")
