"""Identical-input RMSNorm substages, dispatched by check-nvrtc.py.

Run in the pinned Torch reference venv with its matching wheel libraries.
Production RMS source is compiled unchanged. Diagnostic variants only expose
or reset F32 stages; no stage output feeds model execution.
"""
import hashlib
import inspect
import json
import os
import runpy

import torch
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaRMSNorm

helpers = runpy.run_path(str(ROOT / "cublas-replay.py"))
tensor_bytes = helpers["tensor"]
compare = helpers["compare"]

def option(name):
    return Path(sys.argv[sys.argv.index(name) + 1])

capture = option("--rms-replay")
checkpoint = option("--rms-replay-checkpoint")
effect = option("--rms-replay-effect")
output = option("--rms-replay-output")
output.mkdir(parents=True, exist_ok=False)
fixed = "--rms-fixed" in sys.argv
torch.set_grad_enabled(False)
assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
source = inspect.getsource(DiffusionGemmaRMSNorm)
(output / "official-rms.py").write_text(source)
for name in ["rms-replay.py", "compute.cu", "typed.cuh", "typed.cu", "check-nvrtc.py"]:
    (output / name).write_bytes((ROOT / name).read_bytes())
(output / "device.rs").write_bytes((ROOT.parent / "device.rs").read_bytes())

diagnostic = r'''
extern "C" __global__ void et_rms_stages(CudaKernelArgs a) {
    et_u64 row = et_thread(), width = a.integers[0];
    if (row >= a.elements) return;
    const float *x = (const float *)a.inputs[0] + row * width;
    const float *weight = (const float *)a.inputs[1];
    float sum = 0.0f;
    for (et_u64 k = 0; k < width; ++k) sum += x[k] * x[k];
    float mean = sum / width;
    if (a.operation == 2 || a.operation == 7) mean = ((const float *)a.inputs[2])[row];
    if (a.operation == 3) mean = ((const float *)a.inputs[3])[row] * (1.0f / (float)width);
    float eps_added = mean + (float)a.scalars[0];
    if (a.operation == 4) eps_added = ((const float *)a.inputs[4])[row];
    float inv = rsqrtf(eps_added);
    if (a.operation == 5) inv = ((const float *)a.inputs[5])[row];
    float *stats = (float *)a.scratch[0] + row * 8;
    stats[0] = sum; stats[1] = sum / width; stats[2] = sum * (1.0f / (float)width);
    stats[3] = mean; stats[4] = eps_added; stats[5] = inv;
    stats[6] = 1.0f / sqrtf(eps_added); stats[7] = sqrtf(eps_added);
    for (et_u64 k = 0; k < width; ++k) {
        float normalized = a.operation == 0 || a.operation == 7 ? x[k] / sqrtf(eps_added) : x[k] * inv;
        ((float *)a.scratch[1])[row * width + k] = normalized;
        ((float *)a.output)[row * width + k] = normalized * weight[k];
    }
}
// Diagnostic candidate: four independent F32 accumulators per lane, followed
// by a warp tree. Products round to F32 before addition (fmad=false).
extern "C" __global__ void et_rms_warp(CudaKernelArgs a) {
    et_u64 width = a.integers[0];
    unsigned int lane = threadIdx.x & 31U;
    for (et_u64 row = et_thread() / 32; row < a.elements; row += (et_u64)gridDim.x * (blockDim.x / 32)) {
        const float *x = (const float *)a.inputs[0] + row * width;
        const float *weight = (const float *)a.inputs[1];
        float p[4] = {0, 0, 0, 0};
        for (et_u64 k = lane * 4; k < width; k += 128) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) if (k + j < width) p[j] += x[k + j] * x[k + j];
        }
        float sum = ((p[0] + p[1]) + p[2]) + p[3];
        for (unsigned int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, offset);
        sum = __shfl_sync(0xffffffffU, sum, 0);
        float mean = sum * (1.0f / (float)width);
        float eps_added = mean + (float)a.scalars[0];
        float inv = rsqrtf(eps_added);
        if (!lane) {
            float *stats = (float *)a.scratch[0] + row * 8;
            stats[0] = sum; stats[1] = sum / width; stats[2] = mean; stats[3] = mean;
            stats[4] = eps_added; stats[5] = inv; stats[6] = 1.0f / sqrtf(eps_added); stats[7] = sqrtf(eps_added);
        }
        for (et_u64 k = lane; k < width; k += 32) {
            float normalized = x[k] * inv;
            ((float *)a.scratch[1])[row * width + k] = normalized;
            ((float *)a.output)[row * width + k] = weight ? normalized * weight[k] : normalized;
        }
    }
}
'''
(output / "diagnostic.cu").write_text(header + diagnostic)
modules["rms_stages"] = compile_module("rms_stages", header + diagnostic)

def raw(value):
    return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()

def store(name, data):
    (output / name).write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def f32(data):
    return [v[0] for v in struct.iter_unpack("<f", data)]

def compare_f32(actual, expected):
    assert len(actual) == len(expected)
    a, b = f32(actual), f32(expected)
    diff = [i for i, (av, bv) in enumerate(zip(a, b)) if av != bv]
    return dict(elements=len(a), exact=len(a) - len(diff),
                maxAbsoluteError=max((abs(a[i] - b[i]) for i in diff), default=0),
                first=[dict(index=i, actual=a[i], expected=b[i]) for i in diff[:12]])

def buffer(size):
    allocation = upload(b"\xa5" * (size + 16))
    return allocation + 8, allocation, size

def checked_bytes(buf):
    _, allocation, size = buf
    data = read(allocation, size + 16)
    assert data[:8] == b"\xa5" * 8 and data[-8:] == b"\xa5" * 8
    return data[8:-8]

report = dict(status="running", diagnostic=True, torchVersion=torch.__version__,
              flags=dict(tf32=torch.backends.cuda.matmul.allow_tf32,
                         bf16ReducedPrecisionReduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                         workspaceConfig=os.environ["CUBLAS_WORKSPACE_CONFIG"], nvrtcFmad=False),
              officialSourceSha256=hashlib.sha256(source.encode()).hexdigest(), cases=[])
def persist():
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
weight_name = "model.decoder.layers.0.input_layernorm.weight"
weight_shape, weight_bytes = tensor_bytes(checkpoint / index[weight_name], weight_name)
store("weight.bf16", weight_bytes)
weight = torch.frombuffer(bytearray(weight_bytes), dtype=torch.bfloat16).cuda()
wf = weight.float()
try:
    for phase in ([] if "--rms-synthetic" in sys.argv else ["decoder", "encoder"]):
        prefix = phase + ".layer0.input_layernorm"
        shape, xb = tensor_bytes(capture / "layer0.safetensors", prefix + ".args.0")
        expected_shape, expected = tensor_bytes(capture / "layer0.safetensors", prefix + ".output")
        assert shape == expected_shape and shape[-1:] == weight_shape
        rows, width = math.prod(shape[:-1]), shape[-1]
        x = torch.frombuffer(bytearray(xb), dtype=torch.bfloat16).reshape(shape).cuda()
        xf = x.float()
        squares = xf.pow(2)
        sum_ref = squares.sum(-1, keepdim=True)
        mean = squares.mean(-1, keepdim=True)
        eps_added = mean + 1e-6
        inverse = torch.pow(eps_added, -0.5)
        normalized = xf * inverse
        weighted = normalized * wf
        official = DiffusionGemmaRMSNorm(width, eps=1e-6).cuda().to(torch.bfloat16)
        official.weight.copy_(weight)
        assert raw(official(x)) == expected
        assert raw(weighted.to(torch.bfloat16)) == expected
        reference = dict(squares=squares, sum=sum_ref, mean=mean, eps=eps_added, inv=inverse,
                         normalized=normalized, weighted=weighted)
        reference_raw = {}
        for name, value in reference.items():
            reference_raw[name] = raw(value)
            store(phase + ".torch." + name + ".f32", reference_raw[name])
        store(phase + ".input.bf16", xb)
        store(phase + ".official.bf16", expected)
        x32, w32 = raw(xf), raw(wf)
        xp, wp = upload(x32), upload(w32)
        metadata = [len(shape), len(shape), 1, 0, 0, 0, 0, 0, 0] + shape + shape + weight_shape
        prod = Args(); prod.inputs[0] = xp; prod.inputs[1] = wp; prod.elements = rows * width
        prod.scalars[0] = 1e-6; prod.compute_dtype = 1; prod.output_dtype = 1; prod.integers[0] = width
        prod.metadata = upload(struct.pack("<" + "Q" * len(metadata), *metadata))
        prod_buf = buffer(rows * width * 4); prod.output = prod_buf[0]
        launch("tensor_f32", "et_rms_norm", prod)
        production = checked_bytes(prod_buf)
        production_bf16 = convert(production, 1, 3, rows * width, 2)
        production_readback = convert(production_bf16, 3, 1, rows * width, 4)
        native27 = (effect / (phase + ".inputNorm.f32")).read_bytes()
        if fixed:
            assert production == reference_raw["weighted"], phase + " fixed production F32"
            assert production_bf16 == expected, phase + " fixed production BF16"
        else:
            assert production_readback == native27, phase + " production must reproduce 27/28"
        store(phase + ".native.weighted.f32", production)
        store(phase + ".native.bf16", production_bf16)
        case = dict(phase=phase, shape=shape, weightName=weight_name, eps=1e-6,
                    productionExact27=production_readback == native27, torchExactCapture=True,
                    fixedProduction=fixed, productionF32=compare_f32(production, reference_raw["weighted"]),
                    inputSha256=hashlib.sha256(xb).hexdigest(),
                    productionComparison=compare(production_bf16, expected), variants=[])
        report["cases"].append(case); persist()
        labels = ["sequential-div-sqrt", "sequential-mul-rsqrt", "torch-mean-mul-rsqrt",
                  "torch-sum-mul-factor-rsqrt", "torch-eps-mul-rsqrt", "torch-inverse-mul",
                  "warp-four-partials-mul-rsqrt", "torch-mean-div-sqrt"]
        for mode, label in enumerate(labels):
            a = Args(); a.inputs[0] = xp; a.inputs[1] = wp
            a.inputs[2] = upload(reference_raw["mean"]); a.inputs[3] = upload(reference_raw["sum"])
            a.inputs[4] = upload(reference_raw["eps"]); a.inputs[5] = upload(reference_raw["inv"])
            a.elements = rows; a.integers[0] = width; a.scalars[0] = 1e-6; a.operation = mode
            stats_buf, norm_buf, out_buf = buffer(rows * 8 * 4), buffer(rows * width * 4), buffer(rows * width * 4)
            a.scratch[0] = stats_buf[0]; a.scratch[1] = norm_buf[0]; a.output = out_buf[0]
            launch("rms_stages", "et_rms_warp" if mode == 6 else "et_rms_stages", a)
            stats = checked_bytes(stats_buf); norm = checked_bytes(norm_buf); values = checked_bytes(out_buf)
            if mode == (6 if fixed else 0): assert values == production
            narrowed = convert(values, 1, 3, rows * width, 2)
            if mode == 0: assert convert(narrowed, 3, 1, rows * width, 4) == native27
            store(phase + "." + label + ".stats.f32", stats)
            store(phase + "." + label + ".normalized.f32", norm)
            store(phase + "." + label + ".weighted.f32", values)
            store(phase + "." + label + ".bf16", narrowed)
            stat_values = f32(stats)
            stages = {}
            for column, stage in [(0,"sum"),(3,"mean"),(4,"eps"),(5,"inv")]:
                data = struct.pack("<" + "f" * rows, *(stat_values[r * 8 + column] for r in range(rows)))
                stages[stage] = compare_f32(data, reference_raw[stage])
            stages["normalized"] = compare_f32(norm, reference_raw["normalized"])
            stages["weighted"] = compare_f32(values, reference_raw["weighted"])
            comparison = compare(narrowed, expected)
            case["variants"].append(dict(label=label, comparison=comparison, stages=stages))
            print(phase, label, "exact", comparison["exact"], "/", rows * width, "stageExact", {k:v["exact"] for k,v in stages.items()}, flush=True)
            persist()
        assert read(xp, len(x32)) == x32 and read(wp, len(w32)) == w32
        case["guardsAndInputsUnchanged"] = True
        del x, xf, squares, sum_ref, mean, eps_added, inverse, normalized, weighted, official, reference
        torch.cuda.synchronize()
    if "--rms-synthetic" in sys.argv:
        report["synthetic"] = []
        def seeded(count, seed):
            data = []
            for i in range(count):
                h = (i + seed) & 0xffffffff
                h = ((h ^ (h >> 16)) * 0x7feb352d) & 0xffffffff
                h = ((h ^ (h >> 15)) * 0x846ca68b) & 0xffffffff
                h ^= h >> 16
                data.append(((h >> 16) & 0x8000) | ((((h >> 24) % 9) + 120) << 7) | ((h >> 8) & 127))
            return torch.tensor(data, dtype=torch.uint16).view(torch.bfloat16).cuda()
        geometries = [(1,1),(3,7),(7,31),(2,32),(19,33),(1,127),(7,128),(16,129),
                      (33,255),(19,256),(3,257),(1,511),(17,512),(7,513),(3,1023),
                      (33,2048),(32,2816),(17,4097),(2,8193),(256,512),
                      (256,2816),(1,256),(1,512),(4096,256),(2048,512)]
        for case_index, (rows, width) in enumerate(geometries):
            for weighted_case in [False, True]:
                x = seeded(rows * width, 17).float().reshape(rows, width)
                if case_index % 5 == 0: x[0] = 0
                if rows > 2:
                    x[1] = x[1] * 1e-37
                    x[2] = x[2] * 1e19
                w = seeded(width, 29).float() if weighted_case else None
                reference = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
                if w is not None: reference = reference * w
                expected = raw(reference)
                a = Args(); a.inputs[0] = upload(raw(x)); a.inputs[1] = upload(raw(w)) if w is not None else 0
                a.elements = rows; a.integers[0] = width; a.scalars[0] = 1e-6
                stats_buf, norm_buf, out_buf = buffer(rows*8*4), buffer(rows*width*4), buffer(rows*width*4)
                a.scratch[0] = stats_buf[0]; a.scratch[1] = norm_buf[0]; a.output = out_buf[0]
                launch("rms_stages", "et_rms_warp", a)
                actual = checked_bytes(out_buf); checked_bytes(stats_buf); checked_bytes(norm_buf)
                av, ev = f32(actual), f32(expected)
                assert all(abs(v - e) <= max(2e-6, abs(e) * 2e-6) for v,e in zip(av,ev)), (rows,width,weighted_case)
                actual_bf16 = convert(actual, 1, 3, rows*width, 2)
                expected_bf16 = raw(reference.to(torch.bfloat16))
                cmp = compare(actual_bf16, expected_bf16)
                assert cmp["beyondAllowance"] == 0, (rows,width,weighted_case,cmp)
                report["synthetic"].append(dict(rows=rows,width=width,weighted=weighted_case, f32=compare_f32(actual,expected), bf16=cmp))
                if rows == 32 and width == 2816 and weighted_case:
                    # Independent fixed Torch fixture, then demonstrate old production failures.
                    meta = [2,2,1,0,0,0,0,0,0,rows,width,rows,width,width]
                    prod = Args(); prod.inputs[0] = a.inputs[0]; prod.inputs[1] = a.inputs[1]
                    prod.elements = rows*width; prod.scalars[0] = 1e-6; prod.compute_dtype = 1; prod.output_dtype = 1
                    prod.metadata = upload(struct.pack("<"+"Q"*len(meta),*meta))
                    prod_buf = buffer(rows*width*4); prod.output = prod_buf[0]
                    launch("tensor_f32", "et_rms_norm", prod)
                    old = convert(checked_bytes(prod_buf),1,3,rows*width,2)
                    old_bits = [b[0] for b in struct.iter_unpack("<H",old)]
                    expected_bits = [b[0] for b in struct.iter_unpack("<H",expected_bf16)]
                    mismatches = [i for i,(ab,eb) in enumerate(zip(old_bits,expected_bits)) if ab!=eb and i//width>2]
                    selected = mismatches[:32]
                    assert selected, "fixed regression must distinguish old production"
                    assert all(actual_bf16[i*2:i*2+2] == expected_bf16[i*2:i*2+2] for i in selected)
                    fixture = dict(rows=rows,width=width,inputSeed=17,weightSeed=29,
                                   selected=[dict(index=i, expected=expected_bits[i], old=old_bits[i]) for i in selected])
                    (output/"independent-fixture.json").write_text(json.dumps(fixture,indent=2)+"\n")
                    store("synthetic.expected.bf16",expected_bf16);store("synthetic.old.bf16",old);store("synthetic.candidate.bf16",actual_bf16)
                persist()
        print("synthetic cases passed",len(report["synthetic"]),flush=True)
    if "--rms-perf-old-source" in sys.argv:
        assert fixed
        old_source = option("--rms-perf-old-source").read_text()
        store("old-compute.cu", old_source.encode())
        modules["rms_old"] = compile_module("rms_old", "\n".join([header,prelude,text("common.cuh"),"#define ET_TENSOR",text("tensor.cu"),old_source]))
        report["performance"] = []
        for rows,width in [(16,2816),(278,2816),(256,2816),(4096,256),(2048,512)]:
            data = torch.arange(rows*width,device="cuda",dtype=torch.float32).remainder(251).sub(125).div(32)
            a = Args(); a.inputs[0] = upload(raw(data)); a.inputs[1] = upload(raw(torch.ones(width,device="cuda")))
            a.elements = rows*width; a.integers[0] = width; a.scalars[0] = 1e-6
            meta = [2,2,1,0,0,0,0,0,0,rows,width,rows,width,width]
            a.metadata = upload(struct.pack("<"+"Q"*len(meta),*meta)); out = buffer(rows*width*4); a.output = out[0]
            timing = dict(rows=rows,width=width,extraWorkspaceBytes=0,method="CUDA events, 2 warmups, 10 launches, isolated F32 opmath kernel")
            for module in ["rms_old","tensor_f32"]:
                function = c.c_void_p(); checked(driver.cuModuleGetFunction(c.byref(function),modules[module],b"et_rms_norm"))
                params = (c.c_void_p*1)(c.addressof(a))
                blocks = (a.elements+255)//256 if module == "rms_old" else min(65535,(rows+7)//8)
                def submit():
                    checked(driver.cuLaunchKernel(function,blocks,1,1,256,1,1,0,None,params,None))
                for _ in range(2): submit()
                checked(driver.cuCtxSynchronize())
                start,end = c.c_void_p(),c.c_void_p()
                checked(driver.cuEventCreate(c.byref(start),0)); checked(driver.cuEventCreate(c.byref(end),0))
                try:
                    checked(driver.cuEventRecord(start,None))
                    for _ in range(10): submit()
                    checked(driver.cuEventRecord(end,None)); checked(driver.cuEventSynchronize(end))
                    elapsed = c.c_float(); checked(driver.cuEventElapsedTime(c.byref(elapsed),start,end))
                    timing[module+"Milliseconds"] = elapsed.value / 10
                finally:
                    checked(driver.cuEventDestroy_v2(start)); checked(driver.cuEventDestroy_v2(end))
                checked_bytes(out)
            report["performance"].append(timing); print(timing,flush=True); persist()
    report["status"] = "completed"
except BaseException as error:
    report["status"] = "failed"; report["error"] = repr(error)
    raise
finally:
    persist()
