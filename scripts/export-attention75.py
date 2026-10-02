"""Offline native PTX export; requires installed Torch/Triton CUDA environment.

Root schedules this compilation. Does not install packages or run model code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "crates/runtime-cuda/src/kernels"))
from attention75 import attention75_native


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--bf16-io", action="store_true",
                   help="Export ABI2 with both legacy F32 and direct BF16 pointer variants")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.get_device_capability() == (12, 0), "SM120 export only"
    q = torch.empty(1, device="cuda", dtype=torch.float32)
    table = torch.empty(1, device="cuda", dtype=torch.uint64)
    out = torch.empty_like(q)
    status = torch.zeros(1, device="cuda", dtype=torch.uint64)
    counts = torch.empty(1, device="cuda", dtype=torch.uint32)
    cursors = torch.empty_like(counts)
    meta = torch.empty(1, device="cuda", dtype=torch.uint64)
    variants = []
    specifications = [(dim, torch.float32) for dim in [256, 512]]
    if args.bf16_io:
        specifications += [(dim, torch.bfloat16) for dim in [256, 512]]
    for dim, dtype in specifications:
        q = torch.empty(1, device="cuda", dtype=dtype)
        out = torch.empty_like(q)
        # Warmup compiles without launching: these dummy allocations are never read.
        compiled = attention75_native.warmup(q, table, out, status, counts, cursors, meta,
            2, 2, 256, 256, 288, 17, 1.0, 75, 1, 1, dim,
            num_warps=4, num_stages=2, grid=(16, 16))
        metadata = compiled.metadata._asdict()
        for field in ["global_scratch_size", "profile_scratch_size", "tmem_size"]:
            assert metadata.get(field, 0) == 0, f"unsupported {field}"
        assert metadata["num_warps"] == 4 and metadata["num_ctas"] == 1
        assert not metadata["launch_cooperative_grid"] and not metadata["launch_pdl"]
        ptx = compiled.asm["ptx"]
        entry = re.search(r"(?:\.visible\s+)?\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
        assert entry
        types = re.findall(r"\.param\s+\.(\w+)", entry.group(2))
        expected = ["u64"] * 7 + ["u32"] * 6 + ["f32"] + ["u32"] * 3
        assert types[:17] == expected, (types, expected)
        assert types[17:] in [[], ["u64"], ["u64", "u64"]], types
        suffix = "-bf16" if dtype == torch.bfloat16 else ""
        name = f"attention75{suffix}-d{dim}.ptx"
        (args.output / name).write_text(ptx)
        variants.append(dict(dim=dim, ptx=name, entry=entry.group(1),
            q_dtype=3 if dtype == torch.bfloat16 else 1,
            output_dtype=3 if dtype == torch.bfloat16 else 1,
            sha256=hashlib.sha256(ptx.encode()).hexdigest(), shared=metadata["shared"],
            num_warps=4, extra_zero_u64_parameters=len(types)-17,
            literal_parameters=entry.group(2), metadata=metadata))
    source = root / "crates/runtime-cuda/src/kernels/attention75.py"
    manifest = dict(abi=2 if args.bf16_io else 1, compute_capability=[12, 0], variants=variants,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validation="Compilation/ABI export only; native numerical and lifetime tests required")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    print(json.dumps({"variants":len(variants), "directory":str(args.output)}))


if __name__ == "__main__":
    main()
