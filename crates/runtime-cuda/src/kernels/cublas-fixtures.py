"""Regenerate the dense samples in cublas_tests.rs with pinned PyTorch.

Run in PyTorch 2.10.0+cu128 with CUBLAS_WORKSPACE_CONFIG=:4096:8. The
BF16 reduction flag is recorded and checked, never changed by this script.
No model checkpoint is needed.
"""
import hashlib
import json
import os
import torch

assert torch.__version__ == "2.10.0+cu128"
assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

def data(size, seed):
    h = (torch.arange(size, dtype=torch.int64) + seed) & 0xffffffff
    h = ((h ^ (h >> 16)) * 0x7feb352d) & 0xffffffff
    h = ((h ^ (h >> 15)) * 0x846ca68b) & 0xffffffff
    h ^= h >> 16
    bits = (((h >> 16) & 0x8000) | ((((h >> 24) % 9) + 120) << 7) | ((h >> 8) & 127)).to(torch.uint16)
    return bits.view(torch.bfloat16)

report = {
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "workspaceConfig": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    "allowBf16ReducedPrecisionReduction": str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
    "cases": [],
}
for m, n, k in [(16, 128, 2816), (278, 2816, 2112)]:
    x = data(m * k, 17).reshape(m, k)
    w = data(n * k, 29).reshape(n, k)
    with torch.no_grad():
        out = torch.nn.functional.linear(x.cuda(), w.cuda()).cpu()
    bits = out.view(torch.uint16).flatten()
    raw = bits.numpy().tobytes()
    samples = [[(i * 104729) % (m * n), bits[(i * 104729) % (m * n)].item()] for i in range(64)]
    report["cases"].append({"m": m, "n": n, "k": k, "sha256": hashlib.sha256(raw).hexdigest(), "samples": samples})
print(json.dumps(report, indent=2))
