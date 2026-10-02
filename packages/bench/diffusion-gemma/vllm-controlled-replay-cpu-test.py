"""Execute the real patched sampler AST on CPU to validate replay boundaries."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import torch


def extract(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


state = extract(Path(__file__).with_name("vllm-policy-cpu-test.py"), "state", {"torch": torch})
# RNG numerical bits have an independent exhaustive fixture. This isolates
# canvas/feedback/phase selection from those already validated random values.
rng = SimpleNamespace(uniform_like=lambda template, seeds, draws: torch.full_like(template, 0.5))
sampler = extract(Path(sys.argv[1]) / "diffusion_gemma.py", "_compiled_sample_step",
                  {"torch": torch, "shared_rng": rng})
x = state(1)
x.update(rng_seeds=torch.tensor([7, 0, 9]), rng_draws=torch.zeros(3, dtype=torch.int64),
         canvas_noise=torch.tensor([[1, 2], [2, 3]]),
         replay_canvas=torch.tensor([3, 1]), replay_draft=torch.tensor([2, 0]),
         replay_done=False, replay_temperature=torch.tensor([0.5]),
         replay_feedback=torch.zeros(2, 2, 4), actual_feedback=torch.empty(2, 2, 4),
         actual_canvas=torch.empty(2, 2, dtype=torch.int64),
         actual_draft=torch.empty(2, 2, dtype=torch.int64))
x["logits"] = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 8
sampler(**x)
assert torch.equal(x["actual_feedback"], (x["logits"] / 0.5).reshape(2, 2, 4))
assert x["canvas"][[0, 2]].tolist() == [[3, 1], [3, 1]]
assert x["argmax_canvas"][[0, 2]].tolist() == [[2, 0], [2, 0]]
assert x["actual_draft"].tolist() == [[3, 3], [3, 3]]
assert not torch.equal(x["actual_canvas"], x["canvas"][[0, 2]])
assert x["is_encoder_phase"][[0, 2]].tolist() == [False, False]
expected_soft = torch.full((2, 2, 4), 0.25) @ x["embed_weight"]
assert torch.equal(x["sc_embeds"][[0, 2]], expected_soft)
# Force terminal convergence despite deliberately non-confident actual logits.
x.update(replay_canvas=torch.tensor([2, 0]), replay_done=True)
sampler(**x)
assert x["is_encoder_phase"][[0, 2]].tolist() == [True, True]
assert x["sc_embeds"][[0, 2]].count_nonzero() == 0
# Final encoder commit is still native and publishes the canonical full canvas.
x.update(replay_canvas=None, replay_draft=None, replay_feedback=None, replay_temperature=None)
sampler(**x)
assert x["sampled"].tolist() == [[2, 0], [2, 0]]
assert x["rng_draws"].tolist() == [2, 0, 2]
assert x["num_sampled"].tolist() == [2, 2]
print("Controlled replay CPU sampler proof passed: native feedback observable, canonical SC once, forced termination, native commit")

# Exercise the actual publication dispatcher and model head bypass with small
# CPU state. Any accidental projection/sampler call fails this fixture.
import numpy as np
from typing import Any

def method(class_name, method_name, namespace):
    tree = ast.parse((Path(sys.argv[1]) / "diffusion_gemma.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "replay-method", "exec"), namespace)
    return namespace[method_name]

native_calls = []
counters = SimpleNamespace(sample=lambda slots: native_calls.append(slots.tolist()))
namespace = {"torch": torch, "np": np, "Any": Any, "SamplerOutput": object, "decode_counters": counters}
# Locate the sampler by its method rather than coupling this test to its name.
tree = ast.parse((Path(sys.argv[1]) / "diffusion_gemma.py").read_text())
sampler_class = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
    and any(isinstance(f, ast.FunctionDef) and f.name == "__call__" for f in n.body))
call = method(sampler_class, "__call__", namespace)
states = SimpleNamespace(_replay_requests={0: SimpleNamespace(steps=[0, 1])}, _replay_indices={0: 2},
    argmax_canvas=torch.tensor([[2, 0]]), is_encoder_phase=torch.tensor([True]),
    step=torch.tensor([2]), accepted_canvas_history_len=torch.tensor([2]),
    self_conditioning_embeds=torch.ones(1, 2, 3))
instance = SimpleNamespace(diffusion_states=states, canvas_length=2,
    _sampled=torch.empty(1, 2, dtype=torch.int64), _num_sampled=torch.empty(1, dtype=torch.int32),
    _build_output=lambda batch, sampled, counts, lengths, device: (sampled, counts))
batch = SimpleNamespace(num_reqs=1, num_draft_tokens=2, idx_mapping_np=np.array([0]), cu_num_logits_np=np.array([0, 2]))
result = call(instance, torch.empty(0, 4), batch)
assert result[0].tolist() == [[2, 0]] and result[1].tolist() == [2]
assert states._replay_indices[0] == 3 and native_calls == [[0]]
assert states.is_encoder_phase.tolist() == [False]
head = method("DiffusionGemmaForConditionalGeneration", "compute_logits", {"torch": torch})
model = SimpleNamespace(_replay_encoder_commit=True, lm_head=SimpleNamespace(weight=torch.empty(4, 3)))
assert head(model, torch.empty(2, 3)).shape == (0, 4)
print("Controlled commit CPU proof passed: canonical publication, no head or denoise sampler")
