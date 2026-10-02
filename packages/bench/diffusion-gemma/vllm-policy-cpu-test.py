"""CPU proof of prior-history patch, without importing vLLM or using CUDA."""
import argparse
import ast
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("baseline")
parser.add_argument("patched")
parser.add_argument("reference")
args = parser.parse_args()

def extract(path, name, globals_):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]

baseline = extract(args.baseline, "_compiled_sample_step", {"torch": torch})
patched = extract(args.patched, "_compiled_sample_step", {"torch": torch})
reference = extract(args.reference, "StableAndConfidentStoppingCriteria", {"torch": torch, "DiffusionGemmaAdaptiveStopping": object})

def state(threshold):
    slots, count, canvas, vocab, hidden = 3, 2, 2, 4, 3
    return dict(
        logits=torch.zeros(count * canvas, vocab),
        decode_slots=torch.tensor([0, 2]), decode_idx=torch.tensor([0, 1]), all_slots=torch.tensor([0, 2]),
        valid_canvas_len=torch.tensor([canvas, canvas]),
        canvas=torch.zeros(slots, canvas, dtype=torch.int64),
        argmax_canvas=torch.zeros(slots, canvas, dtype=torch.int64),
        step_tensor=torch.zeros(slots, dtype=torch.int32),
        is_encoder_phase=torch.zeros(slots, dtype=torch.bool),
        confident_tensor=torch.zeros(slots, dtype=torch.bool),
        sc_embeds=torch.zeros(slots, canvas, hidden),
        embed_weight=torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden) / 12,
        normalizer=torch.tensor(1.0), history=torch.zeros(slots, threshold, canvas, dtype=torch.int64),
        history_len_tensor=torch.zeros(slots, dtype=torch.int32),
        sampled=torch.zeros(count, canvas, dtype=torch.int64), num_sampled=torch.zeros(count, dtype=torch.int32),
        draft_tokens=torch.zeros(slots, canvas, dtype=torch.int64),
        max_denoising_steps=48.0, t_min=0.4, t_max=0.8, confidence_threshold=0.005,
        vocab_size=vocab, CL=canvas, ST=threshold, entropy_bound=0.1,
    )

sequence = [
    [[0, 0], [1, 1]], [[0, 0], [1, 1]], [[1, 1], [1, 1]],
    [[1, 1], [0, 0]], [[1, 1], [0, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 0]],
]
reports = []
for threshold in range(4):
    current = state(threshold)
    shifted = state(threshold + 1)
    original = state(threshold) if threshold else None
    criterion = reference(threshold, 0.005)
    observations = []
    for index, tokens in enumerate(sequence + sequence):
        if index == len(sequence):
            # Reset availability without clearing stored canvas history, as vLLM does.
            for data in [current, shifted] + ([original] if original is not None else []):
                data["history_len_tensor"].zero_()
                data["step_tensor"].zero_()
            criterion.reset()
        target = torch.tensor(tokens)
        logits = torch.full((2, 2, 4), -100.0).scatter_(2, target.unsqueeze(-1), 100.0)
        outputs = []
        rng = []
        for function, data in [(patched, current), (baseline, shifted)] + ([(baseline, original)] if original is not None else []):
            data["logits"] = logits.reshape(4, 4)
            data["is_encoder_phase"].zero_()  # Observe each refinement predicate independently.
            torch.manual_seed(923 + index)
            outputs.append(function(**data))
            rng.append(torch.get_rng_state().clone())
        expected = criterion(target, logits)
        actual = current["is_encoder_phase"][[0, 2]]
        assert torch.equal(actual, expected), (threshold, index, actual, expected)
        assert torch.equal(actual, shifted["is_encoder_phase"][[0, 2]]), ("threshold shift", threshold, index)
        assert all(torch.equal(rng[0], value) for value in rng[1:]), "RNG draws changed"
        assert all(torch.equal(outputs[0], value) for value in outputs[1:]), "scaled logits changed"
        for key in ["canvas", "argmax_canvas", "sc_embeds", "draft_tokens", "confident_tensor"]:
            assert torch.equal(current[key], shifted[key]), ("shifted full-sampler mismatch", threshold, index, key)
        observations.append(actual.tolist())
    reports.append({"officialThreshold": threshold, "legacyEquivalentThreshold": threshold + 1,
                    "predicates": observations, "rngStateExact": True,
                    "scaledLogitsExact": True, "resetChecked": True})
print(json.dumps({"device": "cpu", "status": "passed", "cases": reports}, indent=2))
