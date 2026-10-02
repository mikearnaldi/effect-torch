"""Prepare an isolated, reversible vLLM prior-history-policy import overlay.

No installed package is modified and no torch/CUDA module is imported. Prefix
PYTHONPATH with the resulting directory for the experimental process only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_prior_history(source: str) -> str:
    before = '''    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )
'''
    after = '''    hist_len = history_len_tensor[decode_slots]
    # Official policy: compare the current argmax with prior history BEFORE
    # replacing its oldest entry. Availability guards the initially empty and
    # reset histories, equivalent to the reference's -1 sentinel for token IDs.
    if ST == 0:
        stable = torch.ones(num_decode, device=device, dtype=torch.bool)
    else:
        stable = hist_len >= ST
        for h in range(ST):
            stable = stable & (history[decode_slots, h] == argmax_tokens).all(dim=-1)
        write_pos = hist_len % ST
        for i in range(ST):
            write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
            history[decode_slots, i] = torch.where(
                write_here, argmax_tokens, history[decode_slots, i]
            )
'''
    old_check = '''    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0
'''
    if source.count(before) != 1 or source.count(old_check) != 1:
        raise ValueError("Installed vLLM policy structure differs from audited source")
    patched = source.replace(before, after).replace(
        old_check, "    # `stable` above compares against the pre-update history.\n")
    # Explicitly ensure the patch did not rewrite, add, or remove RNG calls.
    def random_calls(text: str):
        return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"rand_like", "randint", "rand", "manual_seed"}]
    if random_calls(source) != random_calls(patched):
        raise AssertionError("Policy patch unexpectedly changed RNG calls")
    compile(patched, "policy-matched-diffusion_gemma.py", "exec")
    return patched


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one audited source fragment: {old[:100]!r}")
    return source.replace(old, new)


def patch_shared_rng(source):
    replacements = [
        ('import torch\n', 'import torch\nimport shared_rng\nshared_rng.install_gpu()\n'),
        ('    # Scalar config\n', '    rng_seeds: torch.Tensor,\n    rng_draws: torch.Tensor,\n    canvas_noise: torch.Tensor,\n    # Scalar config\n'),
        ('    u = torch.rand_like(scaled).clamp(min=1e-20)',
         '    u = shared_rng.uniform_like(scaled, rng_seeds[decode_slots], rng_draws[decode_slots])'),
        ('    is_denoise = ~is_commit\n',
         '    is_denoise = ~is_commit\n    rng_draws[decode_slots] = rng_draws[decode_slots] + is_denoise.to(torch.int64)\n'),
        ('''    random_tokens = torch.randint(
        0, vocab_size, (num_decode, CL), device=device, dtype=canvas.dtype
    )''', '    random_tokens = canvas_noise'),
        ('        self.device = device\n\n        self.is_encoder_phase',
         '''        self.device = device
        self.rng_seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.rng_draws = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self._canvas_streams = {}
        self._initial_canvases = {}

        self.is_encoder_phase'''),
        ('''    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
        """Initialize canvas with random tokens for the given slots."""
        n = slot_indices_np.shape[0]
        self.canvas[slot_indices_np] = torch.randint(
            0,
            self.vocab_size,
            (n, self.canvas_length),
            dtype=torch.int64,
            device=self.device,
        )''', '''    def set_random_seed(self, slot, seed):
        self.rng_seeds[slot] = seed
        self.rng_draws[slot] = 0
        stream = shared_rng.CanvasStream(seed)
        self._canvas_streams[slot] = stream
        self._initial_canvases[slot] = stream.canvas(self.canvas_length, self.vocab_size)

    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
        # vLLM initializes at add_request and after prefill; the first value is
        # unused. Reuse it so this consumes one request-local initial canvas.
        # Engine dummy profiling can bypass add_request entirely. Real requests
        # always reset this slot with their explicit seed before using it.
        for slot in slot_indices_np:
            if int(slot) not in self._initial_canvases:
                self.set_random_seed(int(slot), 0)
        self.canvas[slot_indices_np] = torch.tensor(
            [self._initial_canvases[int(slot)] for slot in slot_indices_np],
            dtype=torch.int64, device=self.device)

    def canvas_noise(self, slots):
        return torch.tensor([
            self._canvas_streams[int(slot)].canvas(self.canvas_length, self.vocab_size)
            for slot in slots
        ], dtype=torch.int64, device=self.device)'''),
        ('self.init_canvas(torch.tensor([slot_idx], device=self.device))',
         'self.init_canvas(np.array([slot_idx]))'),
        ('        self.diffusion_states.add_request(req_index)',
         '''        seed = new_req_data.sampling_params.seed
        if seed is None and not new_req_data.req_id.startswith("_warmup_"):
            raise ValueError("Matched RNG experiment requires explicit request seeds")
        self.diffusion_states.set_random_seed(req_index, 0 if seed is None else seed)
        self.diffusion_states.add_request(req_index)'''),
        ('            # Config\n',
         '''            states.rng_seeds,
            states.rng_draws,
            states.canvas_noise(decode_slots_np),
            # Config
'''),
    ]
    for old, new in replacements:
        source = replace_once(source, old, new)
    compile(source, "shared-rng-overlay.py", "exec")
    return source


def patch_diagnostic(source):
    source = replace_once(source, 'import torch\n', 'import torch\nimport os\nimport json\n')
    marker = '        # --- Logprobs: stash on convergence, return on commit ---'
    diagnostic = '''        # Diagnostic-only synchronous trace. Never use these latencies as results.
        if trace_path := os.environ.get("ET_VLLM_STEP_TRACE"):
            commit_flags = is_committing.detach().cpu().tolist()
            step_values = states.step[decode_slots].detach().cpu().tolist()
            argmax_values = states.argmax_canvas[decode_slots].detach().cpu().tolist()
            with open(trace_path, "a", encoding="utf-8") as trace_file:
                for i, batch_idx in enumerate(decode_indices_np):
                    trace_file.write(json.dumps({
                        "requestId": input_batch.req_ids[int(batch_idx)],
                        "phase": "commit" if commit_flags[i] else "denoise",
                        "deviceStepAfterCall": int(step_values[i]),
                        "argmaxTokens": argmax_values[i],
                    }) + "\\n")

'''
    source = replace_once(source, marker, diagnostic + marker)
    compile(source, "diagnostic-overlay.py", "exec")
    return source


def patch_sampler_math(source):
    source = replace_once(source, '''    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()

    token_entropy = -(probs * log_probs).sum(dim=-1)''', '''    normalized = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
    log_probs = normalized.clamp(min=torch.finfo(torch.float32).min)
    probs = torch.softmax(normalized, dim=-1)

    token_entropy = -(probs * log_probs).sum(dim=-1)''')
    source = replace_once(source, 'torch.sort(token_entropy, dim=-1)',
                          'torch.sort(token_entropy, dim=-1, stable=True)')
    source = replace_once(source, 'torch.cumsum(sorted_ent, dim=-1)',
                          'torch.cumsum(sorted_ent, dim=-1, dtype=torch.float64).float()')
    source = replace_once(source,
        '    soft_embeds = torch.matmul(probs.to(embed_weight.dtype), embed_weight) * normalizer',
        '''    # Official feedback crosses a BF16 processed-logit boundary first.
    sc_probs = torch.softmax(scaled.to(embed_weight.dtype).float(), dim=-1)
    soft_embeds = torch.matmul(sc_probs.to(embed_weight.dtype), embed_weight) * normalizer.to(embed_weight.dtype)''')
    compile(source, "sampler-math-overlay.py", "exec")
    return source


def patch_decode_counters(source):
    source = replace_once(source, 'import torch\n', 'import torch\nimport decode_counters\n')
    source = replace_once(source,
        '        self._req_id_to_index[new_req_data.req_id] = req_index',
        '''        self._req_id_to_index[new_req_data.req_id] = req_index
        decode_counters.add(req_index, new_req_data.req_id)''')
    source = replace_once(source, '        decode_slots_np = slots_np[decode_indices_np]',
        '''        decode_slots_np = slots_np[decode_indices_np]
        decode_counters.sample(decode_slots_np)''')
    source = replace_once(source, '''        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)''',
        '''        if input_batch.num_draft_tokens == 0:
            decode_counters.prefill(input_batch.idx_mapping_np[:num_reqs])
            return self._handle_prefill(input_batch, device)''')
    source = replace_once(source, '        scaled = _compiled_sample_step(\n',
        '        decode_counters.sampler(decode_slots_np)\n        scaled = _compiled_sample_step(\n')
    compile(source, "decode-counter-overlay.py", "exec")
    return source


DECODE_COUNTER_SOURCE = '''"""CPU-only decode-call evidence; no GPU readbacks or per-step file syscalls."""
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct

_slots = {}

def add(slot, request_id):
    previous = _slots.pop(slot, None)
    if previous is not None:
        previous[0].close()
    directory = os.environ.get("ET_VLLM_DECODE_COUNTERS")
    if not directory or request_id.startswith("_warmup_"):
        return
    path = Path(directory) / hashlib.sha256(request_id.encode()).hexdigest()
    with path.open("xb+") as stream:
        stream.truncate(24)
        mapping = mmap.mmap(stream.fileno(), 24)
    _slots[slot] = (mapping, 0)
    with (Path(directory) / "requests.jsonl").open("a") as index:
        index.write(json.dumps({"internalRequestId": request_id}) + "\\n")

def sample(slots):
    for slot in slots:
        entry = _slots.get(int(slot))
        if entry is not None:
            mapping, count = entry
            count += 1
            struct.pack_into("<Q", mapping, 0, count)
            _slots[int(slot)] = (mapping, count)

def prefill(slots):
    for slot in slots:
        entry = _slots.get(int(slot))
        if entry is not None:
            mapping, _ = entry
            count = struct.unpack_from("<Q", mapping, 8)[0]
            struct.pack_into("<Q", mapping, 8, count + 1)

def sampler(slots):
    for slot in slots:
        entry = _slots.get(int(slot))
        if entry is not None:
            mapping, _ = entry
            count = struct.unpack_from("<Q", mapping, 16)[0]
            struct.pack_into("<Q", mapping, 16, count + 1)
'''


def patch_controlled_replay(source):
    source = replace_once(source, 'import torch\n',
        'import torch\nimport os\nimport vllm_controlled_replay\n')
    source = replace_once(source, '    entropy_bound: float,\n) -> torch.Tensor:', '''    entropy_bound: float,
    replay_feedback: torch.Tensor | None = None,
    replay_canvas: torch.Tensor | None = None,
    replay_draft: torch.Tensor | None = None,
    replay_done: bool = False,
    replay_temperature: torch.Tensor | None = None,
    actual_feedback: torch.Tensor | None = None,
    actual_canvas: torch.Tensor | None = None,
    actual_draft: torch.Tensor | None = None,
) -> torch.Tensor:''')
    source = replace_once(source,
        '    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)',
        '''    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)
    if replay_temperature is not None:
        temp = replay_temperature.expand_as(temp)''')
    source = replace_once(source,
        '    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]',
        '''    if replay_canvas is not None:
        # Controlled experiment only: canonical convergence determines reads.
        is_encoder_phase[decode_slots] = replay_done
    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]''')
    source = replace_once(source,
        '    sc_probs = torch.softmax(scaled.to(embed_weight.dtype).float(), dim=-1)',
        '''    native_feedback = scaled.to(embed_weight.dtype)
    if actual_feedback is not None:
        # Observable actual BF16 sampler output, mirroring Effect's feedback
        # production even though replay substitutes the next read's input.
        actual_feedback.copy_(native_feedback)
    sc_input = native_feedback if replay_feedback is None else replay_feedback
    sc_probs = torch.softmax(sc_input.float(), dim=-1)''')
    source = replace_once(source, '''    sc_probs = torch.softmax(sc_input.float(), dim=-1)
    soft_embeds = torch.matmul(sc_probs.to(embed_weight.dtype), embed_weight) * normalizer.to(embed_weight.dtype)
    sc_embeds[decode_slots] = soft_embeds * sc_keep''',
        '''    if replay_canvas is not None and replay_done:
        # The canonical terminal read has no following conditioned read.
        sc_embeds[decode_slots] = 0
    else:
        sc_probs = torch.softmax(sc_input.float(), dim=-1)
        soft_embeds = torch.matmul(sc_probs.to(embed_weight.dtype), embed_weight) * normalizer.to(embed_weight.dtype)
        sc_embeds[decode_slots] = soft_embeds * sc_keep''')
    source = replace_once(source,
        '    # ---- Phase 6: Stability + convergence ----',
        '''    if actual_canvas is not None:
        # Keep the native stochastic sampler result observable before replacing
        # the next input, so compilation cannot delete Gumbel/entropy-mask work.
        actual_canvas.copy_(canvas[decode_slots])
        actual_draft.copy_(argmax_canvas[decode_slots])
    # ---- Phase 6: Stability + convergence ----''')
    source = replace_once(source,
        '    # ---- Phase 7: Copy canvas → draft_tokens for all slots ----',
        '''    if replay_canvas is not None:
        canvas[decode_slots] = replay_canvas
        argmax_canvas[decode_slots] = replay_draft
    # ---- Phase 7: Copy canvas → draft_tokens for all slots ----''')
    source = replace_once(source,
        '        self._initial_canvases = {}',
        '''        self._initial_canvases = {}
        self.replay_bank = vllm_controlled_replay.preload(os.environ["ET_VLLM_REPLAY_DIR"], device)
        self._replay_temperatures = {
            seed: tuple(torch.tensor([step.temperature], dtype=torch.float32, device=device)
                        for step in request.steps)
            for seed, request in self.replay_bank.requests.items()
        }
        self._replay_requests = {}
        self._replay_indices = {}
        self._actual_feedback = torch.empty((1, canvas_length, vocab_size),
                                            dtype=torch.bfloat16, device=device)
        self._actual_canvas = torch.empty((1, canvas_length), dtype=torch.int64, device=device)
        self._actual_draft = torch.empty_like(self._actual_canvas)''')
    source = replace_once(source,
        '    def set_random_seed(self, slot, seed):',
        '''    def set_replay(self, slot, seed, prompt_ids):
        request = self.replay_bank.get_request(seed, prompt_ids)
        self._replay_requests[slot] = request
        self._replay_indices[slot] = 0

    def replay_step(self, slots):
        if len(slots) != 1:
            raise ValueError("Controlled replay currently requires concurrency one")
        slot = int(slots[0])
        result = {"actual_feedback": self._actual_feedback,
                  "actual_canvas": self._actual_canvas, "actual_draft": self._actual_draft}
        request = self._replay_requests.get(slot)
        if request is None:
            return result
        index = self._replay_indices[slot]
        self._replay_indices[slot] = index + 1
        if index == len(request.steps):
            # Mandatory final encoder commit remains ordinary native work.
            return result
        if index > len(request.steps):
            raise ValueError("Controlled replay exceeded the canonical final commit")
        step = request.steps[index]
        result.update(replay_canvas=request.committed_canvas if step.done else step.post_canvas,
                      replay_draft=request.committed_canvas if step.done else step.draft,
                      replay_done=step.done, replay_temperature=self._replay_temperatures[request.seed][index])
        if index + 1 < len(request.steps):
            # vLLM computes SC for the next read inside this sampler. Replace
            # its source once; do not add a second SC softmax/embedding GEMM.
            result["replay_feedback"] = request.steps[index + 1].feedback
        return result

    def set_random_seed(self, slot, seed):''')
    # Both startup init calls must restore the exact same canonical first input.
    source = replace_once(source,
        '            dtype=torch.int64, device=self.device)\n\n    def canvas_noise',
        '''            dtype=torch.int64, device=self.device)
        for slot in slot_indices_np:
            request = self._replay_requests.get(int(slot))
            if request is not None:
                self.canvas[int(slot)] = request.steps[0].canvas

    def canvas_noise''')
    source = replace_once(source,
        '        self.diffusion_states.add_request(req_index)',
        '''        self.diffusion_states.set_replay(req_index, 0 if seed is None else seed,
                                         new_req_data.prompt_token_ids)
        self.diffusion_states.add_request(req_index)''')
    source = replace_once(source,
        '            entropy_bound=self.entropy_bound,\n',
        '            entropy_bound=self.entropy_bound,\n            **states.replay_step(decode_slots_np),\n')
    source = replace_once(source,
        '        for slot, idx in zip(decode_slots_np.tolist(), decode_idx_np.tolist()):\n',
        '''        for slot, idx in zip(decode_slots_np.tolist(), decode_idx_np.tolist()):
            replay_states = self.diffusion_states
            replay_request = replay_states._replay_requests.get(slot)
            if (replay_request is not None and
                    replay_states._replay_indices[slot] == len(replay_request.steps)):
                # Canonical terminal commit is an encoder pass, whose input
                # has no decoder self-conditioning or post-conditioning norm.
                continue
''')
    source = replace_once(source,
        '    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n',
        '''    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if getattr(self, "_replay_encoder_commit", False):
            # Encoder/KV commit has no vocabulary readout. The sampler's
            # publication fast path consumes no logits for this phase.
            return hidden_states.new_empty((0, self.lm_head.weight.shape[0]), dtype=torch.float32)
''')
    source = replace_once(source,
        '''        states = self.diffusion_states
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs''',
        '''        states = self.diffusion_states
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs
        self.model._replay_encoder_commit = False
        if num_reqs == 1 and input_batch.num_draft_tokens > 0:
            slot = int(input_batch.idx_mapping_np[0])
            request = states._replay_requests.get(slot)
            self.model._replay_encoder_commit = (request is not None and
                states._replay_indices[slot] == len(request.steps))''')
    source = replace_once(source,
        '        decode_counters.sample(decode_slots_np)\n',
        '''        decode_counters.sample(decode_slots_np)
        if len(decode_slots_np) == 1:
            slot = int(decode_slots_np[0])
            request = states._replay_requests.get(slot)
            if request is not None and states._replay_indices[slot] == len(request.steps):
                # The full encoder already committed canonical tokens to KV.
                # Publish those tokens through the ordinary scheduler output,
                # without an unused head, denoising sampler, RNG, or SC GEMM.
                if num_reqs != 1 or int(per_req_nlogits_np[0]) != CL:
                    raise ValueError("Replay commit requires one full-width canvas")
                sampled = self._sampled[:num_reqs]
                num_sampled = self._num_sampled[:num_reqs]
                sampled.copy_(states.argmax_canvas[slot:slot + 1].to(sampled.dtype))
                num_sampled.fill_(CL)
                states.is_encoder_phase[slot] = False
                states.step[slot] = 0
                states.accepted_canvas_history_len[slot] = 0
                states.self_conditioning_embeds[slot] = 0
                states._replay_indices[slot] += 1
                return self._build_output(input_batch, sampled, num_sampled,
                                          per_req_nlogits_np, device)
''')
    compile(source, "controlled-replay-overlay.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--shared-rng", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--sampler-math", action="store_true")
    parser.add_argument("--decode-counters", action="store_true")
    parser.add_argument("--controlled-replay", type=Path)
    args = parser.parse_args()
    original = args.source.read_bytes()
    if sha256(original) != args.expected_sha256:
        raise ValueError("Installed vLLM source hash differs from audited baseline")
    patched = patch_prior_history(original.decode())
    if args.shared_rng:
        patched = patch_shared_rng(patched)
    if args.sampler_math:
        patched = patch_sampler_math(patched)
    if args.diagnostic:
        patched = patch_diagnostic(patched)
    if args.decode_counters:
        patched = patch_decode_counters(patched)
    if args.controlled_replay:
        if not args.shared_rng or not args.sampler_math or not args.decode_counters:
            raise ValueError("Controlled replay requires shared RNG, sampler math, and decode counters")
        patched = patch_controlled_replay(patched)
    patched = replace_once(patched, 'import torch\n', '''import torch
import hashlib as _overlay_hashlib
import json as _overlay_json
import os as _overlay_os
if _overlay_attestation := _overlay_os.environ.get("ET_VLLM_OVERLAY_ATTESTATION"):
    with open(__file__, "rb") as _overlay_source:
        _overlay_digest = _overlay_hashlib.sha256(_overlay_source.read()).hexdigest()
    with open(_overlay_attestation, "a") as _overlay_file:
        _overlay_file.write(_overlay_json.dumps({"pid": _overlay_os.getpid(),
            "source": __file__, "sha256": _overlay_digest}) + "\\n")
''')
    compile(patched, "complete-isolated-overlay.py", "exec")
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / "baseline-diffusion_gemma.py").write_bytes(original)
    (args.destination / "diffusion_gemma.py").write_text(patched)
    if args.shared_rng:
        (args.destination / "shared_rng.py").write_bytes(args.shared_rng.read_bytes())
    if args.decode_counters:
        (args.destination / "decode_counters.py").write_text(DECODE_COUNTER_SOURCE)
    if args.controlled_replay:
        (args.destination / "vllm_controlled_replay.py").write_bytes(args.controlled_replay.read_bytes())
    (args.destination / "sitecustomize.py").write_text('''# Experimental process-local import overlay; no package files are modified.
import importlib.abc
import importlib.util
from pathlib import Path
import sys

class _EffectTorchVllmPolicyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm.model_executor.models.diffusion_gemma":
            return importlib.util.spec_from_file_location(
                fullname, Path(__file__).with_name("diffusion_gemma.py"))
        return None

sys.meta_path.insert(0, _EffectTorchVllmPolicyFinder())
''')
    manifest = {
        "kind": "vllm-official-prior-history-policy-overlay",
        "originalSource": str(args.source.resolve()),
        "originalSha256": sha256(original),
        "patchedSha256": sha256(patched.encode()),
        "patchScope": {
            "priorHistory": True,
            "sharedRng": bool(args.shared_rng),
            "samplerMath": args.sampler_math,
            "cpuDecodeCounters": args.decode_counters,
            "controlledReplay": bool(args.controlled_replay),
            "replayCommitSkipsDecoderSelfConditioning": bool(args.controlled_replay),
            "replayCommitSkipsUnusedHeadAndSampler": bool(args.controlled_replay),
        },
        "randomCallsUnchanged": not bool(args.shared_rng),
        "sharedRngSha256": sha256(args.shared_rng.read_bytes()) if args.shared_rng else None,
        "controlledReplayHelperSha256": sha256(args.controlled_replay.read_bytes()) if args.controlled_replay else None,
        "cpuCounterSha256": sha256(DECODE_COUNTER_SOURCE.encode()) if args.decode_counters else None,
        "containsDiagnosticReadbacks": args.diagnostic,
        "installedBaselineModified": False,
        "limitations": ["Forward model and compiled reduction numerics may still differ",
                        "vLLM retains mandatory encoder commit work",
                        "Matching generation requires validating actual tokens and refinements"],
    }
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
