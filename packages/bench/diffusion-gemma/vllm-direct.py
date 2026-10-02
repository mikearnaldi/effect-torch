"""Run the pinned DiffusionGemma matrix through the vLLM offline API."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    append_jsonl,
    load_manifest,
    numeric_version,
    prompt_cases,
    verify_vllm_distribution,
)


DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = DIRECTORY.parents[2]
MANIFEST_PATH = Path(os.environ.get("MANIFEST", DIRECTORY / "manifest.json"))
MANIFEST = load_manifest(MANIFEST_PATH)
OUTPUT = Path(
    os.environ.get(
        "OUTPUT",
        REPO_ROOT
        / "bench-results"
        / "diffusion-gemma"
        / f'vllm-direct-{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")}.jsonl',
    )
)
if OUTPUT.exists():
    raise FileExistsError(f"refusing to overwrite {OUTPUT}")


def request_seed(
    target: int, output_tokens: int, concurrency: int, run: int, request: int
) -> int:
    # Fixed replay repeats one trajectory. Distinct mode retains the original
    # per-run seed schedule, including separate warmup requests.
    if os.environ.get("ET_VLLM_REPLAY_DIR") and os.environ.get("ET_VLLM_REPLAY_DISTINCT_SEEDS") != "1":
        run = 0
    return (
        MANIFEST["generation"]["seed"]
        + target * 101
        + output_tokens * 17
        + concurrency * 13
        + run * 7
        + request
    ) & 0xFFFFFFFF



def validate_replay_schedule(replay_metadata: dict, prepared: list[dict[str, Any]]) -> None:
    """Reject missing/mismatched warmups as well as measured replay cases.

    The overlay intentionally supports non-replay requests, so a missing warmup
    seed otherwise falls through to natural generation without an error.
    """
    matrix = MANIFEST["matrix"]
    if matrix["concurrencies"] != [1]:
        raise ValueError("controlled replay requires concurrency one")
    runs = [-(warmup + 1) for warmup in range(matrix["warmupRuns"])]
    runs.extend(range(matrix["measuredRuns"]))
    for target in matrix["promptTargets"]:
        prompt = next((item for item in prepared if item["targetTokens"] == target), None)
        if prompt is None:
            raise ValueError(f"controlled replay has no prompt fixture for target {target}")
        for output_tokens in matrix["outputTokens"]:
            for run in runs:
                seed = request_seed(target, output_tokens, 1, run, 0)
                record = replay_metadata.get(seed)
                if record is None:
                    raise ValueError(f"missing controlled replay case for target {target}, run {run}, seed {seed}")
                request = record[0]["request"]
                if (request["seed"] != seed or request["promptTokenIds"] != prompt["ids"]
                        or request["maxNewTokens"] != output_tokens):
                    raise ValueError(f"controlled replay request differs from fixture for target {target}, run {run}")



def validate_replay_execution(outputs, replay_record, counter_directory) -> None:
    """Validate actual worker calls for warmups and timed requests alike."""
    if not counter_directory:
        raise ValueError("controlled replay requires CPU invocation counters")
    if len(outputs) != 1:
        raise ValueError("controlled replay requires one output")
    output = outputs[0]
    document = replay_record[0]
    if list(output.outputs[0].token_ids) != document["outputTokenIds"]:
        raise RuntimeError("controlled replay emitted tokens differ from its canonical output")
    if output.outputs[0].finish_reason != "length":
        raise RuntimeError("controlled replay counter audit requires one length-terminated block")
    directory = Path(counter_directory)
    known_ids = {json.loads(line)["internalRequestId"] for line in
                 (directory / "requests.jsonl").read_text().splitlines()}
    candidates = [rid for rid in known_ids if rid == output.request_id or
                  re.fullmatch(re.escape(output.request_id) + r"-[0-9a-f]{8}", rid)]
    if len(candidates) != 1:
        raise RuntimeError("cannot uniquely identify controlled replay worker request")
    counts = struct.unpack("<QQQ", (directory / hashlib.sha256(candidates[0].encode()).hexdigest()).read_bytes())
    reads = len(document["steps"])
    if counts != (reads + 1, 1, reads):
        raise RuntimeError(f"controlled replay actual (decode,prefill,sampler) calls {counts} differ from {(reads + 1, 1, reads)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_assets(model: dict[str, Any]) -> None:
    from huggingface_hub import hf_hub_download

    def download(filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=model["id"],
                filename=filename,
                revision=model["revision"],
            )
        )

    tokenizer = download("tokenizer.json")
    template = download("chat_template.jinja")
    if sha256(tokenizer) != model["tokenizerSha256"]:
        raise RuntimeError("pinned tokenizer SHA-256 mismatch")
    if sha256(template) != model["chatTemplateSha256"]:
        raise RuntimeError("pinned chat template SHA-256 mismatch")

    with download("config.json").open(encoding="utf-8") as source:
        model_config = json.load(source)
    with download("generation_config.json").open(encoding="utf-8") as source:
        generation_config = json.load(source)

    generation = MANIFEST["generation"]
    expected = {
        "max_new_tokens": generation["maxNewTokens"],
        "max_denoising_steps": generation["maxSteps"],
        "t_min": generation["minTemperature"],
        "t_max": generation["maxTemperature"],
        "stability_threshold": generation["stabilityThreshold"],
        "confidence_threshold": generation["confidenceThreshold"],
        "eos_token_id": generation["eosTokenIds"],
        "pad_token_id": generation["padTokenId"],
    }
    for key, value in expected.items():
        if generation_config.get(key) != value:
            raise RuntimeError(f"generation_config.json differs at {key}")
    sampler = generation_config.get("sampler_config") or {}
    if sampler.get("_cls_name") != "EntropyBoundSamplerConfig":
        raise RuntimeError("generation_config.json does not select entropy-bound sampling")
    if sampler.get("entropy_bound") != generation["entropyBound"]:
        raise RuntimeError("generation_config.json entropy bound differs")
    if model_config.get("canvas_length") != generation["canvasLength"]:
        raise RuntimeError("config.json canvas length differs")
    if model_config.get("dtype") != model["precision"]:
        raise RuntimeError("config.json dtype differs")


def main() -> None:
    attestation_path = os.environ.get("ET_VLLM_OVERLAY_ATTESTATION")
    if attestation_path and Path(attestation_path).exists():
        raise FileExistsError("refusing stale overlay import attestation")
    replay_directory = os.environ.get("ET_VLLM_REPLAY_DIR")
    replay_metadata = {}
    if replay_directory:
        replay_path = Path(replay_directory)
        paths = [replay_path] if replay_path.is_file() else sorted(replay_path.glob("**/manifest.json"))
        for path in paths:
            document = json.loads(path.read_text())
            seed = document["request"]["seed"]
            if seed in replay_metadata:
                raise ValueError("duplicate controlled replay seed")
            replay_metadata[seed] = (document, sha256(path))
    counter_directory = os.environ.get("ET_VLLM_DECODE_COUNTERS")
    if counter_directory:
        Path(counter_directory).mkdir(parents=True, exist_ok=False)
    vllm_version = importlib.metadata.version("vllm")
    minimum_version = MANIFEST["target"]["minimumVersion"]
    if numeric_version(vllm_version) < numeric_version(minimum_version):
        raise RuntimeError(f"vLLM {vllm_version} is older than {minimum_version}")
    distribution = verify_vllm_distribution(MANIFEST)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import torch

    visible_gpus = torch.cuda.device_count()
    if visible_gpus != MANIFEST["hardware"]["gpuCount"]:
        raise RuntimeError(f"expected one visible GPU, got {visible_gpus}")
    gpu = torch.cuda.get_device_properties(0)
    expected_gpu = MANIFEST["hardware"]["gpu"]
    if expected_gpu not in gpu.name:
        raise RuntimeError(f"expected {expected_gpu}, got {gpu.name}")
    required_memory = MANIFEST["hardware"]["minimumMemoryBytes"]
    if gpu.total_memory < required_memory:
        raise RuntimeError(
            f"GPU memory {gpu.total_memory} is below the required {required_memory}"
        )

    model = MANIFEST["model"]
    verify_pinned_assets(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"]
    )
    prepared: list[dict[str, Any]] = []
    for prompt in prompt_cases(MANIFEST):
        encoded = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=MANIFEST["generation"]["enableThinking"],
        )
        ids = encoded if isinstance(encoded, list) else encoded["input_ids"]
        if len(ids) != prompt["targetTokens"]:
            raise RuntimeError(
                f'prompt {prompt["id"]} encoded to {len(ids)} tokens, '
                f'expected {prompt["targetTokens"]}'
            )
        prepared.append({**prompt, "ids": list(ids)})

    if replay_directory:
        validate_replay_schedule(replay_metadata, prepared)

    load_started = time.perf_counter()
    llm = LLM(
        model=model["id"],
        revision=model["revision"],
        dtype="bfloat16",
        max_model_len=MANIFEST["deployment"]["maxTokens"],
        max_num_seqs=max(MANIFEST["matrix"]["concurrencies"]),
        gpu_memory_utilization=MANIFEST["deployment"]["gpuMemoryUtilization"],
        diffusion_config={
            "canvas_length": MANIFEST["generation"]["canvasLength"],
            "max_denoising_steps": MANIFEST["generation"]["maxSteps"],
        },
        enable_chunked_prefill=True,
        **({"enable_prefix_caching": False, "async_scheduling": False} if replay_directory else {}),
        seed=MANIFEST["generation"]["seed"],
    )
    load_milliseconds = (time.perf_counter() - load_started) * 1000

    for target in MANIFEST["matrix"]["promptTargets"]:
        candidates = [
            prompt for prompt in prepared if prompt["targetTokens"] == target
        ]
        for output_tokens in MANIFEST["matrix"]["outputTokens"]:
            for concurrency in MANIFEST["matrix"]["concurrencies"]:
                prompts = candidates[:concurrency]
                if len(prompts) != concurrency:
                    raise RuntimeError(
                        f"prompt target {target} has {len(prompts)} cases, "
                        f"expected {concurrency}"
                    )

                token_prompts = [
                    {"prompt_token_ids": prompt["ids"]} for prompt in prompts
                ]
                warmup_started = time.perf_counter()
                for warmup in range(MANIFEST["matrix"]["warmupRuns"]):
                    warmup_outputs = llm.generate(
                        token_prompts,
                        [
                            SamplingParams(
                                max_tokens=output_tokens,
                                seed=request_seed(
                                    target,
                                    output_tokens,
                                    concurrency,
                                    -(warmup + 1),
                                    request,
                                ),
                            )
                            for request in range(concurrency)
                        ],
                        use_tqdm=False,
                    )
                    if replay_directory:
                        warmup_record = replay_metadata[request_seed(target, output_tokens, concurrency, -(warmup + 1), 0)]
                        validate_replay_execution(warmup_outputs, warmup_record, counter_directory)
                warmup_milliseconds = (time.perf_counter() - warmup_started) * 1000

                for run in range(MANIFEST["matrix"]["measuredRuns"]):
                    replay_record = None
                    if replay_directory:
                        if concurrency != 1:
                            raise ValueError("controlled replay requires concurrency one")
                        replay_record = replay_metadata[request_seed(target, output_tokens, concurrency, run, 0)]
                        request = replay_record[0]["request"]
                        if request["promptTokenIds"] != prompts[0]["ids"] or request["maxNewTokens"] != output_tokens:
                            raise ValueError("controlled replay request differs from benchmark fixture")
                    timeline_window = os.environ.get("EFFECT_TORCH_CUPTI_WINDOW") == "1"
                    window_start = time.time_ns() if timeline_window else None
                    started = time.perf_counter()
                    outputs = llm.generate(
                        token_prompts,
                        [
                            SamplingParams(
                                max_tokens=output_tokens,
                                seed=request_seed(
                                    target,
                                    output_tokens,
                                    concurrency,
                                    run,
                                    request,
                                ),
                            )
                            for request in range(concurrency)
                        ],
                        use_tqdm=False,
                    )
                    elapsed_milliseconds = (time.perf_counter() - started) * 1000
                    window_end = time.time_ns() if timeline_window else None
                    if replay_record:
                        validate_replay_execution(outputs, replay_record, counter_directory)
                    generated = [len(output.outputs[0].token_ids) for output in outputs]
                    if replay_record and list(outputs[0].outputs[0].token_ids) != replay_record[0]["outputTokenIds"]:
                        raise RuntimeError("controlled replay emitted tokens differ from its canonical output")
                    total_generated = sum(generated)
                    trace_path = os.environ.get("ET_VLLM_STEP_TRACE")
                    steps = {}
                    if trace_path and Path(trace_path).exists():
                        for line in Path(trace_path).read_text().splitlines():
                            step = json.loads(line)
                            counts = steps.setdefault(step["requestId"], [0, 0])
                            counts[0 if step["phase"] == "denoise" else 1] += 1
                    overlay_manifest = os.environ.get("ET_VLLM_OVERLAY_MANIFEST")
                    attestations = ([json.loads(line) for line in Path(attestation_path).read_text().splitlines()]
                        if attestation_path else None)
                    if attestations is not None:
                        expected_overlay = json.loads(Path(overlay_manifest).read_text())
                        if not attestations or any(item["sha256"] != expected_overlay["patchedSha256"] for item in attestations):
                            raise RuntimeError("loaded overlay source differs from declared provenance")
                    known_internal_ids = set(steps)
                    if counter_directory:
                        known_internal_ids.update(json.loads(line)["internalRequestId"] for line in
                            (Path(counter_directory) / "requests.jsonl").read_text().splitlines())
                    internal_ids = []
                    for output in outputs:
                        candidates = [rid for rid in known_internal_ids if rid == output.request_id or
                            re.fullmatch(re.escape(output.request_id) + r"-[0-9a-f]{8}", rid)]
                        if (trace_path or counter_directory) and len(candidates) != 1:
                            raise RuntimeError(f"Cannot uniquely join worker request {output.request_id}: {candidates}")
                        internal_ids.append(candidates[0] if candidates else output.request_id)
                    decode_calls = None
                    prefill_calls = None
                    sampler_calls = None
                    if counter_directory:
                        invocation_counts = [struct.unpack("<QQQ", (
                            Path(counter_directory) / hashlib.sha256(rid.encode()).hexdigest()
                        ).read_bytes()) for rid in internal_ids]
                        decode_calls = [counts[0] for counts in invocation_counts]
                        prefill_calls = [counts[1] for counts in invocation_counts]
                        sampler_calls = [counts[2] for counts in invocation_counts]
                        if replay_record and prefill_calls != [1]:
                            raise RuntimeError("controlled replay requires one fresh full-prompt prefill")
                        # Valid only for the audited full-width, one-block fixture.
                        # Every request must publish exactly one mandatory commit.
                        if not (0 < output_tokens < MANIFEST["generation"]["canvasLength"]
                                and all(n == output_tokens for n in generated)
                                and all(output.outputs[0].finish_reason == "length" for output in outputs)
                                and all(len(prompt["ids"]) + MANIFEST["generation"]["canvasLength"]
                                        <= MANIFEST["deployment"]["maxTokens"] for prompt in prompts)):
                            raise RuntimeError("CPU counter inference requires full-width single-block length termination")
                        for rid, calls in zip(internal_ids, decode_calls):
                            inferred = [calls - 1, 1]
                            if rid in steps and steps[rid] != inferred:
                                raise RuntimeError("CPU decode-call evidence differs from diagnostic phases")
                            steps[rid] = inferred
                        if replay_record and sampler_calls != [len(replay_record[0]["steps"])]:
                            raise RuntimeError("controlled replay sampler count differs from canonical reads")

                    append_jsonl(
                        OUTPUT,
                        {
                            "schemaVersion": MANIFEST["schemaVersion"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "boundary": "direct",
                            "engine": "vllm",
                            "engineVersion": vllm_version,
                            "engineDistribution": distribution,
                            **({
                                "workload": "controlled-trajectory-replay",
                                "replayManifestSha256": replay_record[1],
                                "replayFinalCommit": "included",
                                "replaySeedMode": "distinct" if os.environ.get("ET_VLLM_REPLAY_DISTINCT_SEEDS") == "1" else "fixed",
                                "encoderCommitsPerRequest": [1],
                                "repetitionCooldownMilliseconds": 0,
                                "modelReadInvocationsPerRequest": [calls - 1 for calls in decode_calls],
                                "prefixCache": "disabled",
                                "scheduling": "synchronous",
                                "replayOutput": "forced-canonical",
                                "terminalCommitWork": "encoder-and-kv-only",
                            } if replay_record else {}),
                            "pythonVersion": platform.python_version(),
                            "gpu": {
                                "name": gpu.name,
                                "totalMemoryBytes": gpu.total_memory,
                            },
                            "model": model,
                            "generation": MANIFEST["generation"],
                            "deployment": MANIFEST["deployment"],
                            "targetPromptTokens": target,
                            "actualPromptTokens": [
                                len(prompt["ids"]) for prompt in prompts
                            ],
                            "promptIds": [prompt["id"] for prompt in prompts],
                            "promptTokenIds": [prompt["ids"] for prompt in prompts],
                            "requestIds": [output.request_id for output in outputs],
                            "internalRequestIds": internal_ids,
                            "requestSeeds": [request_seed(target, output_tokens, concurrency, run, request)
                                             for request in range(concurrency)],
                            "generatedTokenIds": [list(output.outputs[0].token_ids) for output in outputs],
                            "refinementsPerRequest": [steps[rid][0] for rid in internal_ids]
                                if all(rid in steps for rid in internal_ids) else None,
                            "blocksPerRequest": [steps[rid][1] for rid in internal_ids]
                                if all(rid in steps for rid in internal_ids) else None,
                            "diagnosticReadbacks": bool(trace_path),
                            "decodeCallsPerRequest": decode_calls,
                            "prefillInvocationsPerRequest": prefill_calls,
                            "samplerInvocationsPerRequest": sampler_calls,
                            "countEvidence": "CPU decode calls minus audited single commit" if decode_calls else
                                ("diagnostic GPU phase readbacks" if trace_path else None),
                            "measurementMode": "diagnostic-profile" if timeline_window else
                                ("generation-diagnostic" if trace_path else "timing"),
                            **({"measurementWindowWallNs": [str(window_start), str(window_end)]}
                                if timeline_window else {}),
                            "experimentalOverlay": json.loads(Path(overlay_manifest).read_text())
                                if overlay_manifest else None,
                            "overlayImportAttestations": attestations,
                            "promptContentSha256": [
                                prompt["contentSha256"] for prompt in prompts
                            ],
                            "requestedOutputTokens": output_tokens,
                            "concurrency": concurrency,
                            "run": run,
                            "generatedTokens": total_generated,
                            "generatedTokensPerRequest": generated,
                            "elapsedMilliseconds": elapsed_milliseconds,
                            "aggregateTokensPerSecond": total_generated
                            * 1000
                            / elapsed_milliseconds,
                            "meanGeneratedTokens": statistics.mean(generated),
                            "finishReasons": [
                                output.outputs[0].finish_reason for output in outputs
                            ],
                            "loadMilliseconds": load_milliseconds,
                            "warmupMilliseconds": warmup_milliseconds,
                        },
                    )

                cooldown = MANIFEST["matrix"]["cooldownMilliseconds"] / 1000
                if cooldown > 0:
                    time.sleep(cooldown)

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
