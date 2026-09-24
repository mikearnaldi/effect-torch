"""Run the pinned DiffusionGemma matrix through the vLLM offline API."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
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
    return (
        MANIFEST["generation"]["seed"]
        + target * 101
        + output_tokens * 17
        + concurrency * 13
        + run * 7
        + request
    ) & 0xFFFFFFFF


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
                    llm.generate(
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
                warmup_milliseconds = (time.perf_counter() - warmup_started) * 1000

                for run in range(MANIFEST["matrix"]["measuredRuns"]):
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
                    generated = [len(output.outputs[0].token_ids) for output in outputs]
                    total_generated = sum(generated)

                    append_jsonl(
                        OUTPUT,
                        {
                            "schemaVersion": MANIFEST["schemaVersion"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "boundary": "direct",
                            "engine": "vllm",
                            "engineVersion": vllm_version,
                            "engineDistribution": distribution,
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
