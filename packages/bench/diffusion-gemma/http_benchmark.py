"""Benchmark an already-running OpenAI-compatible DiffusionGemma server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from common import append_jsonl, load_manifest, prompt_cases, verify_vllm_distribution


DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = DIRECTORY.parents[2]
MANIFEST_PATH = Path(os.environ.get("MANIFEST", DIRECTORY / "manifest.json"))
MANIFEST = load_manifest(MANIFEST_PATH)
ENGINE = os.environ.get("ENGINE", "")
if ENGINE not in {"effect-torch", "vllm"}:
    raise ValueError("ENGINE must be effect-torch or vllm")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "diffusiongemma" if ENGINE == "effect-torch" else MANIFEST["model"]["id"],
)
OUTPUT = Path(
    os.environ.get(
        "OUTPUT",
        REPO_ROOT
        / "bench-results"
        / "diffusion-gemma"
        / f'{ENGINE}-http-{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")}.jsonl',
    )
)
if OUTPUT.exists():
    raise FileExistsError(f"refusing to overwrite {OUTPUT}")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]


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


async def generate(
    session: aiohttp.ClientSession,
    prompt: dict[str, Any],
    output_tokens: int,
    seed: int,
) -> dict[str, Any]:
    body = {
        "model": MODEL_NAME,
        "messages": prompt["messages"],
        "max_tokens": output_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_text_milliseconds: float | None = None
    text: list[str] = []
    page_milliseconds: list[float] = []
    page_utf8_bytes: list[int] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    async with session.post(f"{BASE_URL}/v1/chat/completions", json=body) as response:
        if response.status != 200:
            payload = await response.text()
            raise RuntimeError(f"server returned {response.status}: {payload}")

        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage") is not None:
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                observed_milliseconds = (time.perf_counter() - started) * 1000
                if first_text_milliseconds is None:
                    first_text_milliseconds = observed_milliseconds
                page_milliseconds.append(observed_milliseconds)
                page_utf8_bytes.append(len(content.encode()))
                text.append(content)

    elapsed_milliseconds = (time.perf_counter() - started) * 1000
    if usage is None:
        raise RuntimeError("stream ended without usage")
    generated = "".join(text)
    return {
        "prompt": prompt,
        "promptTokens": usage["prompt_tokens"],
        "generatedTokens": usage["completion_tokens"],
        "elapsedMilliseconds": elapsed_milliseconds,
        "firstTextMilliseconds": first_text_milliseconds or elapsed_milliseconds,
        "pageMilliseconds": page_milliseconds,
        "pageUtf8Bytes": page_utf8_bytes,
        "finishReason": finish_reason,
        "textSha256": hashlib.sha256(generated.encode()).hexdigest(),
    }


async def run_batch(
    session: aiohttp.ClientSession,
    prompts: list[dict[str, Any]],
    output_tokens: int,
    target: int,
    run: int,
) -> tuple[float, list[dict[str, Any]]]:
    started = time.perf_counter()
    requests = await asyncio.gather(
        *(
            generate(
                session,
                prompt,
                output_tokens,
                request_seed(target, output_tokens, len(prompts), run, request),
            )
            for request, prompt in enumerate(prompts)
        )
    )
    return (time.perf_counter() - started) * 1000, requests


async def main() -> None:
    distribution = verify_vllm_distribution(MANIFEST) if ENGINE == "vllm" else None
    timeout = aiohttp.ClientTimeout(total=60 * 60)
    connector = aiohttp.TCPConnector(limit=max(MANIFEST["matrix"]["concurrencies"]))
    cases = prompt_cases(MANIFEST)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(f"{BASE_URL}/v1/models") as response:
            if response.status != 200:
                raise RuntimeError(
                    f"server health check returned {response.status}: {await response.text()}"
                )
            models = await response.json()
            model_ids = {model["id"] for model in models.get("data", [])}
            if MODEL_NAME not in model_ids:
                raise RuntimeError(f"server does not expose {MODEL_NAME}: {sorted(model_ids)}")

        for target in MANIFEST["matrix"]["promptTargets"]:
            candidates = [prompt for prompt in cases if prompt["targetTokens"] == target]
            for output_tokens in MANIFEST["matrix"]["outputTokens"]:
                for concurrency in MANIFEST["matrix"]["concurrencies"]:
                    prompts = candidates[:concurrency]
                    if len(prompts) != concurrency:
                        raise RuntimeError(
                            f"prompt target {target} has {len(prompts)} cases, "
                            f"expected {concurrency}"
                        )

                    warmup_started = time.perf_counter()
                    for warmup in range(MANIFEST["matrix"]["warmupRuns"]):
                        await run_batch(
                            session,
                            prompts,
                            output_tokens,
                            target,
                            -(warmup + 1),
                        )
                    warmup_milliseconds = (time.perf_counter() - warmup_started) * 1000

                    for run in range(MANIFEST["matrix"]["measuredRuns"]):
                        elapsed_milliseconds, requests = await run_batch(
                            session, prompts, output_tokens, target, run
                        )
                        if any(
                            request["promptTokens"] != target for request in requests
                        ):
                            actual = [request["promptTokens"] for request in requests]
                            raise RuntimeError(
                                f"prompt target {target} encoded as {actual}"
                            )
                        generated_tokens = sum(
                            request["generatedTokens"] for request in requests
                        )
                        request_times = [
                            request["elapsedMilliseconds"] for request in requests
                        ]
                        first_text_times = [
                            request["firstTextMilliseconds"] for request in requests
                        ]
                        append_jsonl(
                            OUTPUT,
                            {
                                "schemaVersion": MANIFEST["schemaVersion"],
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "boundary": "http",
                                "engine": ENGINE,
                                **(
                                    {"engineDistribution": distribution}
                                    if distribution is not None
                                    else {}
                                ),
                                "baseUrl": BASE_URL,
                                "modelName": MODEL_NAME,
                                "model": MANIFEST["model"],
                                "generation": MANIFEST["generation"],
                                "deployment": MANIFEST["deployment"],
                                "targetPromptTokens": target,
                                "actualPromptTokens": [
                                    request["promptTokens"] for request in requests
                                ],
                                "promptIds": [prompt["id"] for prompt in prompts],
                                "promptContentSha256": [
                                    prompt["contentSha256"] for prompt in prompts
                                ],
                                "requestedOutputTokens": output_tokens,
                                "concurrency": concurrency,
                                "run": run,
                                "generatedTokens": generated_tokens,
                                "generatedTokensPerRequest": [
                                    request["generatedTokens"] for request in requests
                                ],
                                "elapsedMilliseconds": elapsed_milliseconds,
                                "requestMilliseconds": request_times,
                                "firstTextMilliseconds": first_text_times,
                                "pageMillisecondsPerRequest": [
                                    request["pageMilliseconds"] for request in requests
                                ],
                                "pageUtf8BytesPerRequest": [
                                    request["pageUtf8Bytes"] for request in requests
                                ],
                                "aggregateTokensPerSecond": generated_tokens
                                * 1000
                                / elapsed_milliseconds,
                                "requestP50Milliseconds": statistics.median(request_times),
                                "requestP95Milliseconds": percentile(request_times, 0.95),
                                "requestP99Milliseconds": percentile(request_times, 0.99),
                                "firstTextP50Milliseconds": statistics.median(
                                    first_text_times
                                ),
                                "firstTextP95Milliseconds": percentile(
                                    first_text_times, 0.95
                                ),
                                "firstTextP99Milliseconds": percentile(
                                    first_text_times, 0.99
                                ),
                                "finishReasons": [
                                    request["finishReason"] for request in requests
                                ],
                                "textSha256": [
                                    request["textSha256"] for request in requests
                                ],
                                "warmupMilliseconds": warmup_milliseconds,
                            },
                        )

                    cooldown = MANIFEST["matrix"]["cooldownMilliseconds"] / 1000
                    if cooldown > 0:
                        await asyncio.sleep(cooldown)

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
