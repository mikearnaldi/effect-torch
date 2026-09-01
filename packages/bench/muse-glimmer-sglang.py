"""Benchmark four concurrent Muse-Glimmer sequences through SGLang.

The benchmark launches either the pinned GGUF or RadixArk's pinned mixed
NVFP4/MXFP8 checkpoint, then submits one streaming batch containing four unique
prompts. Prefill is measured as client TTFT, so it includes first-token sampling.
Per-session decode starts after each sequence's first token. Aggregate decode
starts only after all four first tokens arrive. Radix caching is disabled, and
streamed prompt/cache/token counts are asserted. SGLang 0.5.18 expects an
expanded sliding-attention array, so launched servers use
sglang-compat/sitecustomize.py to expand the GGUF's scalar period in memory.
Triton attention is the default because FlashInfer 0.6.17 emits an unresolved
Muse-Glimmer paged-prefill symbol on Blackwell.

CONTEXT defaults to 64. DECODE_TOKENS requests 128 total generated tokens per
session by default; decode-rate accounting excludes each first token. RUNS defaults
to 3, WARMUP_RUNS to 1, and COOLDOWN_SECONDS to 2. OUTPUT defaults to
<repo>/bench-results/muse-glimmer/sglang-<variant>-concurrency-4-<time>.jsonl.
Set MODEL_VARIANT=radixark-nvfp4 to use the Safetensors checkpoint and optionally
set SGLANG_FP4_GEMM_BACKEND=flashinfer_cutlass to force native FP4. Install the
pinned environment with `uv pip install --prerelease=allow -r
packages/bench/sglang-requirements.txt`.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import signal
import statistics
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
from tokenizers import Tokenizer

CONCURRENCY = 4
GGUF_MODEL_NAME = "Muse-Glimmer-30B-UD-Q2_K_XL.gguf"
GGUF_MODEL_SHA256 = "3d63a1daff23fdc2a6927316151e855cacffe89b5cb9b9397a5aec0c412ec08d"
RADIXARK_MODEL_ID = "RadixArk/Muse-Glimmer-NVFP4"
RADIXARK_MODEL_REVISION = "1416629950afdfb276618fd4810103681dd10f4a"
RADIXARK_WEIGHT_SHA256 = (
    (
        "model-00001-of-00005.safetensors",
        "7f2de92688887833ac718aca71fb2ef3b53e9501af3001bf8784e09769938055",
    ),
    (
        "model-00002-of-00005.safetensors",
        "2b4007bbb6618217d5fd304f0639b1dd9f77bbf3f884a698846f05ae91ac6f4f",
    ),
    (
        "model-00003-of-00005.safetensors",
        "7395bb093d097ac86c646b836edb6248ed878f232714baedaafa8c05bfabe180",
    ),
    (
        "model-00004-of-00005.safetensors",
        "9eeaca869d4cb331d7ea87663a929820e9ebf59d6ac6678e96db483e78a7eae9",
    ),
    (
        "model-00005-of-00005.safetensors",
        "881ff22c3bf39a31c13f56273aec8a9e5beac8c247d3f217eb5a8bffe3b21424",
    ),
)
TOKENIZER_SHA256 = "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d"
WORDS = (
    "system",
    "harbor",
    "signal",
    "meadow",
    "copper",
    "lantern",
    "orbit",
    "timber",
    "cascade",
    "archive",
    "willow",
    "granite",
    "compass",
    "ember",
    "tunnel",
    "fabric",
)


def env_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return fallback
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw}") from error


def env_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return fallback
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw}") from error


def file_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Config:
    context: int
    decode_tokens: int
    runs: int
    warmup_runs: int
    warmup_tokens: int
    cooldown_seconds: float
    seed: int
    model_variant: str
    model_path: str
    model_name: str
    model_revision: str | None
    model_sha256: str | None
    load_format: str
    language_model_only: bool
    fp4_gemm_backend: str | None
    tokenizer_path: Path
    output: Path
    host: str
    port: int
    startup_timeout_seconds: float
    attention_backend: str

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_config() -> Config:
    repo_root = Path(__file__).resolve().parents[2]
    context = env_int("CONTEXT", 64)
    decode_tokens = env_int("DECODE_TOKENS", 128)
    runs = env_int("RUNS", 3)
    warmup_runs = env_int("WARMUP_RUNS", 1)
    warmup_tokens = env_int("WARMUP_TOKENS", 8)
    cooldown_seconds = env_float("COOLDOWN_SECONDS", 2.0)
    port = env_int("SGLANG_PORT", 30000)
    if context < 16:
        raise ValueError(f"CONTEXT must be at least 16, got {context}")
    if decode_tokens < 2:
        raise ValueError(f"DECODE_TOKENS must be at least 2, got {decode_tokens}")
    if runs <= 0:
        raise ValueError(f"RUNS must be positive, got {runs}")
    if warmup_runs < 0:
        raise ValueError(f"WARMUP_RUNS must be non-negative, got {warmup_runs}")
    if warmup_tokens < 2:
        raise ValueError(f"WARMUP_TOKENS must be at least 2, got {warmup_tokens}")
    if cooldown_seconds < 0:
        raise ValueError(
            f"COOLDOWN_SECONDS must be non-negative, got {cooldown_seconds}"
        )

    model_variant = os.environ.get("MODEL_VARIANT", "gguf")
    fp4_gemm_backend = os.environ.get("SGLANG_FP4_GEMM_BACKEND") or None
    if model_variant == "gguf":
        model_file = Path(
            os.environ.get(
                "MODEL_PATH",
                repo_root
                / "packages/examples/data/Muse-Glimmer-30B-UD-Q2_K_XL.gguf",
            )
        ).resolve()
        if not model_file.is_file():
            raise FileNotFoundError(f"model not found: {model_file}")
        model_hash = sha256(model_file)
        if model_hash != GGUF_MODEL_SHA256:
            raise ValueError(
                "model SHA-256 mismatch: "
                f"expected {GGUF_MODEL_SHA256}, got {model_hash}"
            )
        if fp4_gemm_backend is not None:
            raise ValueError("SGLANG_FP4_GEMM_BACKEND only applies to radixark-nvfp4")
        model_path = str(model_file)
        model_name = GGUF_MODEL_NAME
        model_revision = None
        model_sha256 = GGUF_MODEL_SHA256
        load_format = "gguf"
        language_model_only = False
        output_variant = ""
    elif model_variant == "radixark-nvfp4":
        configured_model_path = os.environ.get("MODEL_PATH")
        if configured_model_path is None:
            model_path = RADIXARK_MODEL_ID
        else:
            model_directory = Path(configured_model_path).resolve()
            if not model_directory.is_dir():
                raise FileNotFoundError(f"model not found: {model_directory}")
            for filename, expected_hash in RADIXARK_WEIGHT_SHA256:
                weight_path = model_directory / filename
                if not weight_path.is_file():
                    raise FileNotFoundError(
                        f"model weight not found: {weight_path}"
                    )
                weight_hash = sha256(weight_path)
                if weight_hash != expected_hash:
                    raise ValueError(
                        f"{filename} SHA-256 mismatch: "
                        f"expected {expected_hash}, got {weight_hash}"
                    )
            model_path = str(model_directory)
        model_name = RADIXARK_MODEL_ID
        model_revision = RADIXARK_MODEL_REVISION
        model_sha256 = None
        load_format = "auto"
        language_model_only = True
        output_variant = f"-radixark-nvfp4-{fp4_gemm_backend or 'auto'}"
    else:
        raise ValueError(
            f"MODEL_VARIANT must be gguf or radixark-nvfp4, got {model_variant}"
        )

    tokenizer_path = Path(
        os.environ.get(
            "TOKENIZER_PATH",
            repo_root / "packages/examples/data/muse-glimmer-tokenizer.json",
        )
    ).resolve()
    output = Path(
        os.environ.get(
            "OUTPUT",
            repo_root
            / "bench-results/muse-glimmer"
            / f"sglang{output_variant}-concurrency-4-{file_timestamp()}.jsonl",
        )
    ).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    tokenizer_hash = sha256(tokenizer_path)
    if tokenizer_hash != TOKENIZER_SHA256:
        raise ValueError(
            "tokenizer SHA-256 mismatch: "
            f"expected {TOKENIZER_SHA256}, got {tokenizer_hash}"
        )

    return Config(
        context=context,
        decode_tokens=decode_tokens,
        runs=runs,
        warmup_runs=warmup_runs,
        warmup_tokens=warmup_tokens,
        cooldown_seconds=cooldown_seconds,
        seed=env_int("SEED", 0),
        model_variant=model_variant,
        model_path=model_path,
        model_name=model_name,
        model_revision=model_revision,
        model_sha256=model_sha256,
        load_format=load_format,
        language_model_only=language_model_only,
        fp4_gemm_backend=fp4_gemm_backend,
        tokenizer_path=tokenizer_path,
        output=output,
        host=os.environ.get("SGLANG_HOST", "127.0.0.1"),
        port=port,
        startup_timeout_seconds=env_float("STARTUP_TIMEOUT_SECONDS", 900.0),
        attention_backend=os.environ.get("SGLANG_ATTENTION_BACKEND", "triton"),
    )


@dataclass(frozen=True)
class SessionResult:
    session: int
    started_at: float
    token_times: tuple[float, ...]
    prompt_tokens: int
    cached_tokens: int
    output_text_sha256: str

    @property
    def first_token_at(self) -> float:
        return self.token_times[0]

    @property
    def finished_at(self) -> float:
        return self.token_times[-1]

    @property
    def prefill_seconds(self) -> float:
        return self.first_token_at - self.started_at

    @property
    def decode_seconds(self) -> float:
        return self.finished_at - self.first_token_at

    @property
    def inter_token_seconds(self) -> tuple[float, ...]:
        return tuple(
            current - previous
            for previous, current in zip(self.token_times, self.token_times[1:])
        )


def percentile(values: tuple[float, ...] | list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def collect_metadata(config: Config, command: list[str]) -> dict[str, Any]:
    gpu_fields = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split(", ")
    )
    git_dirty = os.environ.get("EFFECT_TORCH_GIT_DIRTY")
    return {
        "benchmarkVersion": 3,
        "engine": "sglang",
        "sglangVersion": package_version("sglang"),
        "model": config.model_name,
        "modelVariant": config.model_variant,
        "modelRevision": config.model_revision,
        "modelSha256": config.model_sha256,
        "modelWeightSha256": (
            dict(RADIXARK_WEIGHT_SHA256)
            if config.model_variant == "radixark-nvfp4"
            else None
        ),
        "tokenizerSha256": TOKENIZER_SHA256,
        "loadFormat": config.load_format,
        "quantization": (
            "modelopt_mixed" if config.model_variant == "radixark-nvfp4" else "gguf"
        ),
        "languageModelOnly": config.language_model_only,
        "fp4GemmBackendRequested": config.fp4_gemm_backend or "auto",
        "fp4GemmBackendEffective": (
            config.fp4_gemm_backend
            or ("marlin" if config.model_variant == "radixark-nvfp4" else None)
        ),
        "attentionBackend": config.attention_backend,
        "radixCacheDisabled": True,
        "submissionMode": "single-streaming-http-batch",
        "pythonVersion": platform.python_version(),
        "torchVersion": package_version("torch"),
        "tritonVersion": package_version("triton"),
        "flashinferVersion": package_version("flashinfer-python"),
        "ggufVersion": package_version("gguf"),
        "tokenizersVersion": package_version("tokenizers"),
        "aiohttpVersion": package_version("aiohttp"),
        "numpyVersion": package_version("numpy"),
        "cudaToolkitVersion": package_version("cuda-toolkit"),
        "cudaCompilerVersion": package_version("nvidia-cuda-nvcc"),
        "gpu": gpu_fields[0],
        "driverVersion": gpu_fields[1],
        "gpuMemoryMiB": int(gpu_fields[2]),
        "platform": platform.platform(),
        "gitRevision": os.environ.get("EFFECT_TORCH_GIT_REVISION"),
        "gitDirty": None if git_dirty is None else git_dirty.lower() == "true",
        "seed": config.seed,
        "warmupRuns": config.warmup_runs,
        "warmupTokens": config.warmup_tokens,
        "serverCommand": shlex.join(command),
    }


def make_prompt(
    tokenizer: Tokenizer, bos_token_id: int, case_id: int, context: int
) -> list[int]:
    target = context - 1
    text = f"Muse Glimmer benchmark case {case_id} context {context}."
    word_index = (case_id * 7) % len(WORDS)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    while len(ids) < target:
        text += f" {WORDS[word_index]}"
        word_index = (word_index + 5) % len(WORDS)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
    return [bos_token_id, *ids[:target]]


async def run_batch(
    client: aiohttp.ClientSession,
    config: Config,
    tokenizer: Tokenizer,
    bos_token_id: int,
    case_base: int,
    output_tokens: int,
) -> tuple[SessionResult, ...]:
    prompts = [
        make_prompt(tokenizer, bos_token_id, case_base + session, config.context)
        for session in range(CONCURRENCY)
    ]
    if any(len(prompt) != config.context for prompt in prompts):
        raise RuntimeError("prompt construction produced an incorrect token count")
    if len({tuple(prompt) for prompt in prompts}) != CONCURRENCY:
        raise RuntimeError("prompt construction produced duplicate token sequences")

    seed = config.seed + case_base * CONCURRENCY
    body = {
        "input_ids": prompts,
        "sampling_params": [
            {
                "temperature": 0,
                "top_k": 0,
                "top_p": 1,
                "sampling_seed": seed + session,
                "max_new_tokens": output_tokens,
                "min_new_tokens": output_tokens,
                "ignore_eos": True,
                "stream_interval": 1,
            }
            for session in range(CONCURRENCY)
        ],
        "stream": True,
        "rid": [
            f"muse-glimmer-bench-{seed}-{session}" for session in range(CONCURRENCY)
        ],
    }
    started_at = time.perf_counter()
    token_times: list[list[float]] = [[] for _ in range(CONCURRENCY)]
    completion_tokens = [0] * CONCURRENCY
    prompt_tokens: list[int | None] = [None] * CONCURRENCY
    cached_tokens: list[int | None] = [None] * CONCURRENCY
    output_text = [""] * CONCURRENCY
    async with client.post(f"{config.server_url}/generate", json=body) as response:
        if response.status != 200:
            detail = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {detail}")
        async for raw_line in response.content:
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"SGLang stream error: {chunk['error']}")
            session = int(chunk.get("index", -1))
            if not 0 <= session < CONCURRENCY:
                raise RuntimeError(f"invalid streamed batch index: {session}")
            meta = chunk.get("meta_info", {})
            if isinstance(chunk.get("text"), str):
                output_text[session] = chunk["text"]
            observed_prompt_tokens = int(meta.get("prompt_tokens", -1))
            observed_cached_tokens = int(meta.get("cached_tokens", -1))
            if observed_prompt_tokens != config.context:
                raise RuntimeError(
                    f"session {session}: server reported {observed_prompt_tokens} "
                    f"prompt tokens, expected {config.context}"
                )
            if observed_cached_tokens != 0:
                raise RuntimeError(
                    f"session {session}: server reused {observed_cached_tokens} tokens"
                )
            prompt_tokens[session] = observed_prompt_tokens
            cached_tokens[session] = observed_cached_tokens

            current_tokens = int(meta.get("completion_tokens", 0))
            if current_tokens <= completion_tokens[session]:
                continue
            if current_tokens != completion_tokens[session] + 1:
                raise RuntimeError(
                    f"session {session}: stream advanced from "
                    f"{completion_tokens[session]} to {current_tokens} tokens"
                )
            token_times[session].append(time.perf_counter())
            completion_tokens[session] = current_tokens

    results: list[SessionResult] = []
    for session in range(CONCURRENCY):
        if completion_tokens[session] != output_tokens:
            raise RuntimeError(
                f"session {session}: generated {completion_tokens[session]} tokens, "
                f"expected {output_tokens}"
            )
        if prompt_tokens[session] is None or cached_tokens[session] is None:
            raise RuntimeError(f"session {session}: missing streamed token metadata")
        results.append(
            SessionResult(
                session,
                started_at,
                tuple(token_times[session]),
                prompt_tokens[session],
                cached_tokens[session],
                hashlib.sha256(output_text[session].encode()).hexdigest(),
            )
        )
    return tuple(results)


def session_record(
    config: Config,
    run: int,
    result: SessionResult,
    metadata: dict[str, Any],
    server_load_ms: float,
) -> dict[str, Any]:
    decode_tokens = len(result.token_times) - 1
    itl_ms = [value * 1000 for value in result.inter_token_seconds]
    return {
        **metadata,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "session",
        "concurrency": CONCURRENCY,
        "run": run,
        "session": result.session,
        "context": config.context,
        "inputTokens": config.context,
        "serverPromptTokens": result.prompt_tokens,
        "serverCachedTokens": result.cached_tokens,
        "outputTokens": len(result.token_times),
        "outputTextSha256": result.output_text_sha256,
        "prefillBasis": "client-ttft",
        "prefillMs": result.prefill_seconds * 1000,
        "prefillTokPerSec": config.context / result.prefill_seconds,
        "decodeTokens": decode_tokens,
        "decodeMs": result.decode_seconds * 1000,
        "decodeTokPerSec": decode_tokens / result.decode_seconds,
        "interTokenP50Ms": percentile(itl_ms, 0.5),
        "interTokenP95Ms": percentile(itl_ms, 0.95),
        "e2eMs": (result.finished_at - result.started_at) * 1000,
        "serverLoadMs": server_load_ms,
    }


def aggregate_record(
    config: Config,
    run: int,
    results: tuple[SessionResult, ...],
    metadata: dict[str, Any],
    server_load_ms: float,
) -> dict[str, Any]:
    started_at = min(result.started_at for result in results)
    first_token_at = min(result.first_token_at for result in results)
    all_prefills_finished_at = max(result.first_token_at for result in results)
    finished_at = max(result.finished_at for result in results)
    input_tokens = CONCURRENCY * config.context
    output_tokens = sum(len(result.token_times) for result in results)
    total_decode_tokens = output_tokens - CONCURRENCY
    decode_tokens = sum(
        token_time > all_prefills_finished_at
        for result in results
        for token_time in result.token_times[1:]
    )
    prefill_seconds = all_prefills_finished_at - started_at
    decode_seconds = finished_at - all_prefills_finished_at
    all_itl_ms = [
        (current - previous) * 1000
        for result in results
        for previous, current in zip(result.token_times, result.token_times[1:])
        if previous >= all_prefills_finished_at
    ]
    if decode_tokens <= 0 or not all_itl_ms:
        raise RuntimeError("no common four-session decode interval was observed")
    return {
        **metadata,
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": "aggregate",
        "concurrency": CONCURRENCY,
        "run": run,
        "context": config.context,
        "inputTokens": input_tokens,
        "serverPromptTokens": sum(result.prompt_tokens for result in results),
        "serverCachedTokens": sum(result.cached_tokens for result in results),
        "outputTokens": output_tokens,
        "prefillBasis": "batched-client-ttft-through-last-first-token",
        "prefillMs": prefill_seconds * 1000,
        "prefillTokPerSec": input_tokens / prefill_seconds,
        "decodeBasis": "all-sessions-active-after-last-first-token",
        "decodeTokens": decode_tokens,
        "decodeTokensBeforeAllActive": total_decode_tokens - decode_tokens,
        "decodeMs": decode_seconds * 1000,
        "decodeTokPerSec": decode_tokens / decode_seconds,
        "interTokenP50Ms": percentile(all_itl_ms, 0.5),
        "interTokenP95Ms": percentile(all_itl_ms, 0.95),
        "e2eMs": (finished_at - started_at) * 1000,
        "outputTokPerSecE2e": output_tokens / (finished_at - started_at),
        "firstTokenSkewMs": (all_prefills_finished_at - first_token_at) * 1000,
        "serverLoadMs": server_load_ms,
    }


def print_run(records: list[dict[str, Any]]) -> None:
    aggregate = next(record for record in records if record["scope"] == "aggregate")
    print(
        f"run={aggregate['run']} aggregate "
        f"prefill={aggregate['prefillTokPerSec']:.2f} tok/s "
        f"({aggregate['prefillMs']:.2f} ms), "
        f"decode={aggregate['decodeTokPerSec']:.2f} tok/s "
        f"({aggregate['decodeMs']:.2f} ms)"
    )
    for record in records:
        if record["scope"] != "session":
            continue
        print(
            f"  session={record['session']} "
            f"prefill={record['prefillTokPerSec']:.2f} tok/s "
            f"({record['prefillMs']:.2f} ms), "
            f"decode={record['decodeTokPerSec']:.2f} tok/s "
            f"({record['decodeMs']:.2f} ms), "
            f"ITL p50={record['interTokenP50Ms']:.2f} ms "
            f"p95={record['interTokenP95Ms']:.2f} ms"
        )


async def wait_for_server(
    server_url: str, process: asyncio.subprocess.Process, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    timeout = aiohttp.ClientTimeout(total=2)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        while time.monotonic() < deadline:
            exit_code = process.returncode
            if exit_code is not None:
                raise RuntimeError(
                    f"SGLang exited during startup with code {exit_code}"
                )
            try:
                async with client.get(f"{server_url}/health") as response:
                    if response.status == 200:
                        return
            except (aiohttp.ClientError, TimeoutError):
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"SGLang did not become healthy within {timeout_seconds}s")


async def stop_server(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=30)
    except TimeoutError:
        os.killpg(process.pid, signal.SIGKILL)
        await asyncio.wait_for(process.wait(), timeout=10)


async def benchmark(config: Config) -> None:
    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    bos_token_id = tokenizer.token_to_id("<|begin_of_text|>")
    if bos_token_id is None:
        raise ValueError("tokenizer has no <|begin_of_text|> token")

    process: asyncio.subprocess.Process | None = None
    log_file = None
    try:
        log_path = config.output.with_suffix(".server.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("wb")
        cuda_home = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
        if not cuda_home.is_dir():
            raise FileNotFoundError(f"SGLang CUDA 13 toolkit not found: {cuda_home}")
        cuda_lib = cuda_home / "lib"
        cuda_lib64 = cuda_home / "lib64"
        if not cuda_lib64.exists():
            cuda_lib64.symlink_to(cuda_lib, target_is_directory=True)
        cudart_link = cuda_lib / "libcudart.so"
        cudart_versioned = cuda_lib / "libcudart.so.13"
        if not cudart_link.exists():
            if not cudart_versioned.is_file():
                raise FileNotFoundError(
                    f"CUDA runtime library not found: {cudart_versioned}"
                )
            cudart_link.symlink_to(cudart_versioned.name)
        server_path = os.pathsep.join(
            (
                str(Path(sys.executable).parent),
                str(cuda_home / "bin"),
                "/usr/bin",
                os.environ["PATH"],
            )
        )
        server_library_path = os.pathsep.join(
            filter(None, (str(cuda_home / "lib"), os.environ.get("LD_LIBRARY_PATH")))
        )
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            config.model_path,
            "--load-format",
            config.load_format,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--max-running-requests",
            str(CONCURRENCY),
            "--max-total-tokens",
            str(max(4096, CONCURRENCY * (config.context + config.decode_tokens + 16))),
            "--cuda-graph-max-bs-decode",
            str(CONCURRENCY),
            "--cuda-graph-max-bs-prefill",
            str(CONCURRENCY * config.context),
            "--attention-backend",
            config.attention_backend,
            "--disable-radix-cache",
        ]
        if config.model_revision is not None:
            command.extend(("--revision", config.model_revision))
        if config.language_model_only:
            command.append("--language-model-only")
        if config.fp4_gemm_backend is not None:
            command.extend(("--fp4-gemm-backend", config.fp4_gemm_backend))
        metadata = collect_metadata(config, command)
        print(f"starting SGLang; server log: {log_path}", file=sys.stderr)
        load_started = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "CC": os.environ.get("SGLANG_CC", "/usr/bin/gcc"),
                "CXX": os.environ.get("SGLANG_CXX", "/usr/bin/g++"),
                "CUDA_HOME": str(cuda_home),
                "CUDA_PATH": str(cuda_home),
                "LD_LIBRARY_PATH": server_library_path,
                "LIBRARY_PATH": server_library_path,
                "PATH": server_path,
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(Path(__file__).resolve().parent / "sglang-compat"),
                            os.environ.get("PYTHONPATH"),
                        ),
                    )
                ),
                "SGLANG_CACHE_DIR": os.environ.get(
                    "SGLANG_BENCH_CACHE_DIR",
                    str(Path.home() / ".cache/sglang-muse-glimmer-cu130"),
                ),
            },
        )
        await wait_for_server(
            config.server_url, process, config.startup_timeout_seconds
        )
        server_load_ms = (time.perf_counter() - load_started) * 1000
        print(f"SGLang ready in {server_load_ms / 1000:.2f}s", file=sys.stderr)

        timeout = aiohttp.ClientTimeout(total=900)
        connector = aiohttp.TCPConnector(limit=1)
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as client:
            for warmup in range(config.warmup_runs):
                print(f"warmup {warmup + 1}/{config.warmup_runs}", file=sys.stderr)
                await run_batch(
                    client,
                    config,
                    tokenizer,
                    bos_token_id,
                    1_000_000 + warmup * CONCURRENCY,
                    config.warmup_tokens,
                )

            config.output.parent.mkdir(parents=True, exist_ok=True)
            aggregate_records: list[dict[str, Any]] = []
            with config.output.open("w", encoding="utf-8") as output:
                for run in range(config.runs):
                    if run > 0 and config.cooldown_seconds > 0:
                        await asyncio.sleep(config.cooldown_seconds)
                    results = await run_batch(
                        client,
                        config,
                        tokenizer,
                        bos_token_id,
                        run * CONCURRENCY,
                        config.decode_tokens,
                    )
                    records = [
                        session_record(config, run, result, metadata, server_load_ms)
                        for result in results
                    ]
                    aggregate = aggregate_record(
                        config, run, results, metadata, server_load_ms
                    )
                    records.append(aggregate)
                    aggregate_records.append(aggregate)
                    for record in records:
                        output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    output.flush()
                    print_run(records)

            print("\nmedian aggregate throughput")
            print(
                f"prefill={statistics.median(record['prefillTokPerSec'] for record in aggregate_records):.2f} tok/s, "
                f"decode={statistics.median(record['decodeTokPerSec'] for record in aggregate_records):.2f} tok/s"
            )
            print(f"wrote {config.output}", file=sys.stderr)
    finally:
        if process is not None:
            await stop_server(process)
        if log_file is not None:
            log_file.close()


def main() -> None:
    config = load_config()
    print(
        f"SGLang {package_version('sglang')}, concurrency={CONCURRENCY}, context={config.context}, "
        f"decodeTokens={config.decode_tokens}, runs={config.runs}",
        file=sys.stderr,
    )
    asyncio.run(benchmark(config))


if __name__ == "__main__":
    main()
