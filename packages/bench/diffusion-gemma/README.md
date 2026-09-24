# DiffusionGemma benchmark

This benchmark compares Effect Torch with vLLM on one RTX PRO 6000 Blackwell
using the pinned BF16 DiffusionGemma checkpoint. It measures the direct engine
boundary and the OpenAI-compatible streaming HTTP boundary. Provision at least
160 GB of container disk for the 51.7 GB checkpoint, both runtime environments,
and native build outputs.

The benchmark does not permit quantization, CPU offload, a different checkpoint,
or altered diffusion settings. The official Transformers implementation is the
correctness oracle, not a third performance target.

## Pinned target

- Model: google/diffusiongemma-26B-A4B-it
- Revision: f7f5b7f5fa82ffc52addd066915886d497f5517b
- Precision: BF16
- Canvas: 256 tokens
- Maximum denoising steps: 48
- Entropy bound: 0.1
- Temperature schedule: 0.8 to 0.4
- Adaptive confidence threshold: 0.005
- EOS token IDs: 1, 106, 50
- Pad token ID: 0
- vLLM: 0.24.0
- vLLM image: v0.24.0 at the digest in manifest.json
- vLLM wheel: v0.24.0 x86_64 at the SHA-256 in manifest.json

vLLM merged native DiffusionGemma support in commit
eb28452b10a1376d143b2847a78b31726db346dd. The pinned release image identifies
build commit ee0da84ab9e04ac7610e28580af62c365e898389, CUDA 13.0.2, and
sm_120 kernels. Its published BF16 recipe caps max_num_seqs at four, so both
engines use concurrency 1, 2, and 4.

## Result files

Every runner writes JSONL under bench-results/diffusion-gemma by default.
Records include the model revision, generation policy, prompt hashes, actual
token counts, output count, elapsed time, and throughput. The comparison script
rejects pairs whose prompt IDs, hashes, token counts, or run counts differ.

The committed prompt definitions contain deterministic padding calibrated with
the pinned tokenizer. Every runner requires actualPromptTokens to equal the
32/128/512/2048 target exactly; the comparison also requires both engines to
report identical counts.

## Correctness gate

Do not time a build until the CUDA runs of
packages/examples/scripts/diffusion-gemma/verify-full.ts and
verify-generation.ts pass against the saved official Transformers oracle. The
generation replay uses the oracle's prompt, initial canvases, F32 exponential
noise, denoising settings, stepwise argmax canvases, and final token IDs. Keep
those oracle and replay runs out of benchmark result files.

Repeat the gate after a change to kernels, compilation, model math, sampling, or
state ownership. Performance-only changes still need the fixed prompt suite
checked for malformed output, empty output, and unexpected stopping before their
measurements are accepted. The performance runners seed each request, but the two
engines use different random generators. Their output text need not match. Only
the oracle replay injects identical random inputs and requires identical steps.

## Direct Effect Torch run

Build the CUDA and tokenizer addons, then point MODEL_PATH at the exact pinned
Hugging Face snapshot:

    MODEL_PATH=/models/diffusiongemma \
      INITIALIZED_STATE_PATH=/oracle/initialized-rope.safetensors \
      pnpm bench:diffusion-gemma

The Effect runner verifies and binds the pinned configuration-derived RoPE state
used by the official oracle. Checkpoint loading, graph compilation, and each
matrix point's warmup are
reported separately. The vLLM runner likewise reports model initialization and
warmup. Timed runs exclude tokenization. HTTP server startup stays outside the
client result file and must be retained with the server log.

## Pinned vLLM environment

The RunPod devbox does not provide a nested Docker daemon. Install the official
vLLM 0.24.0 wheel into an isolated environment before the first run:

    pnpm bench:diffusion-gemma-vllm-setup
    VLLM_ENV=$HOME/.cache/effect-torch/vllm-0.24.0

The setup script installs the exact wheel URL and SHA-256 from manifest.json. The
runner verifies its PEP 610 installation record. On a host with Docker, the
pinned release image in the manifest is also accepted after its build commit is
verified.

## Direct vLLM run

Keep the Hugging Face cache between the direct and HTTP runs so downloads never
enter a measurement:

    VLLM_PYTHON=$VLLM_ENV/bin/python pnpm bench:diffusion-gemma-vllm

The runner rejects other vLLM distributions, versions older than 0.24.0,
multiple visible GPUs, and GPUs with less than the manifest's required memory.

## Effect Torch HTTP run

Start the server with four admission slots:

    MAX_CONCURRENT_REQUESTS=4 MAX_TOKENS=8192 \
      pnpm --filter @effect-torch/examples serve-cuda /models/diffusiongemma 8000

Run the client through the isolated environment in another shell:

    BENCH_PYTHON=$VLLM_ENV/bin/python \
      ENGINE=effect-torch BASE_URL=http://127.0.0.1:8000 \
      pnpm bench:diffusion-gemma-http

## vLLM HTTP run

Start vLLM with the exact model revision and generation policy:

    $VLLM_ENV/bin/vllm serve google/diffusiongemma-26B-A4B-it \
      --revision f7f5b7f5fa82ffc52addd066915886d497f5517b \
      --served-model-name diffusiongemma-vllm \
      --dtype bfloat16 \
      --max-model-len 8192 \
      --max-num-seqs 4 \
      --gpu-memory-utilization 0.85 \
      --enable-chunked-prefill \
      --diffusion-config '{"canvas_length":256,"max_denoising_steps":48}' \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      --host 127.0.0.1 --port 8000

Then run the same client matrix:

    BENCH_PYTHON=$VLLM_ENV/bin/python \
      ENGINE=vllm MODEL_NAME=diffusiongemma-vllm \
      BASE_URL=http://127.0.0.1:8000 \
      pnpm bench:diffusion-gemma-http

The HTTP client measures end-to-end latency, time to first nonempty text,
committed output tokens per second, per-request token counts, page count, and
inter-page cadence. It requests stream usage and fails if the server omits it or
if a request emits no text page.

## GPU telemetry

Wrap each measured command with telemetry.py. It samples the one visible GPU
every 100 milliseconds and refuses to overwrite an existing output file. For an
Effect Torch direct run:

    pnpm bench:diffusion-gemma-telemetry \
      --output bench-results/diffusion-gemma/effect-direct.telemetry.jsonl \
      -- env MODEL_PATH=/models/diffusiongemma pnpm bench:diffusion-gemma

For vLLM, run the wrapper itself through the isolated environment so the package
hash is tied to the measured result:

    BENCH_PYTHON=$VLLM_ENV/bin/python pnpm bench:diffusion-gemma-telemetry \
      --output bench-results/diffusion-gemma/vllm-direct.telemetry.jsonl \
      -- env VLLM_PYTHON=$VLLM_ENV/bin/python pnpm bench:diffusion-gemma-vllm

Use the same wrapper around each HTTP client command. A synced devbox has no
`.git` directory, so pass `BENCH_REPOSITORY_COMMIT`, `BENCH_SOURCE_TREE_SHA256`,
and `BENCH_REPOSITORY_DIRTY` from the local `repository_identity` result. The
JSONL metadata records the GPU UUID, driver, power limit, maximum clocks,
source-tree hash, Python package-set hash, sampling interval, and measured
command. Samples record memory
use, GPU utilization, power, clocks, and temperature.

## Compare results

Pass one Effect Torch file and one vLLM file from the same boundary:

    pnpm bench:diffusion-gemma-summary \
      bench-results/diffusion-gemma/effect-direct-*.jsonl \
      bench-results/diffusion-gemma/vllm-direct-*.jsonl \
      --telemetry effect-torch=bench-results/diffusion-gemma/effect-direct.telemetry.jsonl \
      --telemetry vllm=bench-results/diffusion-gemma/vllm-direct.telemetry.jsonl

Run direct and HTTP comparisons separately. The summary rejects mixed boundaries,
missing matrix points, missing run numbers, and points without exactly the two
target engines. Use --allow-partial only while debugging the harness, never for a
published baseline. With telemetry, the report includes peak VRAM, mean
utilization, mean power, and joules per committed token. Energy uses mean sampled
power over each timed window.

## Measurement rules

- Run one engine at a time on an otherwise idle GPU.
- Record driver, CUDA, image digest, GPU clocks, power limit, and repository
  commit before measuring.
- Do not include model downloads in load time.
- Let every matrix point finish its configured warmup.
- Alternate engine order on repeated full runs.
- Profile only after collecting an uncontested baseline. Profilers change the
  timings and their runs must not enter the comparison JSONL.
