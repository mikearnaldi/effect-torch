# DiffusionGemma benchmark

This benchmark compares Effect Torch with vLLM on one RTX PRO 6000 Blackwell
using the pinned BF16 DiffusionGemma checkpoint. It measures the direct engine
boundary and the OpenAI-compatible streaming HTTP boundary. Provision at least
160 GB of container disk for the 51.7 GB checkpoint, both runtime environments,
and native build outputs.

The benchmark does not permit quantization, CPU offload, a different checkpoint,
or altered diffusion settings. The official Transformers implementation is the
correctness oracle, not a third performance target.

## Generation matching

The archived vLLM 0.24.0 medians are not a matched-generation baseline. Its
threshold-1 stability check does not compare against the prior prediction, and
its diffusion sampler does not use the request seed for its random draws.
Sampler entropy and self-conditioning rounding also differ from the official
reference. Equal configuration values alone do not establish equal work.

The experimental `vllm-policy-overlay.py` and `shared_rng.py` isolate alignment
changes from the installed vLLM package. Their results must identify the overlay
source hashes. Shared RNG has been checked bitwise against Effect's device
uniforms and canvas stream; this does not imply identical model numerics or
generation trajectories.

Direct results record actual prompt/output token IDs, request seeds, refinement
counts, block counts, and finish reasons. Run `compare-generation.py effect.jsonl
vllm.jsonl` before comparing latency. It rejects missing or different generation
evidence and compares the same end-to-end batch wall-time field. Diagnostic
traces (`GENERATION_TRACE=1` on Effect) are marked separately and cannot supply
timing claims. Tokenization, initialization, and warmup remain outside measured
requests; backend work needed to return the requested tokens remains inside.

Controlled trajectory replay is a separate workload for comparing identical
model inputs when natural generation diverges. `controlled-replay-direct.ts`
captures canonical canvases, BF16 feedback, temperatures, and stopping decisions;
its timing mode and `vllm_controlled_replay.py` replay those inputs with feedback
banks preloaded outside the timer. Both backends still execute the model and
sampler for every refinement and include one terminal full-canvas encoder
commit. Prefix caching is disabled, with two warmups and five measured repeats
of each canonical request. The comparator requires matching replay manifests,
outputs, refinement counts, and commit metadata. A replay win does not establish
equal natural-generation outputs or natural-generation performance.

For a comparison across distinct trajectories, set `REPLAY_DISTINCT_SEEDS=1`
when capturing and timing Effect and `ET_VLLM_REPLAY_DISTINCT_SEEDS=1` when timing
vLLM. This preserves the original request seed schedule: warmups -1 and -2,
then measured runs 0 through 4. Capture into a fresh directory. Use one prompt
target per directory/process (`REPLAY_TARGETS` on Effect and a single-target
matrix manifest on vLLM) to keep all seven feedback banks resident outside the
timers. The comparator rejects different seed modes, repeated manifests, missing
measured runs, and incorrect request seeds. Distinct runs also require actual
prefill, model-read, sampler, and terminal-commit invocation evidence and zero
cooldown between repetitions on both backends. Fixed-trajectory replay
remains the default; graph-cache gains on repeated routes need this additional
distinct-trajectory validation before a broader performance claim.

The audited 2026-09-30 controlled replay baseline used 15 and 17 refinements
respectively. Five-run medians were 877.915 / 993.944 ms for Effect and
557.195 / 637.257 ms for the vLLM replay overlay, at 32 / 128 prompt tokens and
64 output tokens. Effect has not beaten this baseline. The vLLM overlay also
removes discarded terminal-commit logits and sampler computation so both
backends perform the same useful encoder work; these figures describe that
explicit replay configuration, not the unmodified vLLM package. Evidence and
source attestations are in the ignored `bench-results/diffusion-gemma-20260930`
directory.

The subsequently validated short-axis Max specialization reduced Effect replay
medians to 821.986 / 942.304 ms, with exact oracle gates and zero cleanup bytes.
Adding the scatter route map and weighted-source fusion measured
808.333 / 924.174 ms, also passing both oracle gates. These opt-ins are
`EFFECT_TORCH_CUDA_SMALL_MAX=1`,
`EFFECT_TORCH_CUDA_ORDERED_SCATTER_ROUTE_MAP=1`, and
`EFFECT_TORCH_CUDA_ORDERED_SCATTER_WEIGHTED=1`; the performance goal remains unmet.
Packing the five small sampler statistics into one exact F32 readback further
measured 802.563 / 917.035 ms. Both oracle gates and a separate comparison of all
10 natural-generation requests passed, including tokens, seeds, refinement
counts, and cleanup. These remain fixed-trajectory replay measurements.

The distinct-seed comparison subsequently passed at both prompt sizes with all
graph caches disabled. Effect medians were **755.445 / 868.406 ms**, versus vLLM
**521.548 / 604.745 ms**. The five measured refinement counts were
15/12/14/14/16 at 32 prompt tokens and 17/16/14/19/12 at 128. These differing
trajectories explain why the absolute medians should not be compared directly
with the fixed 15/17-refinement figures above. The performance goal remains unmet.

All seven cases per target, including warmups, validate actual prefill/model-read/
sampler/terminal-commit counts of 1/N/N/1. Their feedback files are hash-checked
before GPU-bank loading, and both runners record zero cooldown between
repetitions. The strict paired comparison checks all ten measured requests.
Evidence, executed sources, import attestations, and CPU proofs are archived in
`bench-results/diffusion-gemma-20260930/distinct-replay-evidence-20260930.tar.gz`;
the extracted `distinct-replay-20260930/comparison-summary.json` records both
points. This remains controlled replay with forced canonical inputs and stopping,
not a claim that natural generation agrees between the engines.

## Rebuild the correctness oracle

In the pinned CUDA shell, use `bash diffusion-gemma/reference.sh` from
`packages/bench`. The `setup` command creates the independent Torch 2.10.0+cu128
environment with the pinned Transformers source. `REFERENCE_ENVIRONMENT` can
select another environment directory; setup refuses to overwrite one.

```bash
bash diffusion-gemma/reference.sh setup
bash diffusion-gemma/reference.sh download /root/models/diffusiongemma
bash diffusion-gemma/reference.sh oracle /root/models/diffusiongemma /root/new-oracle
bash diffusion-gemma/reference.sh components /root/models/diffusiongemma /root/new-components
```

The downloader verifies all checkpoint shards against official hashes. The
oracle exports the CPU-initialized RoPE buffers, four independent full-vocabulary
answer rows, and two generated blocks with six refinement steps. It explicitly
selects eager attention and eager experts; Transformers' default grouped expert
implementation changes BF16 results. The generation recorder checks each saved
exponential tensor against the original multinomial sample and CUDA RNG state.
Use new output directories to preserve existing evidence. The `state` command
generates only the initialized buffers without loading the full checkpoint.

Use `new-oracle/initialized-state/manifest.json` for
`EFFECT_TORCH_DIFFUSION_GEMMA_INITIALIZED_STATE`, and run `verify-full.ts` with
`4 256,278`. `new-oracle/generation-inputs.json` supplies the generation replay
inputs and the same encoder geometry. The native expert fixture directory is
`new-components/expert-components`; select it with
`EFFECT_TORCH_GROUPED_EXPERT_FIXTURE_DIR` for the captured-projection Rust test.
Copy oracle data and reports off the pod before deleting it.

`expert-gemm-graphs.cu` measures unchanged cuBLAS calls against per-expert,
per-stream, and combined-stream CUDA graph replay at several expert sizes.
Its fixed routing and stable pointers give a limited estimate of replay benefits;
its timings exclude graph construction and are not full-model measurements.
Compile in the pinned CUDA shell with `nvcc -O3 -arch=sm_120
expert-gemm-graphs.cu -lcublas -o /tmp/expert-gemm-graphs`, then run that binary.

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
measurements are accepted. The original performance runners use different random
generators despite equal request seeds. Those results remain exploratory;
matched-generation claims additionally require the comparison checks above.
The oracle replay injects recorded random inputs and requires identical steps.

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

## Concurrent CUDA timeline

`cupti-timeline.cpp` is a standalone CUPTI injection library. It records concurrent
kernel execution, copies, and driver/runtime API intervals without inserting
CUDA synchronization or disabling planned overlap. Build against the CUDA/CUPTI
headers and matching library used on the machine, for example:

```bash
cupti_root=/root/.cache/effect-torch/reference-2.10/lib/python3.12/site-packages/nvidia/cuda_cupti
g++ -std=c++17 -O2 -shared -fPIC -pthread cupti-timeline.cpp \
  -I"$cupti_root/include" -I/usr/local/cuda/include \
  -L"$cupti_root/lib" -Wl,-rpath,"$cupti_root/lib" \
  -l:libcupti.so.12 -o /root/cupti-timeline.so
```

Set `CUDA_INJECTION64_PATH=/root/cupti-timeline.so` and
`EFFECT_TORCH_CUPTI_OUTPUT=/root/new-timeline.jsonl` on the benchmark process,
alongside the candidate's normal environment. Do not enable the runtime's serial
`EFFECT_TORCH_CUDA_TRACE` or `EFFECT_TORCH_CUDA_GROUPED_PROFILE_PATH` diagnostics.
Use a bounded manifest: one warmup and one measurement is sufficient initially.
Use `%p` in the output filename for multiprocess programs; it expands to each
process PID while preserving exclusive creation. For workers that exit through
`os._exit`, diagnostic-only `cupti_finalize.register()` installs an idempotent
multiprocessing/atexit flush when `EFFECT_TORCH_CUPTI_FINALIZE=1`. Require a
complete summary from the CUDA worker, not only the parent process.
`EFFECT_TORCH_CUPTI_WINDOW=1` records wall-clock bounds immediately outside the
existing request timer and labels rows `diagnostic-profile`. Effect preserves
both warmups and all seven preloaded distinct cases, measuring only run0 in
this mode; use a matching one-measurement manifest for vLLM. The analyzer uses
these semantic bounds instead of inferring a boundary from later bookkeeping.

The output path must be new. Records are buffered asynchronously, retained up to
1 GiB, and written at normal process exit. A killed process may leave an empty
file. The final summary reports CUPTI drops, collector omissions, and errors;
the analyzer rejects an incomplete capture.

```bash
python3 cupti-timeline.py /root/new-timeline.jsonl \
  --benchmark /root/new-effect-benchmark.jsonl --benchmark-row 0
```

Benchmark timestamp alignment has approximately millisecond boundary precision,
plus the small bookkeeping delay between request completion and record creation.
Alternatively use explicit CUPTI `--start-ns` and `--end-ns`. `--list-segments`
locates GPU activity separated by large gaps; these segments are not necessarily
individual requests. A measured request may immediately follow warmup.

Kernel/API duration sums overlap. GPU activity union measures whether any kernel
or copy is active, not SM utilization. API durations include synchronization waits.
An API beginning during a GPU idle gap supports a late-submission explanation but
does not identify its cause. Compare instrumented request time with the ordinary
benchmark to assess observer overhead; do not accept profiled timing as the final
performance result.

## Experimental exact merged expert kernel

The opt-in merged expert path keeps each mapped split-K partial in registers,
rounds it to BF16, and sums those rounded partials in the original F32 order.
M1 and unsupported shapes continue through cuBLAS. It is restricted to the
validated RTX PRO 6000 Blackwell Server Edition and cuBLAS 12.9.1 fingerprint.

Build its separate PTX artifact inside the CUDA Nix environment:

```bash
./scripts/build-cuda-expert-merged.sh /absolute/cutlass/include /absolute/expert-merged.ptx
export EFFECT_TORCH_CUDA_EXPERT_MERGED_PTX=/absolute/expert-merged.ptx
```

The normal package build does not require CUTLASS or compile this artifact.
Set the variable before creating the CUDA runtime and compiling the model.
The runtime loads the module eagerly; its bounded descriptor and shape storage
belongs to each invocation. Unset the variable for a paired baseline. Record the
PTX SHA256 alongside native addon/source hashes because changing that external
artifact changes the implementation being measured.

The focused hardware test covers dynamic routing, descriptor upload boundaries,
ordinary fallback, retained outputs, and failure cleanup:

```bash
cargo test -p effect-torch-runtime-cuda --lib \
  device::expert_merged::tests::merged_descriptor_chunks_dynamic_routing_and_error_ownership \
  -- --ignored --exact
```

The accepted exact sampler flags are `EFFECT_TORCH_CUDA_DIV_FEEDBACK=1` and
`EFFECT_TORCH_CUDA_BF16_SOFTMAX=1`. The first writes the F32 division and BF16
feedback outputs in one pass, with separate allocations. The second preserves
the original F32 maximum/sum trees and exponentials while removing intermediate
BF16 feedback widening and reduction passes. Both remain opt-in.

On the pinned GPU, the combined distinct replay medians are **725.6818 ms**
(prompt32) and **843.4752 ms** (prompt128), versus matched vLLM **521.5485 ms**
and **604.7446 ms**. The optimization goal remains unmet. Combined oracle and
natural request contracts match exactly. Evidence is in
`bench-results/diffusion-gemma-20260930/dual-store-distinct`.
The measured policy21 native source archive SHA256 is
`8a5adf0ae7597d04af06e4eb5d5a88a571f0bb69690874158f7552aa3042d92b`, addon SHA256
`52f58513ed05e78edabef60953df7c1c3564acbf2d67e83ecfa70cb511549fd0`, and merged
PTX SHA256 `56e1f7b1934ac08afc48625600d3e25f3ffe1f5d64e4fcee04a87359b63f6b4e`.
