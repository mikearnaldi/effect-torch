# Development

This guide covers work on effect-torch itself. See [README.md](README.md) for package usage and the public API.

## Contents

- [Environment](#environment)
- [Native development builds](#native-development-builds)
- [Quality checks](#quality-checks)
- [CUDA devbox](#cuda-devbox)
- [Release build](#release-build)
- [Examples](#examples)
- [Benchmarks](#benchmarks)
- [Repository layout](#repository-layout)
- [Design documents](#design-documents)

## Environment

A Nix flake and direnv configure the macOS and Linux development shells. The
shell includes Node.js 22, Corepack, Rustup, Zig, `cargo-zigbuild`, dprint,
CMake, and pkg-config.

```bash
direnv allow
pnpm install
```

Without direnv:

```bash
nix develop
pnpm install
```

`rust-toolchain.toml` pins Rust, rustfmt, rust-analyzer, and the complete
standard-library target set.

Outside Nix, install Node, pnpm, the pinned Rust toolchain, Zig, and
`cargo-zigbuild`. Darwin builds also require Xcode Command Line Tools.

Do not edit generated output under `dist/`, `target/`, or `node_modules/`.

## Native development builds

Workspace TypeScript resolves directly to package source, but native packages
load addons from their own `dist/internal` directories. Build a host addon
before running code against a fresh checkout.

```bash
pnpm --filter @effect-torch/backend-cpu build:debug
pnpm --filter @effect-torch/backend-apple-native build:debug
pnpm --filter @effect-torch/backend-cuda build:debug
pnpm --filter @effect-torch/tokenizers build:debug
```

The Apple command is macOS-only. The CUDA command requires x86_64 Linux with
glibc, an NVIDIA GPU and driver, and the `nix develop .#cuda` shell.

Host debug builds preserve any other already-assembled matrix artifacts. A
host release build is available through `scripts/build-native.mjs --host
--profile release` from a native package directory.

## Quality checks

Build the native addon needed by a test before running it. Use the narrowest
useful check while editing:

```bash
pnpm --filter @effect-torch/core exec vitest run test/Tensor.test.ts
pnpm --filter @effect-torch/core exec vitest run test/Tensor.test.ts -t "broadcasting"
cargo test -p effect-torch-compiler
```

Run the full checks before handing off a change:

```bash
pnpm test
pnpm typecheck
pnpm lint
pnpm check:native-types
pnpm verify:native-packages

cargo check --workspace --features napi-addon
cargo test --workspace --features napi-addon
cargo fmt --all -- --check
```

`pnpm test` runs the workspace Vitest suites. Backend-neutral numerical suites
always run on CPU. They also run on Metal and CUDA when those backends are
available.

Rust checks need the `napi-addon` feature because each backend gates its
Node-facing module when compiled as a normal `rlib`. VS Code configuration
enables this feature for rust-analyzer.

Do not edit `packages/**/src/internal/native-addon.d.ts` directly. Change the
Rust Node-API boundary, run `pnpm generate:native-types`, then verify the result
with `pnpm check:native-types`.

## CUDA devbox

The CUDA devbox is a billed RunPod machine. Every configured pod is user-owned,
even when it is intended to be temporary. `create` starts billing. `destroy`
permanently deletes the configured pod and ends billing.

### Configure access

Run the devbox commands from the default Nix shell:

```bash
nix develop
cp .cuda-devbox.env.example .cuda-devbox.env
runpodctl doctor
./scripts/cuda-devbox.sh key
./scripts/cuda-devbox.sh template
```

`key` creates an unencrypted Ed25519 key at `.cuda-devbox-ssh-key`, registers
its public key with RunPod, and installs it on the configured pod when one is
already running. Both key files are ignored by Git. `create` also performs this
key setup before requesting a pod.

The `CUDA devbox image` GitHub workflow publishes
`ghcr.io/mikearnaldi/effect-torch:cuda-devbox` when its dependency inputs change
on `main`. The GHCR package must be public before RunPod can pull it. Run
`template` after the first image build and after later image updates. It resolves
the image tag to an immutable digest, creates or updates a RunPod template, and
saves the template ID in the ignored `.cuda-devbox.env`.

### Create and bootstrap

Run `create` only when you intend to start a billed session:

```bash
./scripts/cuda-devbox.sh create
./scripts/cuda-devbox.sh bootstrap
```

`create` requests one RTX PRO 6000 Blackwell and saves the pod ID, SSH address,
and port. Change the GPU, cloud, disk size, or other creation settings in
`.cuda-devbox.env`. Community Cloud requires `CUDA_DEVBOX_PUBLIC_IP=1` for
direct SSH. You can use an existing pod by setting `CUDA_DEVBOX_POD_ID`,
`CUDA_DEVBOX_ADDRESS`, and `CUDA_DEVBOX_PORT`.

The image extends RunPod's pinned Ubuntu base. It contains Determinate Nix, the
`.#cuda` closure, the Rust toolchain, and warm pnpm and Cargo caches. `bootstrap`
reconciles changed lockfiles and runs the CUDA kernel, NVRTC, and cuBLASLt
checks. If no managed template exists, `create` falls back to the official
RunPod PyTorch template and `bootstrap` performs the full installation.

### Work on the pod

`show` prints the resolved configuration. `check` verifies SSH and GPU access
without changing the pod. `sync` uploads the non-ignored worktree.

```bash
./scripts/cuda-devbox.sh show
./scripts/cuda-devbox.sh check
./scripts/cuda-devbox.sh sync
./scripts/cuda-devbox.sh ssh
./scripts/cuda-devbox.sh run pnpm typecheck
```

The flake pins CUDA 12.9 and compiles the check for Blackwell `sm_120`. The
default development shell remains CUDA-free on macOS. Set
`EFFECT_TORCH_CUDA_DEVBOX_CONFIG` to use a config file outside the repository.

### End the billed session

Run `destroy` only when you intend to delete the configured pod:

```bash
./scripts/cuda-devbox.sh destroy
```

## Release build

```bash
pnpm build
```

The root build:

1. Builds the complete native release matrix.
2. Builds TypeScript for CPU, Apple, CUDA, and tokenizers.
3. Verifies native artifacts and npm tarball contents.
4. Builds `@effect-torch/core`.

The build does not implicitly run tests, typechecking, lint, or Rust tests.
Run the quality commands separately before a release build.

`pnpm build` assembles the release matrix on macOS because Apple artifacts
require the macOS SDK. It cross-compiles Linux outputs with Zig. Linux can build
and test CPU and tokenizer packages. A CUDA-capable x86_64 glibc host can also
build and test the CUDA package. Metal tests require macOS.

### Native linkage

Each backend addon is a self-contained `cdylib`. Shared Rust graph, compiler,
autodiff, runtime, and N-API helper crates are statically linked into the addon.
Addons communicate through Node-API rather than a Rust dynamic-plugin ABI.

### Native build matrix

`pnpm build` must run on macOS because Apple Metal artifacts require Xcode, the
macOS SDK, and Apple's linker tools. The command builds Darwin targets locally
and cross-compiles Linux targets with Zig:

| Artifact suffix    | Build target                     |
| ------------------ | -------------------------------- |
| `darwin-arm64`     | `aarch64-apple-darwin`           |
| `darwin-x64`       | `x86_64-apple-darwin`            |
| `linux-arm64-gnu`  | `aarch64-unknown-linux-gnu.2.17` |
| `linux-arm64-musl` | `aarch64-unknown-linux-musl`     |
| `linux-x64-gnu`    | `x86_64-unknown-linux-gnu.2.17`  |
| `linux-x64-musl`   | `x86_64-unknown-linux-musl`      |

Darwin uses Cargo and Apple's system SDK. Linux uses `cargo-zigbuild`. Darwin
artifacts target macOS 11 or newer. Release verification limits GNU artifacts
to glibc 2.17 symbols. Musl addons link dynamically against musl libc.

### Package verification

The build verifies:

- Exact package platform policy, `files` whitelist, and binary-name metadata.
- Exact native artifact sets with no missing or extra binaries.
- Artifact architecture.
- macOS deployment target and install ID.
- Absence of Nix, user-home, and Homebrew paths in Darwin linkage.
- Presence of Metal.framework in Apple artifacts.
- Absence of Metal.framework in CPU artifacts.
- Maximum glibc symbol version for GNU artifacts.
- Musl libc references and absence of glibc symbols in musl artifacts.
- Native files included by `npm pack --dry-run`.

`pnpm verify:native-packages` performs metadata and loader checks without
requiring assembled artifacts. Full artifact verification runs as part of the
release matrix build.

## Examples

```bash
pnpm --filter @effect-torch/examples xor
pnpm --filter @effect-torch/examples nano-gpt # macOS
pnpm --filter @effect-torch/examples muse-glimmer:metal # macOS
pnpm --filter @effect-torch/examples muse-glimmer:cuda # CUDA devbox
```

The examples include:

- XOR training on the CPU backend.
- Nano-GPT with tokenizer training, causal attention, RoPE, compiled training,
  paged KV-cache inference, and generation.
- FineWeb preparation from Parquet into flat token bins.
- FineWeb compiled AdamW training with restorable sampling and checkpoints.
- Mixed-BF16 full-epoch training.
- Checkpoint export and streaming generation.

## Benchmarks

```bash
pnpm bench
pnpm bench:compile
pnpm bench:inference
pnpm bench:mlx
pnpm bench:muse-glimmer
pnpm bench:muse-glimmer-concurrency

cargo bench -p effect-torch-compiler --bench pipeline
cargo bench -p effect-torch-compiler --bench pipeline -- --workload stress
```

The benchmark package covers matmul shapes, compiled programs, cold native
compilation, warm structural caches, attention, and optional MLX comparisons.
`N`, `ITERS`, and `METAL_ONLY` configure the default matmul benchmark.
`pnpm bench:compile -- --help` lists backend, workload, size, iteration, and
optimization controls. `pnpm bench:muse-glimmer` benchmarks the local
Muse-Glimmer GGUF in standard and DFlash modes at several context depths and
writes JSONL records under `bench-results/`. Set `LLAMA_CPP_BIN` to a directory
containing `llama-bench`, `llama-cli`, and `llama-speculative-simple` to include
matched llama.cpp kernel and end-to-end rows.

The Rust compiler benchmark measures `GraphIndex` plus side-table optimization
separately from graph construction and reports deterministic structural-work
counts.
Its `stress` workload runs 50,000- and 100,000-node graphs on a 256 KiB thread
stack; it does not include lowering, memory/physical planning, or pipeline
preparation.

Benchmark numbers are omitted because results depend on the hardware and software
environment. `pnpm bench` runs CPU measurements on Linux and adds
Metal when available on macOS. The MLX comparison is macOS-only.

## Repository layout

```text
packages/
  core/                    Backend-neutral TypeScript API and tests
  backend-cpu/             CPU package, adapter, loader, and artifacts
  backend-apple-native/    Apple package, adapter, loader, and artifacts
  backend-cuda/            CUDA package, adapter, loader, and artifact
  tokenizers/              TypeScript tokenizer API and Rust addon
  examples/                Runnable applications
  bench/                   Benchmarks

crates/
  runtime/                 IDs, signatures, memory, diagnostics, and ownership contracts
  graph/                   Nongeneric semantic graph and leaf contracts
  compiler/                Requests, graph index, regions, lowering tables, and memory planning
  autodiff/                Semantic graph differentiation and transforms
  napi/                    Backend-neutral Node-API helpers
  runtime-cpu/             Typed CPU executable runtime and CPU-owned addon
  runtime-cuda/            Typed CUDA executable runtime and CUDA-owned addon
  runtime-metal/           Typed Metal executable runtime and Metal-owned addon

scripts/
  build-native.mjs         Host and release-matrix native builder
  native-packages.mjs      Package and target manifest
  verify-native-packages.mjs
                            Metadata, ABI, linkage, and tarball verifier
  clean-native-declarations.mjs
                            Publish-output cleanup

docs/rfcs/                 Architecture and feature design records
```

The pnpm workspace contains seven packages plus the local OpenCode package. The
Cargo workspace contains eight shared and backend crates plus the tokenizer Rust
package.

## Design documents

The main architecture records are:

- [RFC 0026: Warning-Free Native Runtime Structure](docs/rfcs/0026-warning-free-native-runtime-structure.md)
- [RFC 0025: Target Dtype Legalization](docs/rfcs/0025-target-dtype-legalization.md)
- [RFC 0021: Compiler Pipeline Refactor](docs/rfcs/0021-compiler-pipeline-refactor.md)
- [RFC 0020: Invocation Ownership](docs/rfcs/0020-invocation-ownership.md)
- [RFC 0019: Executable Compilation](docs/rfcs/0019-executable-compilation.md)
- [RFC 0017: Multi-Backend Runtime](docs/rfcs/0017-multi-backend-runtime.md)
- [RFC 0002: Autodiff](docs/rfcs/0002-autodiff.md)
- [RFC 0003: Memory Management](docs/rfcs/0003-memory-management.md)
- [RFC 0004: Optimizers](docs/rfcs/0004-optimizers.md)
- [RFC 0005: Models](docs/rfcs/0005-models.md)
- [RFC 0007: Kernel Fusion](docs/rfcs/0007-kernel-fusion.md)
- [RFC 0008: Compilation](docs/rfcs/0008-compilation.md)
- [RFC 0009: Tokenizers](docs/rfcs/0009-tokenizers.md)
- [RFC 0010: Inference](docs/rfcs/0010-inference.md)
- [RFC 0012: Dtype System](docs/rfcs/0012-dtype-system.md)
- [RFC 0013: Batched Decode](docs/rfcs/0013-batched-decode.md)
- [RFC 0016: Frozen Program Memory](docs/rfcs/0016-frozen-program-memory.md)
