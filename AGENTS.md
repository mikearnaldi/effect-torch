# Repository guide

## Structure

- `packages/core`: backend-neutral TypeScript API for tensors, compilation, models, training, and inference.
- `packages/backend-cpu`: CPU Effect layer, TypeScript adapter, native loader, and packaged addon.
- `packages/backend-apple-native`: Metal equivalent. Builds and runtime tests require macOS.
- `packages/tokenizers`: independent TypeScript API and Rust Node addon. It exchanges host-owned `Uint32Array` values with core.
- `packages/examples` and `packages/bench`: private runnable programs and benchmarks.
- `crates/runtime`, `graph`, `compiler`, and `autodiff`: shared Rust types and graph transforms.
- `crates/napi`, `runtime-cpu`, and `runtime-metal`: Node-API helpers and backend execution.
- `scripts`: native builds, generated declarations, package checks, and CUDA devbox tools.
- `docs/rfcs`: design records. Read the relevant RFC before changing ownership, compilation, dtype, or backend contracts.

Do not edit `dist/`, `target/`, or `node_modules/`. They are generated outputs.

## Local setup

Use the pinned Nix shell and pnpm version.

```bash
direnv allow
pnpm install
```

Without direnv, run `nix develop`, then `pnpm install`. Workspace TypeScript imports source directly, but native loaders read `.node` files from each package's `dist/internal`. Build the host addons after a fresh checkout or Rust change:

```bash
pnpm --filter @effect-torch/backend-cpu build:debug
pnpm --filter @effect-torch/tokenizers build:debug
pnpm --filter @effect-torch/backend-apple-native build:debug # macOS only
```

## Commands

```bash
pnpm typecheck                 # all pnpm packages, including .opencode
pnpm lint                      # Oxlint plus dprint check
pnpm lint-fix                  # apply safe lint and formatting fixes
pnpm test                      # core, CPU, and Apple Vitest suites
pnpm build                     # complete native matrix and TypeScript build
pnpm check:native-types        # verify generated Node-API declarations
pnpm verify:native-packages    # verify native package metadata

cargo check --workspace --features napi-addon
cargo test --workspace --features napi-addon
cargo fmt --all -- --check
```

`pnpm build` does not run tests, typechecking, lint, or Rust tests. The full native release matrix requires macOS because it builds Metal artifacts and cross-compiles Linux artifacts with Zig. Prefer host debug builds during development.

Run examples with `pnpm --filter @effect-torch/examples xor` or `pnpm --filter @effect-torch/examples nano-gpt`. Run benchmarks only when the task needs measurements. The root scripts are `pnpm bench`, `pnpm bench:compile`, `pnpm bench:inference`, `pnpm bench:mlx`, and `pnpm bench:muse-glimmer`.

## Testing

Build the needed host addon first. Then run the narrowest useful test while editing:

```bash
pnpm --filter @effect-torch/core test
pnpm --filter @effect-torch/core exec vitest run test/Tensor.test.ts
pnpm --filter @effect-torch/core exec vitest run test/Tensor.test.ts -t "broadcasting"
cargo test -p effect-torch-compiler
```

Use `@effect/vitest` and `it.effect` for Effect tests. Shared core numerical suites use `test/utils/devices.ts`; they always run on CPU and run on Metal when available. Never hide unsupported Metal behavior behind a CPU fallback. Cover failure paths, resource release, and interruption when code crosses the TypeScript/native boundary.

## CUDA devbox

The devbox is a billed RunPod machine. Treat every configured pod as user-owned, even when it is intended to be disposable. These commands need the default Nix shell and configured RunPod credentials. `.cuda-devbox.env` is ignored and may contain local paths and pod state. Never commit it.

Agents are forbidden from running `./scripts/cuda-devbox.sh create` or `./scripts/cuda-devbox.sh destroy` unless the user explicitly requests that exact action in the current conversation. Permission to use, check, sync, bootstrap, or run commands on a devbox does not grant permission to create or destroy one. Never infer permission from billing, task completion, or the pod being described as disposable.

```bash
nix develop
cp .cuda-devbox.env.example .cuda-devbox.env
runpodctl doctor
./scripts/cuda-devbox.sh template   # first use, and after the image digest changes
./scripts/cuda-devbox.sh create     # create the pod and start billing
./scripts/cuda-devbox.sh bootstrap  # sync files, install deps, verify CUDA
```

During a session, use `show`, `check`, `sync`, `ssh`, or `run <command>`. `sync` uploads tracked and non-ignored files. It does not upload ignored build outputs or secrets.

```bash
./scripts/cuda-devbox.sh run pnpm typecheck
```

The wrapper has no stop command. `destroy` deletes the pod, clears its saved connection state, and ends the billed session. Run it only when the user explicitly instructs you to destroy the devbox.

## Working rules

- Keep `@effect-torch/core` independent of concrete backends. Applications choose a runtime by providing an Effect `Layer`.
- Preserve explicit ownership. Clear concrete tensor handles deterministically; finalizers are only a fallback. Keep cancellation and late-result cleanup intact.
- Match existing strict TypeScript and Effect patterns. Use inline type imports, `.ts` source imports where the package already does, and public JSDoc with `@since` and `@category`.
- Do not hand-edit `packages/**/src/internal/native-addon.d.ts`. Change the Rust Node-API boundary, run `pnpm generate:native-types`, then run `pnpm check:native-types`.
- Keep native names, targets, package file lists, and loaders aligned with `scripts/native-packages.mjs`. Run package verification after touching them.
- Add or update focused tests with behavior changes. Run targeted tests first, then `pnpm typecheck`, `pnpm lint`, and the relevant Rust checks before handing off.
