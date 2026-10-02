# DiffusionGemma CUDA optimization — measured goal achieved

## Current result

Both requested matched end-to-end median targets have been measured below vLLM.
Final combined receipt validation passed. Goal achieved for this benchmark scope;
do not resume optimization without a new user request.

| Prompt / output / concurrency | Effect median | Fresh matched vLLM | Reduction |
| ----------------------------- | ------------: | -----------------: | --------: |
| 32 / 64 / 1                   | 552.562360 ms |      554.558223 ms |   0.3599% |
| 128 / 64 / 1                  | 530.969076 ms |      532.181951 ms |   0.2279% |

Small measured margins, not a statistical performance guarantee. Effect was
faster in 5/5 individual 32-token cases and 2/5 individual 128-token cases.
The goal is both medians, not every individual case.

Same sealed seven-case bank, two warmups and five measured cases per target.
Both backends perform actual model and sampler computations along identical
canonical generation trajectories. Timing includes prefill and the full
256-token terminal commit; tokenization and initialization are excluded.
Natural generation quality is checked separately.

## Validated candidate

Use the **release build with 97, 98, 100 and 101 enabled**, plus the inherited
BF16, packed projection, sampler, attention and rotary reuse flags in the pinned
entry. User explicitly permits vLLM-style BF16 arithmetic; original bitwise
oracle and historical 502/473 ms thresholds are superseded by matched work.

- Native source archive: `54979d50b5cb2e5ae68ece16540d26150957210dc42c4cbfb74508067a3e337e`.
- Release addon: `fab05c99fb5158489345af782f1fdbc53f8630e0ad723dda6e86b399e91a8be4`.
- Release runner archive: `b0f8bbfec8176d2aafe18bcbebf3cccb4908e5146246daa8cd1ee5fb6f660c7d`.
- Release entry archive: `69809aa47b5bdf6361e8e0c33821a3b4040ac483c3b2be4e1a3b891df69ab599`.
- Release profile: opt-level3, LTO, codegen-units1, debug/assertions/overflow checks
  off, panic unwind, incremental off. Explicit profile receipts required.
- Corrected101 helper PTX: `c3984b2208ee1cdeb4094fb975be3136b2cbe604dafbb744f6fad0bd0a9f5e33`.

Remote build: `/root/combined97-100-101-release-build-v1`.
Remote entry: `/root/combined97-100-101-release-entry-v1`.
Entry archives are flat: extract inside a newly created destination directory.

## Correctness and evidence

- 230 native host tests passed in release mode.
- Seven combined GPU gates passed: native97 parity; native101 loader, compiled
  lifecycle, lazy KV schedule and actual-model table schedule; native100 loader
  and compiled lifecycle.
- Three supplemental native98 release gates passed: loader, paired lifecycle
  and stateful instruction mapping.
- Natural quality: 12/12 tasks; exact outputs and refinement traces versus the
  corresponding debug candidate. This is bounded smoke coverage.
- Native97 executed all31 natural refinements;12 request-private samplers.
- Native101:25 groups/50 aliases at T256, zero at other widths.
- Native100: exactly one self-conditioning refinement graph.
- Native98:29 pairs in Append,30 in ReadOnly, no norm63 fallback.
- Benchmark runners restored all8 installed files and removed private imports.
  The installed baseline remains intact; the release candidate is reproducibly
  selected by the guarded runner, not permanently installed over it.

Remote final runs:

- `/root/combined97-98-100-101-release-natural-v1`
- `/root/combined97-98-100-101-release-timing32-v1`
- `/root/combined97-98-100-101-release-timing128-v1`
- `/root/combined97-100-101-release-build-v1/hardware/assessment.json`
- `/root/combined97-100-101-release-build-v1/hardware-norm98-v1/assessment.json`

Local final evidence:
`bench-results/diffusion-gemma-20260930/combined97-100-101-release-evidence-v1/accepted-norm98/`.
Shared build receipts are in the parent `build/` directory. Preserve raw records,
comparison receipts, provenance and restoration receipts, including failures.

Final receipt: `accepted-norm98/measured-goal.json`, SHA256
`ef7c08a0c72af2dc5a694c08296f2e5f7467985f5221ee6b3c6d3a65a13622a3`.
`ACCEPTANCE-README.txt` contains exact run commands and the independent validator.
Final workspace TypeScript typecheck and lint passed; touched native files passed
formatting checks and release native tests as listed above.

## Reproduction

Use the pinned Nix shell on a fresh matching GPU machine after restoring the
verified external backup. See `packages/bench/diffusion-gemma/reproduction/README.txt`
for restoration and CPU evidence verification. Choose fresh output directories.
The entry checks source, addon, helper images, quality, release profile and
installed baseline before changing anything; it restores installed files after
execution. Root schedules remote workloads sequentially.

```bash
nix develop --command ./scripts/cuda-devbox.sh run python3 \
  /root/combined97-100-101-release-entry-v1/run.py timing32 \
  --norm98 --refresh-vllm \
  --quality /root/combined97-98-100-101-release-natural-v1 \
  --baseline-effect /root/combined97-98-100-101-release-timing32-v1/prompt-32/effect-combined97-100-101-release-v1.jsonl \
  --output /root/combined101-recheck32
```

For128 change `timing32` to `timing128`, use the corresponding timing128/prompt-128
baseline, and a fresh output directory. Do not compare unmatched stopping policies
or omit the actual terminal commit.

## Important corrections and parked work

Recent isolated build runners accidentally omitted the package debug script's
`CARGO_PROFILE_DEV_OPT_LEVEL=2` and produced unoptimized Rust addons. Correcting
this with the identical-source release build reduced the 32-token median from
569.548225 to557.205200ms before enabling98. Earlier release68 experiments were
release versus optimized dev level2, so did not cover this regression.

Native101 originally admitted zero real groups. Fixed structural repeated-half
proof for full-width tables, normalized Q/K alias equivalence, and valid V RMS
operation bits0..3. Actual-model stateful fixture now exercises all three.
Full-table component passed16 proofs/33 formula records; complete pipeline
23.30→12.10us. Do not quote the obsolete compact-table63% component gain.

Local profiling draft `combined101-profile-v1` is parked. Global-layer fusion and
shared table preparation are unimplemented proposals; no savings claims.

## Infrastructure and authorization

Local `/home/michaelarnaldi/effect-torch`; remote `/root/effect-torch`.
Former RunPod `ctdfyl5lyv09eg`, RTX PRO6000 Blackwell96GB, is **destroyed**.
RunPod acknowledged deletion; the follow-up query returned404 `pod not found`.
Paths below refer to the preserved archive, not a live machine.
Model `/root/models/diffusiongemma`; oracle `/root/oracle-eager-20260930`;
initialized RoPE `initialized-state/initialized-rope.safetensors`;
sealed bank `/root/native66-fresh-distinct-v1`.
Shared Cargo target `/root/whole-read71-build-v1/target`.

The latest user request explicitly authorizes destroying this devbox after preserving
reproduction assets, committing the complete work on a branch, pushing and creating
a PR. This supersedes the earlier no-commit/no-destruction handoff. Credentials
and generated large dependencies remain outside Git. Never commit env files.

Branch: `perf/diffusion-gemma-bf16-reproduction`. Implementation commit: `52c290b`.
The branch is pushed; final preservation receipts accompany the PR.

Backup: `bench-results/devbox-archive-20261001/remote-root.tar.zst`, with full
`inventory.jsonl`, environment/toolchain/Nix registration metadata and a separate
`local-history.tar.zst`. All296,044 entries and236,879 regular files passed
size/hash verification before deletion; all123 required guard hashes matched.
The archive preserves59,118,287,624 uncompressed file bytes. Large assets remain
outside Git; model weights are downloaded at the pinned revision and checked
against preserved hashes. Keep the complete backup directory.

Committed reproduction tools and receipts:
`packages/bench/diffusion-gemma/reproduction/README.txt`,
`external-backup.json`, `teardown.json` and `worktree-validation.json`.

Earlier history, including all original failed attempts and artifact references,
is archived in `bench-results/diffusion-gemma-20260930/continuation-history-through101-release-v1.md`
and the prior through81/through87 history files.
