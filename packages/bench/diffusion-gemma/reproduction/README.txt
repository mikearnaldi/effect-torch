Accepted DiffusionGemma reproduction bundle

The release candidate with native97 processing, norm98, softmax100 and normrope101 beat the fresh matched vLLM median on both measured prompt targets. Prompt32: 552.562360 ms versus 554.558223 ms (-0.359901%). Prompt128: 530.969076 ms versus 532.181951 ms (-0.227906%). Natural smoke quality passed 12/12. These are small measured median wins: prompt128 won 2/5 paired cases, so this is not a statistical confidence or universal speed claim.

Tracked assets total approximately 1.8 MB. They contain source and evidence, not native binaries, model weights, credentials or generated build directories:
- archives/combined97-100-101-release-runner-v1.tar.gz preserves the release build/validation scripts and the exact 290-file source snapshot. Its nested source-v1.tar.gz SHA256 is 54979d50b5cb2e5ae68ece16540d26150957210dc42c4cbfb74508067a3e337e.
- archives/combined97-100-101-release-entry-v1.tar.gz preserves the guarded eight-file transactional swap, processed native97 replay adaptation, coverage observer, pinned module dependencies, natural scorer and matched timing runner.
- archives/accepted-release-norm98-evidence.tar.gz preserves the raw accepted natural/timing JSONL, quality/coverage, build and hardware receipts, provenance/restoration receipts, immutable measured-goal receipt, original CPU validator and exact commands. JSONL and JSON are archived without reformatting to preserve their recorded hashes.
- manifest.json pins all archives. reproduce.py verifies/unpacks them without network or GPU access.

CPU verification from the repository root:

nix develop --command python3 packages/bench/diffusion-gemma/reproduction/reproduce.py verify --output /tmp/effect-torch-measured-goal.json

Use a fresh output path. The validator recomputes the strict workload comparison and medians from the raw rows, verifies matching release/source/candidate/flags and quality hashes, all eight release checks (host plus seven GPU) and three supplemental norm98 GPU gates, actual graph coverage, and complete restoration. It requires both targets and will not treat absent evidence as success. The original validator checks the comparator source hash; do not replace the comparator with a different implementation to bypass that guard.

Unpack immutable sources and evidence into a fresh staging directory:

nix develop --command python3 packages/bench/diffusion-gemma/reproduction/reproduce.py unpack --destination /tmp/effect-torch-reproduction

This creates combined97-100-101-release-build-v1/, combined97-100-101-release-entry-v1/, and accepted-evidence/. The accepted native addon SHA256 is fab05c99fb5158489345af782f1fdbc53f8630e0ad723dda6e86b399e91a8be4. The original build uses Rust 1.93.1 and explicit release optimization: opt-level 3, LTO, one codegen unit, no debug assertions/overflow checks/debug/incremental compilation, panic unwind. Building default cargo dev mode changes host runtime performance and does not reproduce the accepted candidate.

GPU reproduction requires the external hashed dependency backup, the same supported CUDA environment, a single RTX PRO 6000 Blackwell 96 GB GPU, and the authorized model snapshot. The frozen runner intentionally uses absolute /root paths. Restore its dependencies at those original /root paths; an arbitrary path relocation without regenerated reviewed pins is not supported. Staging elsewhere is safe for CPU inspection only. Do not source or copy credential files from a backup.

The working /root/effect-torch tree must match every installed baseline/source guard before the runner swaps in the candidate. The final development branch is not itself that installed baseline. Restore the exact baseline tree and native addon from the dependency backup before executing the frozen entry. The eight replacements are transactional and restored afterward. Run no concurrent GPU process or workspace mutation during a measurement.

Place the unpacked release build and entry directories directly under /root. Restore all external dependencies before build or execution. The build script extracts its immutable source archive into a fresh source-v1 directory; do not pre-extract source-v1 there. On a supported machine:

python3 /root/combined97-100-101-release-build-v1/build.py --target-dir /root/whole-read71-build-v1/target
python3 /root/combined97-100-101-release-build-v1/validate.py

The entry requires the exact accepted addon hash, all eight release checks and the original dependency hashes. Rebuilding under a changed toolchain may produce a different binary hash; that requires a new reviewed candidate entry and fresh quality/hardware/timing evidence rather than editing the accepted receipt. The external backup retains the accepted addon for exact restoration.

The supplemental normalization98 validator and its original source, smoke-task/oracle generation programs, initialized state, PTX sources/modules, canonical seven-case feedback banks and model revision must also be retained in the external dependency closure. Their complete inventory and restore procedure are supplied with the backup; no claim of standalone GPU reproducibility is made by these small tracked archives alone.

After release and supplemental norm98 hardware gates pass, use accepted-evidence/ACCEPTANCE-README.txt for exact accepted natural/timing commands. Natural and both timing runs require --norm98, and timing requires the matching natural --quality directory plus --refresh-vllm. Use fresh output paths. The sealed bank includes two warmups and five measured cases per target. Effect executes first, followed by its fresh vLLM control; the full prefill/refinement/terminal-commit boundary is retained and diagnostics remain disabled during measurement.

Pod creation or destruction is not performed by this bundle. Use the repository's CUDA devbox instructions and the current user's explicit authorization for lifecycle actions.

External archive restore and dependency audit

The archive is retained outside Git at bench-results/devbox-archive-20261001/remote-root.tar.zst, alongside its full per-file inventory, metadata and checksum. external-backup.json records its final size and SHA256 after the owner's completed download verification. Keep that archive on durable storage with this branch; the tracked source/evidence bundle alone cannot replace the approximately 26 GB canonical feedback bank, exact initialized oracle state, cached PTX/SO modules and installed baseline.

Verify and extract into a fresh staging root:
python3 packages/bench/diffusion-gemma/reproduction/external-assets.py verify-archive --archive /path/to/remote-root.tar.zst
python3 packages/bench/diffusion-gemma/reproduction/external-assets.py restore --archive /path/to/remote-root.tar.zst --destination /fresh/restore-root
python3 packages/bench/diffusion-gemma/reproduction/external-assets.py audit-root --root-prefix /fresh/restore-root

Archive members preserve root/... and nix/store/... paths. The restore helper is create-only and does not install into an existing /root automatically. Inspect the staging tree, then restore its /root and /nix/store hierarchies on a dedicated fresh machine. Absolute symlinks and ELF search paths may only resolve after installation at the original paths. Run audit-root --root-prefix / again after installation. Verify the backup's complete per-file inventory as well; the small guard audit is necessary but not a substitute for checking every bank payload.

Recreate omitted node_modules with the pinned Nix shell and pnpm lockfile from the restored baseline. Recreate omitted Cargo targets only when rebuilding is needed. Inspect retained environment metadata and ELF dependency records for every native addon, libexpert-flashinfer73.so, cached fused_moe_120.so and libtvm_ffi.so. Restore captured nonstandard libraries and the Python/uv interpreter targets; Python virtualenv directories alone do not include their interpreter or Nix store closure. The baseline GPU runtime must match its recorded driver/CUDA environment.

Model weights are excluded from the archive. Download google/diffusiongemma-26B-A4B-it at revision f7f5b7f5fa82ffc52addd066915886d497f5517b using the retained packages/bench/diffusion-gemma/reference.py download command and the saved Hugging Face LFS SHA256/size inventory. Credentials are supplied separately by the operator. Restore metadata to /root/models/diffusiongemma and verify every weight hash before running. reference.sh setup pins Torch 2.10.0 CUDA128 and Transformers revision 93ebf6b11127967f2725cf4d012aae55c3654f5a for oracle generation; the archived initialized state and bank should be used for exact accepted reproduction.

generation-sources.json indexes the checked source programs for model/oracle download, controlled replay and kernel exports. The exact bank prepare/capture program also lives inside the sealed entry archive. Regenerating state or feedback trajectories is useful for a new experiment but changes the accepted evidence and requires fresh pins, quality and comparisons. Cached native98/100/101 PTX bytes and the expert bridge dependencies must come from the verified archive for exact reproduction.

Complete streaming inventory verification (no extraction space required)

python3 packages/bench/diffusion-gemma/reproduction/verify-inventory.py --archive /path/to/remote-root.tar.zst --inventory /path/to/inventory.jsonl --output /fresh/inventory-verification.json

The inventory has one JSON line per relativePath: regular files include sizeBytes and sha256, symlinks include type="symlink" and linkTarget, and directories include type="directory". The verifier streams zstd into tar, hashes every regular payload, resolves hardlinks only through previously verified members, and rejects missing, extra, duplicate, unsupported or mismatched members. Its create-only JSON report records counts, payload/logical bytes and all errors. A nonzero exit means the archive must not be treated as verified. Small positive/negative fixtures run with verify-inventory-test.py; they never read the external backup.

The full external archive also preserves nix/store/... alongside root/.... Preserve that hierarchy on the replacement machine: the retained Python/native ELF files resolve their absolute /nix/store dependencies using those bytes. Nix store contents and Nix database registration are distinct. If registration was not included or is incompatible with the replacement installation, restore/import registration separately as appropriate; recreating the pinned Nix shell from flake.nix/flake.lock lets Nix rebuild or download and register the required store paths. Do not remove the retained store bytes before the absolute ELF/interpreter closure has been verified. The archive is substantially larger than the bank alone because it retains this runtime closure.

Concatenated archive streams

The completed remote-root.tar.zst may contain multiple concatenated zstd frames, each wrapping a tar stream. The supplemental stream restores immutable Nix store entries that the initial cache exclusion omitted. The merged inventory covers every stream. Both the streaming verifier and external-assets.py restore continue past tar zero padding; do not extract only the first stream.

For manual restoration with GNU tar, --ignore-zeros is mandatory:
tar --zstd --extract --ignore-zeros --file /path/to/remote-root.tar.zst --directory /fresh/restore-root --no-same-owner --keep-old-files

Verify the final whole-archive checksum and merged inventory before extraction. concatenated-archive-test.py exercises two tiny zstd/tar frames through both the actual verifier and checksum-gated restore helper, including a supplemental nix/store/.../node_modules entry. It does not read or modify the real backup.

Preserved environment companions

The backup directory includes environment.json (GPU/driver, ELF dependency reports, Python package versions and model weight hashes), toolchain.txt, container-image.txt, and nix-registration.txt. Their byte sizes and SHA256 values are recorded in external-backup.json. After restoring the complete /nix/store on a dedicated matching machine, the saved registration can be imported with `nix-store --load-db < /path/to/nix-registration.txt`; review compatibility with the installed Nix version first. Nine system-library snapshots under usr/... are retained for dependency inspection; use the recorded container image rather than blindly replacing libraries on an unrelated host.

The separately checksummed local-history.tar.zst contains the prior local diffusion-gemma-20260930 evidence/history directory. It is supplementary to the remote-root archive and is not needed by the exact final runner. Keep the complete backup directory together; its large archives are local assets outside Git.
