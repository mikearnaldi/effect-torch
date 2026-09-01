// Canonical native release matrix shared by build and verification scripts.
// `suffix` is the npm artifact/loader selector; `triple` identifies Cargo's
// target output directory, while `buildTarget` may request cargo-zigbuild's
// glibc-baseline target. Package `targets` are also the exact .node files placed
// on each package's npm whitelist. Keep these values aligned with the loaders;
// Linux loaders select GNU versus musl from Node's runtime glibc report.

import path from "node:path"
import { fileURLToPath } from "node:url"

export const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")

export const targets = [
  {
    suffix: "darwin-arm64",
    triple: "aarch64-apple-darwin",
    platform: "darwin",
    architecture: "arm64",
    extension: "dylib"
  },
  {
    suffix: "darwin-x64",
    triple: "x86_64-apple-darwin",
    platform: "darwin",
    architecture: "x64",
    extension: "dylib"
  },
  {
    suffix: "linux-arm64-gnu",
    triple: "aarch64-unknown-linux-gnu",
    buildTarget: "aarch64-unknown-linux-gnu.2.17",
    platform: "linux",
    architecture: "arm64",
    libc: "gnu",
    extension: "so"
  },
  {
    suffix: "linux-arm64-musl",
    triple: "aarch64-unknown-linux-musl",
    platform: "linux",
    architecture: "arm64",
    libc: "musl",
    extension: "so"
  },
  {
    suffix: "linux-x64-gnu",
    triple: "x86_64-unknown-linux-gnu",
    buildTarget: "x86_64-unknown-linux-gnu.2.17",
    platform: "linux",
    architecture: "x64",
    libc: "gnu",
    extension: "so"
  },
  {
    suffix: "linux-x64-musl",
    triple: "x86_64-unknown-linux-musl",
    platform: "linux",
    architecture: "x64",
    libc: "musl",
    extension: "so"
  }
]

export const nativePackages = [
  {
    directory: path.join(rootDirectory, "packages/backend-cpu"),
    npmName: "@effect-torch/backend-cpu",
    cargoName: "effect-torch-runtime-cpu",
    cargoFeatures: ["napi-addon"],
    crateName: "effect_torch_runtime_cpu",
    binaryName: "effect-torch-backend-cpu",
    os: ["darwin", "linux"],
    cpu: ["arm64", "x64"],
    targets: targets.map((target) => target.suffix),
    staticFiles: [
      "dist/index.js",
      "dist/index.d.ts",
      "dist/internal/adapter.js",
      "dist/internal/native.js"
    ]
  },
  {
    directory: path.join(rootDirectory, "packages/backend-apple-native"),
    npmName: "@effect-torch/backend-apple-native",
    cargoName: "effect-torch-runtime-metal",
    cargoFeatures: ["napi-addon"],
    crateName: "effect_torch_runtime_metal",
    binaryName: "effect-torch-backend-apple-native",
    installAnywhere: true,
    os: ["darwin"],
    cpu: ["arm64", "x64"],
    targets: targets.filter((target) => target.platform === "darwin").map((target) => target.suffix),
    staticFiles: [
      "dist/index.js",
      "dist/index.d.ts",
      "dist/internal/adapter.js",
      "dist/internal/native.js"
    ]
  },
  {
    directory: path.join(rootDirectory, "packages/backend-cuda"),
    npmName: "@effect-torch/backend-cuda",
    cargoName: "effect-torch-runtime-cuda",
    cargoFeatures: ["napi-addon"],
    crateName: "effect_torch_runtime_cuda",
    binaryName: "effect-torch-backend-cuda",
    installAnywhere: true,
    os: ["linux"],
    cpu: ["x64"],
    targets: ["linux-x64-gnu"],
    staticFiles: [
      "dist/index.js",
      "dist/index.d.ts",
      "dist/internal/adapter.js",
      "dist/internal/native.js"
    ]
  },
  {
    directory: path.join(rootDirectory, "packages/tokenizers"),
    npmName: "@effect-torch/tokenizers",
    cargoName: "effect-torch-tokenizers",
    crateName: "effect_torch_tokenizers",
    binaryName: "effect-torch-tokenizers",
    os: ["darwin", "linux"],
    cpu: ["arm64", "x64"],
    targets: targets.map((target) => target.suffix),
    staticFiles: [
      "dist/index.js",
      "dist/index.d.ts",
      "dist/internal/native.js"
    ]
  }
]

export const nativeFile = (nativePackage, suffix) =>
  `dist/internal/${nativePackage.binaryName}.${suffix}.node`

export const nativeFiles = (nativePackage) =>
  nativePackage.targets.map((suffix) => nativeFile(nativePackage, suffix))
