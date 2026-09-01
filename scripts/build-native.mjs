// Assembles native .node artifacts without generating JavaScript bindings.
// It refreshes private TypeScript declarations from napi-rs metadata before
// compiling the selected native packages.
// `--host` must run from one native package directory; it removes non-preserved
// dist outputs, retains configured non-host matrix binaries, then builds and
// copies the selected host/profile artifact. `--matrix` runs from the workspace
// root or one package directory, clears the selected package dist contents,
// cross-builds configured release targets, copies/renames shared
// libraries to loader-selected .node names, and requires the complete selected
// matrix at the end. Darwin targets require macOS; Linux-only package matrices
// can run on Linux. TypeScript build and package verification are separate package-script stages.

import fs from "node:fs"
import path from "node:path"
import { spawnSync } from "node:child_process"
import { nativeFile, nativeFiles, nativePackages, rootDirectory, targets } from "./native-packages.mjs"

const fail = (message) => {
  throw new Error(message)
}

const run = (command, args, environment = {}) => {
  const env = { ...process.env }
  for (const [name, value] of Object.entries(environment)) {
    if (value === undefined) delete env[name]
    else env[name] = value
  }
  const result = spawnSync(command, args, {
    cwd: rootDirectory,
    env,
    stdio: "inherit"
  })
  if (result.error) fail(`could not start ${command}: ${result.error.message}`)
  if (result.status !== 0) fail(`${command} ${args.join(" ")} failed with exit code ${result.status}`)
}

const darwinBuildEnvironment = (target) => {
  // Avoid inheriting a Nix/Xcode selection into release artifacts: pin the
  // system macOS SDK/toolchain and deployment target for both Darwin triples.
  const env = { ...process.env }
  delete env.DEVELOPER_DIR
  const sdk = spawnSync("/usr/bin/xcrun", ["--sdk", "macosx", "--show-sdk-path"], {
    encoding: "utf8",
    env
  })
  if (sdk.error || sdk.status !== 0) {
    fail("Darwin builds require the macOS command line tools and SDK")
  }
  const targetName = target.triple.toUpperCase().replaceAll("-", "_")
  const targetVariable = target.triple.replaceAll("-", "_")
  return {
    AR: "/usr/bin/ar",
    CC: "/usr/bin/clang",
    CXX: "/usr/bin/clang++",
    DEVELOPER_DIR: undefined,
    MACOSX_DEPLOYMENT_TARGET: "11.0",
    NIX_CFLAGS_COMPILE: undefined,
    NIX_LDFLAGS: undefined,
    SDKROOT: sdk.stdout.trim(),
    [`AR_${targetVariable}`]: "/usr/bin/ar",
    [`CC_${targetVariable}`]: "/usr/bin/clang",
    [`CXX_${targetVariable}`]: "/usr/bin/clang++",
    [`CARGO_TARGET_${targetName}_LINKER`]: "/usr/bin/clang"
  }
}

const assertFile = (file) => {
  if (!fs.existsSync(file) || fs.statSync(file).size === 0) {
    fail(`expected native artifact was not produced: ${path.relative(rootDirectory, file)}`)
  }
}

const cleanPackage = (nativePackage, preserve) => {
  const dist = path.join(nativePackage.directory, "dist")
  if (!fs.existsSync(dist)) return

  for (const entry of fs.readdirSync(dist)) {
    const entryPath = path.join(dist, entry)
    if (entry !== "internal") {
      fs.rmSync(entryPath, { recursive: true, force: true })
      continue
    }

    for (const internalEntry of fs.readdirSync(entryPath)) {
      if (!preserve.has(internalEntry)) {
        fs.rmSync(path.join(entryPath, internalEntry), { recursive: true, force: true })
      }
    }
  }
}

const copyArtifact = (nativePackage, target, sourceDirectory) => {
  const source = path.join(sourceDirectory, `lib${nativePackage.crateName}.${target.extension}`)
  const destination = path.join(nativePackage.directory, nativeFile(nativePackage, target.suffix))
  assertFile(source)
  fs.mkdirSync(path.dirname(destination), { recursive: true })
  fs.copyFileSync(source, destination)
  if (target.platform === "darwin") {
    run("install_name_tool", ["-id", `@rpath/${path.basename(destination)}`, destination])
  }
  assertFile(destination)
  process.stdout.write(`Copied ${path.relative(rootDirectory, destination)}\n`)
}

const addPackageArguments = (args, nativePackage) => {
  args.push("--package", nativePackage.cargoName)
  if (nativePackage.cargoFeatures !== undefined) {
    args.push("--no-default-features", "--features", nativePackage.cargoFeatures.join(","))
  }
}

const generateDeclarations = (packages) => {
  for (const nativePackage of packages) {
    run(process.execPath, [
      path.join(rootDirectory, "scripts/generate-native-declarations.mjs"),
      "--package",
      nativePackage.npmName
    ])
  }
}

const targetForHost = (nativePackage) => {
  if (!nativePackage.os.includes(process.platform)) {
    fail(
      `${nativePackage.npmName} cannot be built on platform "${process.platform}"; supported platforms: ${nativePackage.os.join(
        ", "
      )}`
    )
  }
  if (!nativePackage.cpu.includes(process.arch)) {
    fail(
      `${nativePackage.npmName} cannot be built for architecture "${process.arch}"; supported architectures: ${nativePackage.cpu.join(
        ", "
      )}`
    )
  }

  // Match the runtime loaders: absence of glibcVersionRuntime selects musl.
  const libc = process.platform === "linux"
    ? process.report.getReport().header.glibcVersionRuntime === undefined ? "musl" : "gnu"
    : undefined
  const suffix = [process.platform, process.arch, libc].filter(Boolean).join("-")
  const target = targets.find((candidate) => candidate.suffix === suffix)
  if (target === undefined || !nativePackage.targets.includes(target.suffix)) {
    fail(`no native target is configured for ${nativePackage.npmName} on ${suffix}`)
  }
  return target
}

const buildHost = (profile) => {
  const directory = fs.realpathSync(process.cwd())
  const nativePackage = nativePackages.find((candidate) => fs.realpathSync(candidate.directory) === directory)
  if (nativePackage === undefined) {
    fail("--host must be run from a native package directory")
  }

  const target = targetForHost(nativePackage)
  generateDeclarations([nativePackage])
  const currentFile = path.basename(nativeFile(nativePackage, target.suffix))
  const preserve = new Set(
    nativeFiles(nativePackage).map((file) => path.basename(file)).filter((file) => file !== currentFile)
  )
  cleanPackage(nativePackage, preserve)

  const args = ["build", "--locked"]
  addPackageArguments(args, nativePackage)
  if (profile === "release") args.push("--release")
  run("cargo", args)
  copyArtifact(nativePackage, target, path.join(rootDirectory, "target", profile))
}

const packagesForWorkingDirectory = () => {
  const directory = fs.realpathSync(process.cwd())
  if (directory === fs.realpathSync(rootDirectory)) return nativePackages
  const nativePackage = nativePackages.find((candidate) => fs.realpathSync(candidate.directory) === directory)
  if (nativePackage !== undefined) return [nativePackage]
  fail("matrix builds must run from the workspace root or a native package directory")
}

const preflightMatrix = (packages) => {
  const selectedTargets = targets.filter((target) =>
    packages.some((nativePackage) => nativePackage.targets.includes(target.suffix))
  )
  if (process.platform !== "darwin" && selectedTargets.some((target) => target.platform === "darwin")) {
    fail(`native matrices containing Darwin targets must run on macOS; current platform is "${process.platform}"`)
  }

  if (!selectedTargets.some((target) => target.platform === "linux")) return

  const result = spawnSync("cargo", ["zigbuild", "--help"], {
    cwd: rootDirectory,
    encoding: "utf8"
  })
  if (result.error || result.status !== 0) {
    fail("native matrix builds require cargo-zigbuild; install it with `cargo install cargo-zigbuild`")
  }
}

const buildMatrix = () => {
  const selectedPackages = packagesForWorkingDirectory()
  preflightMatrix(selectedPackages)
  generateDeclarations(selectedPackages)
  for (const nativePackage of selectedPackages) cleanPackage(nativePackage, new Set())

  for (const target of targets) {
    const packagesForTarget = selectedPackages.filter((nativePackage) => nativePackage.targets.includes(target.suffix))
    if (packagesForTarget.length === 0) continue
    // Darwin uses Cargo plus the pinned SDK. Linux cross-builds use zigbuild;
    // GNU buildTarget values enforce glibc 2.17, while musl leaves crt-static
    // disabled so the loader can use the platform's dynamic musl runtime.
    const command = target.platform === "linux" ? "zigbuild" : "build"
    const environment = target.platform === "darwin"
      ? darwinBuildEnvironment(target)
      : target.libc === "musl"
      ? { RUSTFLAGS: "-C target-feature=-crt-static" }
      : {}
    const sourceDirectory = path.join(rootDirectory, "target", target.triple, "release")
    for (const nativePackage of packagesForTarget) {
      const args = [command, "--locked", "--release", "--target", target.buildTarget ?? target.triple]
      addPackageArguments(args, nativePackage)
      run("cargo", args, environment)
      copyArtifact(nativePackage, target, sourceDirectory)
    }
  }

  for (const nativePackage of selectedPackages) {
    for (const file of nativeFiles(nativePackage)) assertFile(path.join(nativePackage.directory, file))
  }
  process.stdout.write("Native release matrix assembled successfully\n")
}

try {
  const matrix = process.argv.includes("--matrix")
  const host = process.argv.includes("--host")
  const profileIndex = process.argv.indexOf("--profile")
  const profile = profileIndex === -1 ? "release" : process.argv[profileIndex + 1]
  const knownArguments = new Set(["--matrix", "--host", "--profile", "release", "debug"])
  const unknownArgument = process.argv.slice(2).find((argument) => !knownArguments.has(argument))

  if (unknownArgument !== undefined) fail(`unknown argument: ${unknownArgument}`)
  if (matrix === host) fail("choose exactly one of --host or --matrix")
  if (profile !== "release" && profile !== "debug") fail(`unsupported profile: ${profile}`)
  if (matrix && profile !== "release") fail("native matrix builds use the release profile")

  if (matrix) buildMatrix()
  else buildHost(profile)
} catch (error) {
  process.stderr.write(`native build: ${error instanceof Error ? error.message : String(error)}\n`)
  process.exitCode = 1
}
