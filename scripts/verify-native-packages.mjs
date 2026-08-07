import assert from "node:assert/strict"
import fs from "node:fs"
import path from "node:path"
import { spawnSync } from "node:child_process"
import { nativeFile, nativeFiles, nativePackages, rootDirectory } from "./native-packages.mjs"

const verifyArtifacts = process.argv.includes("--artifacts")
const unknownArgument = process.argv.slice(2).find((argument) => argument !== "--artifacts")
if (unknownArgument !== undefined) throw new Error(`unknown argument: ${unknownArgument}`)

const sorted = (values) => [...values].sort()
const commandOutput = (command, args) => {
  const result = spawnSync(command, args, { encoding: "utf8" })
  if (result.error) throw result.error
  assert.equal(result.status, 0, `${command} ${args.join(" ")} failed\n${result.stderr}`)
  return result.stdout
}
const workingDirectory = fs.realpathSync(process.cwd())
const packagesToVerify = workingDirectory === fs.realpathSync(rootDirectory)
  ? nativePackages
  : nativePackages.filter((nativePackage) => fs.realpathSync(nativePackage.directory) === workingDirectory)

assert.notEqual(
  packagesToVerify.length,
  0,
  "native package verification must run from the workspace root or a native package directory"
)

for (const nativePackage of packagesToVerify) {
  const manifestPath = path.join(nativePackage.directory, "package.json")
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"))
  const expectedNativeFiles = nativeFiles(nativePackage)
  const expectedFiles = [...nativePackage.staticFiles, ...expectedNativeFiles]

  if (nativePackage.installAnywhere === true) {
    assert.equal(manifest.os, undefined, `${nativePackage.npmName}: must be installable on every OS`)
    assert.equal(manifest.cpu, undefined, `${nativePackage.npmName}: must be installable on every architecture`)
  } else {
    assert.deepEqual(manifest.os, nativePackage.os, `${nativePackage.npmName}: incorrect os metadata`)
    assert.deepEqual(manifest.cpu, nativePackage.cpu, `${nativePackage.npmName}: incorrect cpu metadata`)
  }
  assert.deepEqual(sorted(manifest.files), sorted(expectedFiles), `${nativePackage.npmName}: incorrect files whitelist`)
  assert.equal(manifest.napi?.binaryName, nativePackage.binaryName, `${nativePackage.npmName}: incorrect binaryName`)
  assert.match(manifest.scripts?.build ?? "", /scripts\/build-native\.mjs --matrix/)
  assert.doesNotMatch(manifest.scripts?.build ?? "", /cargo|node -e/)

  const loader = fs.readFileSync(path.join(nativePackage.directory, "src/internal/native.ts"), "utf8")
  assert.match(loader, /process\.platform/, `${nativePackage.npmName}: loader does not inspect process.platform`)
  assert.match(loader, /process\.arch/, `${nativePackage.npmName}: loader does not inspect process.arch`)
  assert.ok(loader.includes(`${nativePackage.binaryName}.`), `${nativePackage.npmName}: loader has the wrong binary name`)
  if (nativePackage.os.includes("linux")) {
    assert.match(loader, /process\.report\.getReport\(\).*glibcVersionRuntime/s)
  }

  if (!verifyArtifacts) continue

  for (const file of expectedFiles) {
    const artifact = path.join(nativePackage.directory, file)
    assert.ok(fs.existsSync(artifact), `${nativePackage.npmName}: missing ${file}`)
    assert.ok(fs.statSync(artifact).size > 0, `${nativePackage.npmName}: empty ${file}`)
  }

  for (const suffix of nativePackage.targets) {
    const artifact = path.join(nativePackage.directory, nativeFile(nativePackage, suffix))
    const description = commandOutput("file", [artifact])
    if (suffix.includes("arm64")) {
      assert.match(description, /arm64|ARM aarch64/, `${nativePackage.npmName}: ${suffix} has the wrong architecture`)
    } else {
      assert.match(description, /x86_64|x86-64/, `${nativePackage.npmName}: ${suffix} has the wrong architecture`)
    }

    if (suffix.startsWith("darwin-")) {
      const dependencies = commandOutput("otool", ["-L", artifact])
      const linkedLibraries = dependencies.split("\n").slice(1).join("\n")
      assert.doesNotMatch(linkedLibraries, /\/nix\/store\//, `${nativePackage.npmName}: ${suffix} links a Nix store path`)
      assert.doesNotMatch(linkedLibraries, /\/Users\//, `${nativePackage.npmName}: ${suffix} links a user path`)
      assert.doesNotMatch(linkedLibraries, /\/opt\/homebrew\//, `${nativePackage.npmName}: ${suffix} links a Homebrew path`)
      assert.ok(
        dependencies.includes(`@rpath/${nativePackage.binaryName}.${suffix}.node`),
        `${nativePackage.npmName}: ${suffix} has a non-portable install id`
      )
      const loadCommands = commandOutput("otool", ["-l", artifact])
      assert.match(loadCommands, /minos 11\.0/, `${nativePackage.npmName}: ${suffix} does not target macOS 11`)
      if (nativePackage.binaryName === "effect-torch-backend-cpu") {
        assert.doesNotMatch(dependencies, /Metal\.framework/, `${nativePackage.npmName}: CPU addon links Metal`)
      }
      if (nativePackage.binaryName === "effect-torch-backend-apple-native") {
        assert.match(dependencies, /Metal\.framework/, `${nativePackage.npmName}: Apple addon does not link Metal`)
      }
    } else {
      const binary = fs.readFileSync(artifact).toString("latin1")
      if (suffix.endsWith("-gnu")) {
        assert.ok(binary.includes("GLIBC_"), `${nativePackage.npmName}: ${suffix} is not a GNU binary`)
        const versions = [...binary.matchAll(/GLIBC_(\d+)\.(\d+)/g)]
        assert.ok(
          versions.every((match) => Number(match[1]) < 2 || Number(match[2]) <= 17),
          `${nativePackage.npmName}: ${suffix} requires glibc newer than 2.17`
        )
      } else {
        assert.ok(binary.includes("libc.so"), `${nativePackage.npmName}: ${suffix} is not a musl binary`)
        assert.ok(!binary.includes("GLIBC_"), `${nativePackage.npmName}: ${suffix} unexpectedly references glibc`)
      }
    }
  }

  const internalDirectory = path.join(nativePackage.directory, "dist/internal")
  const actualNativeFiles = fs.readdirSync(internalDirectory)
    .filter((file) => file.endsWith(".node"))
    .map((file) => `dist/internal/${file}`)
  assert.deepEqual(
    sorted(actualNativeFiles),
    sorted(expectedNativeFiles),
    `${nativePackage.npmName}: unexpected native artifacts`
  )

  const packed = spawnSync("npm", ["pack", "--dry-run", "--json", "--ignore-scripts"], {
    cwd: nativePackage.directory,
    encoding: "utf8"
  })
  if (packed.error) throw packed.error
  assert.equal(packed.status, 0, `${nativePackage.npmName}: npm pack failed\n${packed.stderr}`)
  const packedFiles = JSON.parse(packed.stdout)[0].files.map((file) => file.path)
  const packedNativeFiles = packedFiles.filter((file) => file.endsWith(".node"))
  assert.deepEqual(
    sorted(packedNativeFiles),
    sorted(expectedNativeFiles),
    `${nativePackage.npmName}: tarball native artifacts do not match the target matrix`
  )
}

process.stdout.write(
  verifyArtifacts
    ? "Native package artifacts and npm tarball shapes are valid\n"
    : "Native package manifests and loaders match the target matrix\n"
)
