import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import type { NativeAddon } from "./native-addon.js"

const require = createRequire(import.meta.url)
const packageName = "@effect-torch/backend-cpu"

// This loader is evaluated by the public entry point, so unsupported hosts and
// missing/incompatible addons fail at import time. The package ships one exact
// addon for each supported target and intentionally has no download, search, or
// alternate-backend fallback.
if (process.platform !== "darwin" && process.platform !== "linux") {
  throw new Error(
    `${packageName} supports only platforms "darwin" and "linux"; received "${process.platform}"`
  )
}
if (process.arch !== "arm64" && process.arch !== "x64") {
  throw new Error(
    `${packageName} supports only architectures "arm64" and "x64"; received "${process.arch}" on "${process.platform}"`
  )
}

const report = process.report.getReport() as {
  readonly header?: { readonly glibcVersionRuntime?: string }
}
// Node's diagnostic report identifies glibc at runtime; absence of its version
// selects the separately shipped musl build. Darwin artifacts need no libc tag.
const target = process.platform === "darwin"
  ? `darwin-${process.arch}`
  : `linux-${process.arch}-${report.header?.glibcVersionRuntime === undefined ? "musl" : "gnu"}`
const binary = new URL(
  `../../dist/internal/effect-torch-backend-cpu.${target}.node`,
  import.meta.url
)

/**
 * The package-local napi-rs namespace selected for this Node platform, CPU
 * architecture, and Linux C library. `require` is necessary because `.node`
 * addons use the CommonJS native-module loader even though this package is ESM.
 *
 * @internal
 */
export default require(fileURLToPath(binary)) as NativeAddon
