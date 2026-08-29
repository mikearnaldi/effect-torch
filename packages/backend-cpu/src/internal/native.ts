import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import type { NativeAddon } from "./native-addon.js"

const require = createRequire(import.meta.url)
const packageName = "@effect-torch/backend-cpu"

// The public entry point evaluates this loader, so unsupported hosts and missing
// or incompatible addons fail during import. The package ships one addon for
// each supported target. It does not download, search for, or fall back to
// another backend.
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

// SAFETY: Node diagnostic reports expose header.glibcVersionRuntime, though @types/node returns object.
const report = process.report.getReport() as {
  readonly header?: { readonly glibcVersionRuntime?: string }
}
// Node's diagnostic report identifies glibc at runtime. If it has no glibc
// version, use the separately shipped musl build. Darwin artifacts need no libc
// tag.
const target = process.platform === "darwin"
  ? `darwin-${process.arch}`
  : `linux-${process.arch}-${report.header?.glibcVersionRuntime === undefined ? "musl" : "gnu"}`
const binary = new URL(
  `../../dist/internal/effect-torch-backend-cpu.${target}.node`,
  import.meta.url
)

/**
 * The package-local napi-rs addon for this Node platform, CPU architecture, and
 * Linux C library. `require` loads `.node` addons through the CommonJS native
 * module loader even though this package is ESM.
 *
 * @internal
 */
// SAFETY: the selected package binary is generated with the NativeAddon Node-API exports.
export default require(fileURLToPath(binary)) as NativeAddon
