/**
 * Eager runtime loader for the tokenizer Node-API addon.
 *
 * Importing this module validates the published host matrix, derives one
 * package-local binary name, and loads it with `require`. There is no download,
 * source-build, WebAssembly, or pure-JavaScript fallback. Selection and loading
 * errors therefore occur during module evaluation rather than in an Effect.
 * `native-addon.ts` supplies only the compile-time shape of the loaded value.
 *
 * @internal
 */
import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import type { NativeAddon } from "./native-addon.js"

const require = createRequire(import.meta.url)
const packageName = "@effect-torch/tokenizers"

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
// Linux hosts reporting a glibc runtime select GNU; every other Linux host is
// classified as musl. The package publishes both variants for arm64 and x64.
const target = process.platform === "darwin"
  ? `darwin-${process.arch}`
  : `linux-${process.arch}-${report.header?.glibcVersionRuntime === undefined ? "musl" : "gnu"}`
// This resolves to the package's dist/internal directory both from source and
// from the compiled dist/internal/native.js. The selected file must exist and
// be loadable by the current Node-API runtime.
const binary = new URL(
  `../../dist/internal/effect-torch-tokenizers.${target}.node`,
  import.meta.url
)

/** The eagerly loaded runtime addon, typed by the private ABI declaration. @internal */
export default require(fileURLToPath(binary)) as NativeAddon
