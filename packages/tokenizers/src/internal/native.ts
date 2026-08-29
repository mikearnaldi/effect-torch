/**
 * Runtime loader for the tokenizer Node-API addon.
 *
 * Importing this module checks the supported platforms and architectures. It
 * chooses a package-local binary and loads it with `require`. It does not
 * download or build a binary. It has no WebAssembly or pure-JavaScript fallback.
 * Selection and loading errors occur during module evaluation, outside an
 * Effect.
 * The generated `native-addon.d.ts` declares the loaded value's TypeScript type.
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

// SAFETY: Node diagnostic reports expose header.glibcVersionRuntime, though @types/node returns object.
const report = process.report.getReport() as {
  readonly header?: { readonly glibcVersionRuntime?: string }
}
// Linux hosts that report a glibc runtime use GNU. Other Linux hosts use musl.
// The package includes both variants for arm64 and x64.
const target = process.platform === "darwin"
  ? `darwin-${process.arch}`
  : `linux-${process.arch}-${report.header?.glibcVersionRuntime === undefined ? "musl" : "gnu"}`
// This resolves to the package's dist/internal directory from both source and
// compiled dist/internal/native.js. The file must exist, and the current
// Node-API runtime must be able to load it.
const binary = new URL(
  `../../dist/internal/effect-torch-tokenizers.${target}.node`,
  import.meta.url
)

/** Runtime addon loaded during import and typed by the private ABI declaration. @internal */
// SAFETY: the selected package binary is generated with the NativeAddon Node-API exports.
export default require(fileURLToPath(binary)) as NativeAddon
