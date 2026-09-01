import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import type { NativeAddon } from "./native-addon.js"

const require = createRequire(import.meta.url)
const packageName = "@effect-torch/backend-cuda"
let native: NativeAddon | undefined

/** Selects and caches the package-local Linux x64 glibc addon. @internal */
export const loadNative = (): NativeAddon => {
  if (native !== undefined) return native
  if (process.platform !== "linux") {
    throw new Error(`${packageName} supports only platform "linux"; received "${process.platform}"`)
  }
  if (process.arch !== "x64") {
    throw new Error(`${packageName} supports only architecture "x64"; received "${process.arch}"`)
  }
  // SAFETY: Node diagnostic reports expose header.glibcVersionRuntime, though @types/node returns object.
  const report = process.report.getReport() as {
    readonly header?: { readonly glibcVersionRuntime?: string }
  }
  if (report.header?.glibcVersionRuntime === undefined) {
    throw new Error(`${packageName} requires glibc on Linux`)
  }
  const binary = new URL(
    "../../dist/internal/effect-torch-backend-cuda.linux-x64-gnu.node",
    import.meta.url
  )
  // SAFETY: the selected binary is generated from the NativeAddon N-API exports.
  return native = require(fileURLToPath(binary)) as NativeAddon
}
