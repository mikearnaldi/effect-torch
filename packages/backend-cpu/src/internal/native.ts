import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import type { NativeAddon } from "./native-addon.js"

const require = createRequire(import.meta.url)
const packageName = "@effect-torch/backend-cpu"

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
const target = process.platform === "darwin"
  ? `darwin-${process.arch}`
  : `linux-${process.arch}-${report.header?.glibcVersionRuntime === undefined ? "musl" : "gnu"}`
const binary = new URL(
  `../../dist/internal/effect-torch-backend-cpu.${target}.node`,
  import.meta.url
)

/** @internal */
export default require(fileURLToPath(binary)) as NativeAddon
