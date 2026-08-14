import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import type { NativeAddon } from "./native-addon.js"

const require = createRequire(import.meta.url)
const packageName = "@effect-torch/backend-apple-native"
let native: NativeAddon | undefined

/**
 * Lazily selects and memoizes the package-local Darwin addon for the current
 * architecture. Validation and `require` happen only on the first successful
 * call, which keeps the public package import-safe on non-Darwin hosts. A failed
 * selection/load is not cached and there is no download, search path, or CPU
 * fallback. `.node` addons use the CommonJS native-module loader even though
 * this package is ESM.
 *
 * @internal
 */
export const loadNative = (): NativeAddon => {
  if (native !== undefined) return native
  if (process.platform !== "darwin") {
    throw new Error(`${packageName} supports only platform "darwin"; received "${process.platform}"`)
  }
  if (process.arch !== "arm64" && process.arch !== "x64") {
    throw new Error(
      `${packageName} supports only architectures "arm64" and "x64"; received "${process.arch}" on "darwin"`
    )
  }

  const binary = new URL(
    `../../dist/internal/effect-torch-backend-apple-native.darwin-${process.arch}.node`,
    import.meta.url
  )
  return native = require(fileURLToPath(binary)) as NativeAddon
}
