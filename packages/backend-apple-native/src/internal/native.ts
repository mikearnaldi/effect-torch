import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import type { NativeAddon } from "./native-addon.js"

const require = createRequire(import.meta.url)
const packageName = "@effect-torch/backend-apple-native"
let native: NativeAddon | undefined

/**
 * Selects and caches the package-local Darwin addon for the current architecture.
 * Until an addon loads, each call validates the platform and architecture. The
 * first successful call uses `require` and caches the result. Importing the public
 * package does not call this loader, so imports do not fail on non-Darwin hosts.
 * Failed selection or loading is not cached. The loader does not download
 * binaries, search other paths, or fall back to CPU. Although this package is
 * ESM, `.node` addons use the CommonJS native-module loader.
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
  // SAFETY: the selected package binary is generated with the NativeAddon Node-API exports.
  return native = require(fileURLToPath(binary)) as NativeAddon
}
