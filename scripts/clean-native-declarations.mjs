// Post-processes one native package's TypeScript build output. Generated napi-rs
// declarations are compile-time inputs only: package entry declarations expose
// the supported public API, while internal .d.ts files and any stale
// native-addon.js stub must not ship beside the hand-written runtime loader.
// This script intentionally operates only from a native package directory.

import fs from "node:fs"
import path from "node:path"
import { nativePackages } from "./native-packages.mjs"

const directory = fs.realpathSync(process.cwd())
const nativePackage = nativePackages.find((candidate) => fs.realpathSync(candidate.directory) === directory)

if (nativePackage === undefined) {
  throw new Error(
    "native declaration cleanup must run from packages/backend-cpu, packages/backend-apple-native, or packages/tokenizers"
  )
}

const internalDirectory = path.join(nativePackage.directory, "dist/internal")
if (fs.existsSync(internalDirectory)) {
  for (const file of fs.readdirSync(internalDirectory)) {
    if (file.endsWith(".d.ts") || file === "native-addon.js") {
      fs.rmSync(path.join(internalDirectory, file))
    }
  }
}
