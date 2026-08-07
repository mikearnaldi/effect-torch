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
