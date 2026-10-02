import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"

// Core owns the workspace tokenizer integration-test dependencies.
const require = createRequire(new URL("../../core/package.json", import.meta.url))

export default {
  root: fileURLToPath(new URL("../", import.meta.url)),
  resolve: {
    alias: { "@effect/vitest": require.resolve("@effect/vitest") }
  },
  test: { include: ["test/**/*.test.ts"] }
}
