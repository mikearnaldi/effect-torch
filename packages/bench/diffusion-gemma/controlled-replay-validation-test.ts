import assert from "node:assert/strict"
import { validateReplayManifest } from "./controlled-replay-validation.ts"
import type { ReplayManifest } from "./controlled-replay.ts"

const canvas = Array.from({ length: 256 }, (_, i) => i)
const valid: ReplayManifest = {
  schema: "effect-torch-controlled-trajectory-v1",
  label: "controlled-trajectory-replay-not-natural-generation",
  request: { promptTokenIds: [2, 7], seed: 17, maxNewTokens: 64 },
  committedCanvasTokenIds: canvas,
  outputTokenIds: canvas.slice(0, 64),
  steps: [0, 1].map((index) => ({
    index,
    block: 0,
    step: index,
    rngDraw: index,
    temperature: 0.7,
    done: index === 1,
    canvasTokenIds: canvas,
    postCanvasTokenIds: canvas,
    draftTokenIds: canvas,
    feedbackInput: index === 0 ? null : {
      file: "feedback-000.safetensors",
      sha256: "a".repeat(64),
      dtype: "BF16",
      shape: [1, 256, 262144]
    }
  }))
}
validateReplayManifest(valid, valid.request)
const mutations: Array<(value: ReplayManifest) => void> = [
  (value) => Object.assign(value.request, { seed: 18 }),
  (value) => Object.assign(value.request, { promptTokenIds: [2, 8] }),
  (value) => Object.assign(value.request, { maxNewTokens: 63 }),
  (value) => Object.assign(value.steps[1]!, { rngDraw: 0 }),
  (value) => Object.assign(value.steps[0]!, { done: true }),
  (value) => Object.assign(value.steps[1]!, { canvasTokenIds: Array(256).fill(9) }),
  (value) => Object.assign(value.steps[1]!.feedbackInput!, { shape: [1, 1, 262144] }),
  (value) => Object.assign(value.steps[1]!.feedbackInput!, { file: "../outside" }),
  (value) => Object.assign(value.steps[1]!.feedbackInput!, { sha256: "bad" }),
  (value) => {
    value.outputTokenIds = [999]
  }
]
for (const mutation of mutations) {
  const invalid = structuredClone(valid)
  mutation(invalid)
  assert.throws(() => validateReplayManifest(invalid, valid.request))
}
process.stdout.write(JSON.stringify({ valid: 1, rejectedMalformedCases: mutations.length, status: "passed" }) + "\n")
