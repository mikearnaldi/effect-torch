/** CPU-only validation before loading any controlled-trajectory GPU bank. */
import assert from "node:assert/strict"
import type { ReplayManifest } from "./controlled-replay.ts"

const tokens = (value: ReadonlyArray<number>, length?: number): void => {
  assert(Array.isArray(value))
  if (length !== undefined) assert.equal(value.length, length)
  assert(value.every((token) => Number.isInteger(token) && token >= 0 && token < 262144))
}

export const validateReplayManifest = (
  replay: ReplayManifest,
  request: ReplayManifest["request"]
): void => {
  assert.equal(replay.schema, "effect-torch-controlled-trajectory-v1")
  assert.equal(replay.label, "controlled-trajectory-replay-not-natural-generation")
  assert.deepEqual(replay.request, request)
  assert(Number.isInteger(request.seed) && request.seed >= 0 && request.seed <= 0xffffffff)
  tokens(request.promptTokenIds)
  assert(request.promptTokenIds.length > 0)
  assert(Number.isInteger(request.maxNewTokens) && request.maxNewTokens > 0 && request.maxNewTokens <= 256)
  tokens(replay.committedCanvasTokenIds, 256)
  tokens(replay.outputTokenIds)
  assert(replay.outputTokenIds.length > 0 && replay.outputTokenIds.length <= request.maxNewTokens)
  assert.deepEqual(replay.outputTokenIds, replay.committedCanvasTokenIds.slice(0, replay.outputTokenIds.length))
  assert(Array.isArray(replay.steps) && replay.steps.length > 0)
  for (const [index, step] of replay.steps.entries()) {
    assert.equal(step.index, index)
    assert.equal(step.block, 0)
    assert.equal(step.step, index)
    assert.equal(step.rngDraw, index)
    assert.equal(step.done, index === replay.steps.length - 1)
    assert(Number.isFinite(step.temperature) && step.temperature > 0)
    tokens(step.canvasTokenIds, 256)
    tokens(step.postCanvasTokenIds, 256)
    tokens(step.draftTokenIds, 256)
    assert.equal(step.feedbackInput === null, index === 0)
    if (step.feedbackInput !== null) {
      const feedback = step.feedbackInput
      assert.equal(feedback.dtype, "BF16")
      assert.deepEqual(feedback.shape, [1, 256, 262144])
      assert.match(feedback.sha256, /^[0-9a-f]{64}$/)
      assert.match(feedback.file, /^.+$/)
      assert(!feedback.file.startsWith("/") && !feedback.file.includes("\\"))
      assert(feedback.file.split("/").every((component) => component !== ".." && component !== ""))
    }
    if (index > 0) assert.deepEqual(step.canvasTokenIds, replay.steps[index - 1]!.postCanvasTokenIds)
  }
}
