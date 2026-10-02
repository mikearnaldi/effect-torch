/** Benchmark-only forced trajectory replay. This is not natural generation. */
import { Diffusion, type Runtime, Safetensors, Tensor } from "@effect-torch/core"
import { Effect, Exit, type Scope } from "effect"
import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { readFile, writeFile } from "node:fs/promises"
import { join } from "node:path"

export interface FeedbackFile {
  readonly file: string
  readonly sha256: string
  readonly dtype: "BF16"
  readonly shape: ReadonlyArray<number>
}
export interface ReplayStep {
  readonly index: number
  readonly block: number
  readonly step: number
  readonly canvasTokenIds: ReadonlyArray<number>
  readonly feedbackInput: FeedbackFile | null
  readonly temperature: number
  readonly rngDraw: number
  postCanvasTokenIds: ReadonlyArray<number>
  draftTokenIds: ReadonlyArray<number>
  done: boolean
}
export interface ReplayManifest {
  readonly schema: "effect-torch-controlled-trajectory-v1"
  readonly label: "controlled-trajectory-replay-not-natural-generation"
  readonly request: {
    readonly promptTokenIds: ReadonlyArray<number>
    readonly seed: number
    readonly maxNewTokens: number
  }
  readonly steps: Array<ReplayStep>
  committedCanvasTokenIds: ReadonlyArray<number>
  outputTokenIds: ReadonlyArray<number>
}

const hash = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex")

const saveFeedback = (tensor: Tensor.Any, directory: string, index: number) =>
  Effect.gen(function*() {
    assert.equal(tensor.dtype, "bf16")
    const values = yield* Tensor.toTypedArray(tensor).pipe(Effect.orDie)
    assert(values instanceof Float32Array)
    const words = new Uint32Array(values.buffer, values.byteOffset, values.length)
    const payload = Buffer.allocUnsafe(values.length * 2)
    for (let i = 0; i < values.length; i++) {
      assert(Number.isFinite(values[i]), "canonical feedback must be finite")
      assert.equal(words[i]! & 65535, 0, "readback must preserve BF16 values exactly")
      payload.writeUInt16LE(words[i]! >>> 16, i * 2)
    }
    const header = Buffer.from(
      JSON.stringify({ feedback: { dtype: "BF16", shape: tensor.shape, data_offsets: [0, payload.length] } })
    )
    const padded = Buffer.alloc(Math.ceil(header.length / 8) * 8, 32)
    header.copy(padded)
    const length = Buffer.alloc(8)
    length.writeBigUInt64LE(BigInt(padded.length))
    const bytes = Buffer.concat([length, padded, payload])
    const file = `feedback-${index.toString().padStart(3, "0")}.safetensors`
    yield* Effect.promise(() => writeFile(join(directory, file), bytes, { flag: "wx" }))
    return { file, sha256: hash(bytes), dtype: "BF16" as const, shape: tensor.shape }
  })

export const captureArtifact = (
  artifact: Diffusion.Artifact,
  directory: string,
  manifest: ReplayManifest,
  temperature: (remaining: number) => number
): Diffusion.Artifact => ({
  ...artifact,
  generate: (options) => {
    let previous: FeedbackFile | null = null
    let currentCanvas: ReadonlyArray<number> = []
    return artifact.generate({
      ...options,
      initialize: (block) =>
        Effect.map(options.initialize(block), (owned) => {
          currentCanvas = Array.from(owned.value.canvas)
          previous = null
          return owned
        }),
      process: (logits, block, step) =>
        Effect.flatMap(options.process(logits, block, step), (owned) =>
          Effect.gen(function*() {
            const index = manifest.steps.length
            const feedback = yield* saveFeedback(owned.value.feedback, directory, index)
            manifest.steps.push({
              index,
              block: block.index,
              step: step.index,
              canvasTokenIds: currentCanvas,
              feedbackInput: previous,
              temperature: temperature(step.remaining),
              rngDraw: index,
              postCanvasTokenIds: [],
              draftTokenIds: [],
              done: false
            })
            previous = feedback
            return owned
          }).pipe(Effect.onExit((exit) => Exit.isFailure(exit) ? owned.release : Effect.void))),
      policy: {
        ...options.policy,
        finish: (draft, block) => {
          const completed = options.policy.finish(draft, block)
          manifest.committedCanvasTokenIds = Array.from(completed.tokens)
          return completed
        },
        refine: (input) =>
          Effect.map(options.policy.refine(input), (result) => {
            const record = manifest.steps[manifest.steps.length - 1]!
            record.postCanvasTokenIds = Array.from(result.canvas)
            record.draftTokenIds = Array.from(result.draft)
            record.done = result.done
            currentCanvas = record.postCanvasTokenIds
            return result
          })
      }
    })
  }
})

/** The caller's scope owns the whole bank. Each model invocation only borrows it. */
export const preloadReplay = (
  directory: string,
  manifest: ReplayManifest
): Effect.Effect<ReadonlyMap<string, Tensor.Any>, unknown, Runtime.Runtime | Scope.Scope> =>
  Effect.gen(function*() {
    const files = new Map<string, Tensor.Any>()
    for (const step of manifest.steps) {
      const record = step.feedbackInput
      if (record === null || files.has(record.file)) continue
      assert.equal(record.dtype, "BF16")
      const bytes = yield* Effect.promise(() => readFile(join(directory, record.file)))
      assert.equal(hash(bytes), record.sha256)
      const loaded = yield* Effect.acquireRelease(Safetensors.load(join(directory, record.file)), (tensors) =>
        Tensor.clearAll(Object.values(tensors)), { interruptible: true })
      const tensor = loaded.feedback
      assert(tensor !== undefined)
      assert.equal(tensor.dtype, "bf16")
      assert.deepEqual(tensor.shape, record.shape)
      files.set(record.file, tensor)
    }
    return files
  })

export const replayArtifact = (
  artifact: Diffusion.Artifact,
  manifest: ReplayManifest,
  bank: ReadonlyMap<string, Tensor.Any>
): Diffusion.Artifact => ({
  ...artifact,
  generate: (options) => {
    let index = 0
    return artifact.generate({
      ...options,
      initialize: (block) =>
        Effect.map(options.initialize(block), (owned) => {
          const record = manifest.steps[index]!
          assert.equal(record.block, block.index)
          assert.equal(record.feedbackInput, null)
          return { ...owned, value: { ...owned.value, canvas: Uint32Array.from(record.canvasTokenIds) } }
        }),
      process: (logits, block, step) =>
        Effect.flatMap(options.process(logits, block, step), (owned) =>
          Effect.gen(function*() {
            const record = manifest.steps[index]!
            assert.equal(record.block, block.index)
            assert.equal(record.step, step.index)
            const next = manifest.steps[index + 1]
            if (next?.feedbackInput === null || next?.feedbackInput === undefined) return owned
            const feedback = bank.get(next.feedbackInput.file)
            assert(feedback !== undefined)
            yield* owned.release
            return { value: { ...owned.value, feedback }, release: Effect.void }
          }).pipe(Effect.uninterruptible, Effect.onExit((exit) => Exit.isFailure(exit) ? owned.release : Effect.void))),
      policy: {
        ...options.policy,
        refine: (input) =>
          Effect.map(options.policy.refine(input), (native) => {
            const record = manifest.steps[index++]!
            assert.deepEqual(Array.from(input.canvas), record.canvasTokenIds)
            return {
              ...native,
              canvas: Uint32Array.from(record.postCanvasTokenIds),
              draft: Uint32Array.from(record.draftTokenIds),
              done: record.done
            }
          })
      }
    }).pipe(Effect.tap(() => Effect.sync(() => assert.equal(index, manifest.steps.length))))
  }
})

export interface ReplayExecutionCounts {
  prefill: number
  reads: number
  samplers: number
  commits: number
}

/** CPU counters at the actual artifact operations, without device readbacks. */
const counted = <A, E, R>(
  operation: Effect.Effect<A, E, R>,
  counts: ReplayExecutionCounts | undefined,
  key: keyof ReplayExecutionCounts
): Effect.Effect<A, E, R> =>
  counts === undefined ? operation : Effect.tap(operation, () =>
    Effect.sync(() => {
      counts[key]++
    }))

/** Controlled replay explicitly includes a terminal full-canvas encoder commit. */
export const includeFinalCommit = (
  artifact: Diffusion.Artifact,
  counts?: ReplayExecutionCounts
): Diffusion.Artifact => ({
  ...artifact,
  generate: (options) =>
    Effect.suspend(() => {
      let current: Diffusion.Prefix | undefined
      let completed: Diffusion.CompletedBlock | undefined
      let published = 0
      return Diffusion.runGeneration({
        ...options,
        policy: {
          ...options.policy,
          finish: (draft, block) => {
            completed = options.policy.finish(draft, block)
            return completed
          }
        },
        onPage: (tokens, block) =>
          Effect.gen(function*() {
            published += tokens.length
            assert(completed !== undefined && current !== undefined)
            if (completed.stop || published >= options.maxNewTokens) {
              yield* Effect.acquireUseRelease(
                counted(artifact.commit(current, completed.tokens), counts, "commits"),
                () => Effect.void,
                (prefix) => Effect.orDie(artifact.release(prefix))
              )
            }
            yield* options.onPage(tokens, block)
          }),
        callbacks: {
          initialize: options.initialize,
          encode: (tokens) =>
            Effect.map(counted(artifact.encode(tokens), counts, "prefill"), (value) => {
              current = value
              return { value, release: Effect.orDie(artifact.release(value)) }
            }),
          commit: (prefix: Diffusion.Prefix, tokens) =>
            Effect.map(counted(artifact.commit(prefix, tokens), counts, "commits"), (value) => {
              current = value
              return { value, release: Effect.orDie(artifact.release(value)) }
            }),
          evaluate: ({ prefix, canvas, feedback, block, step }) =>
            Effect.suspend(() => {
              let processed: Effect.Success<ReturnType<typeof options.process>> | undefined
              return Effect.scoped(Effect.gen(function*() {
                const logits = yield* Effect.acquireRelease(
                  counted(
                    artifact.evaluate(
                      prefix,
                      canvas,
                      feedback._tag === "Initial" ? { _tag: "Initial" } : { _tag: "Refinement", logits: feedback.value }
                    ),
                    counts,
                    "reads"
                  ),
                  Tensor.clear,
                  { interruptible: true }
                )
                return yield* counted(options.process(logits, block, step), counts, "samplers").pipe(
                  Effect.onExit((exit) =>
                    Effect.sync(() => {
                      if (Exit.isSuccess(exit)) processed = exit.value
                    })
                  )
                )
              })).pipe(
                Effect.onExit((exit) =>
                  Exit.isFailure(exit) && processed !== undefined ? processed.release : Effect.void
                )
              )
            })
        }
      })
    })
})
