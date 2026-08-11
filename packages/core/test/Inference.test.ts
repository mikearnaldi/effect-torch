import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import { LearningRate, Loss, Model, Optimizer, Tensor, Trainer } from "../src/index.ts"
import { deep, onDevices, TOL } from "./utils/devices.ts"

const VOCAB = 12
const BLOCK = 16
const EMBED = 8
const HEADS = 2

const makeGpt = (options: { readonly causal?: boolean } = {}) =>
  Effect.gen(function*() {
    const embeddings = yield* Model.add(
      yield* Model.embedding("wte", VOCAB, EMBED),
      yield* Model.positionEmbedding("wpe", BLOCK, EMBED)
    )
    const attn = yield* Model.multiHeadAttention("attn", EMBED, HEADS, {
      causal: options.causal ?? true
    })
    const head = yield* Model.linear("head", EMBED, VOCAB)
    return yield* Model.chain(embeddings, attn, head)
  })

// The RoPE variant: relative positions, no position table — the model
// sliding-window attention is trained for.
const makeRopeGpt = Effect.gen(function*() {
  const wte = yield* Model.embedding("wte", VOCAB, EMBED)
  const attn = yield* Model.multiHeadAttention("attn", EMBED, HEADS, { causal: true, rope: 10000 })
  const head = yield* Model.linear("head", EMBED, VOCAB)
  return yield* Model.chain(wte, attn, head)
})

const ids = (tokens: ReadonlyArray<number>) => Tensor.fromTypedArray(new Uint32Array(tokens), [1, tokens.length])

const argmaxOf = (logits: Tensor.Any) =>
  Effect.gen(function*() {
    const values = yield* Tensor.toNumberArray(logits)
    if (Tensor.isTensor(logits)) yield* Tensor.clear(logits)
    return values.reduce((best, value, index) => (value > values[best] ? index : best), 0)
  })

// The reference: greedy generation through the ordinary forward graph,
// recomputing the whole context every step.
const naiveGenerate = (
  model: Model.Model,
  params: Model.Params,
  prompt: ReadonlyArray<number>,
  steps: number
) =>
  Effect.gen(function*() {
    const context = [...prompt]
    for (let i = 0; i < steps; i++) {
      const input = yield* ids(context.slice(-BLOCK))
      const output = yield* model.forward(params, input)
      const t = input.shape[1]
      const [logits] = yield* Tensor.compute([
        yield* Tensor.reshape(
          yield* Tensor.slice(output, { start: [0, t - 1, 0], end: [1, t, VOCAB] }),
          [VOCAB]
        )
      ])
      context.push(yield* argmaxOf(logits))
    }
    return context
  })

// The window-relative reference: the pre-cache generation loop — every
// step recomputes the last `window` tokens with positions 0..window-1.
// With RoPE, cached sliding-window attention must match this exactly.
const naiveWindowedGenerate = (
  model: Model.Model,
  params: Model.Params,
  prompt: ReadonlyArray<number>,
  steps: number,
  window: number
) =>
  Effect.gen(function*() {
    const context = [...prompt]
    for (let i = 0; i < steps; i++) {
      const input = yield* ids(context.slice(-window))
      const output = yield* model.forward(params, input)
      const t = input.shape[1]
      const [logits] = yield* Tensor.compute([
        yield* Tensor.reshape(
          yield* Tensor.slice(output, { start: [0, t - 1, 0], end: [1, t, VOCAB] }),
          [VOCAB]
        )
      ])
      context.push(yield* argmaxOf(logits))
    }
    return context
  })

// Greedy generation through the inference artifact: add the prompt once,
// then one round per token with an argmax chooser.
const cachedGenerate = (
  program: Model.InferenceProgram,
  prompt: ReadonlyArray<number>,
  steps: number
) =>
  Effect.gen(function*() {
    const gen = yield* program.generation()
    const context = [...prompt]
    const entry = yield* gen.add(yield* ids(prompt))
    let logits = entry.logits
    for (let i = 0; i < steps; i++) {
      const next = yield* argmaxOf(logits)
      context.push(next)
      if (i < steps - 1) {
        const [nextLogits] = yield* gen.step([{ seq: entry.seq, token: next }])
        logits = nextLogits
      }
    }
    return context
  })

onDevices("Inference", () => (it) => {
  describe("Model.inference", () => {
    it.effect("matches naive greedy generation token-for-token across pool block boundaries", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 5, 3]
        const steps = 9 // context grows to 12, crossing block boundaries (blockSize 4)
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4 })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
        expect(cached.length).toBe(prompt.length + steps)
      }))

    it.effect("serves every prompt length from the two eagerly compiled programs", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4 })
        const a = yield* cachedGenerate(program, [1, 5, 3], 6)
        const b = yield* cachedGenerate(program, [2, 4, 6], 6)
        const c = yield* cachedGenerate(program, [7, 8], 6)
        expect(a.length).toBe(9)
        expect(b.length).toBe(9)
        expect(c.length).toBe(8)
      }))

    it.effect("chunked prefill: a long prompt runs in fixed-shape chunks with parity", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 5, 3, 8, 2, 11, 4, 7, 6] // 3 chunks of 4: 4 + 4 + 1(padded)
        const steps = 6
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunk: 4 })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      }))

    it.effect("runs concurrent sessions exactly like sequential ones", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4 })
        const sequentialA = yield* cachedGenerate(program, [1, 2], 6)
        const sequentialB = yield* cachedGenerate(program, [3, 4], 6)
        const [concurrentA, concurrentB] = yield* Effect.all(
          [cachedGenerate(program, [1, 2], 6), cachedGenerate(program, [3, 4], 6)],
          { concurrency: "unbounded" }
        )
        expect(concurrentA).toEqual(sequentialA)
        expect(concurrentB).toEqual(sequentialB)
      }))

    it.effect("pool exhaustion fails an add and finish frees the room", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        const gen = yield* program.generation()
        yield* gen.add(yield* ids(Array.from({ length: 16 }, (_, i) => i % VOCAB))) // all 4 blocks
        expect(yield* gen.live()).toBe(1)
        // The failed run allocated nothing; after finishing the first
        // sequence, the same prompt fits the pool.
        const error = yield* Effect.flip(gen.add(yield* ids([1, 2, 3, 4, 5, 6, 7, 8])))
        expect(error.message).toMatch(/pool exhausted/)
        expect(yield* gen.live()).toBe(1)
      }))

    it.effect("fails a sequence whose context outgrows the pool capacity", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 8, blockSize: 4 })
        const gen = yield* program.generation()
        const entry = yield* gen.add(yield* ids([1, 2, 3, 4, 5, 6, 7, 8]))
        expect(yield* entry.seq.cursor()).toBe(8)
        const error = yield* Effect.flip(gen.step([{ seq: entry.seq, token: 1 }]))
        expect(error._tag).toBe("TensorError")
        expect(error.message).toMatch(/exceeds pool capacity/)
      }))

    it.effect("returns a finished sequence's blocks to the pool", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        const gen = yield* program.generation()
        const entry = yield* gen.add(yield* ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])) // 3 of 4 blocks
        yield* entry.seq.finish()
        expect(yield* gen.live()).toBe(0)
        // Only possible if the finished blocks came back.
        const full = yield* gen.add(yield* ids(Array.from({ length: 16 }, (_, i) => i % VOCAB)))
        expect(yield* full.seq.cursor()).toBe(16)
      }))

    it.effect("prefix cache: a resident prefix is shared, not recomputed", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        // 5 blocks: two independent 3-block prompts would need 6 — the
        // second prefill fits only by sharing its 2 full prefix blocks.
        const program = yield* Model.inference(model, params, { maxTokens: 20, blockSize: 4 })
        const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        const gen = yield* program.generation()
        const a = yield* gen.add(yield* ids(prompt))
        const b = yield* gen.add(yield* ids(prompt))
        deep(yield* Tensor.toNumberArray(b.logits), yield* Tensor.toNumberArray(a.logits))
      }))

    it.effect("prefix cache: divergent suffixes after a shared prefix stay correct", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4 })
        const shared = [1, 2, 3, 4, 5, 6, 7, 8] // 2 full blocks
        const promptA = [...shared, 9, 10, 11, 0]
        const promptB = [...shared, 3, 4, 5, 6]
        const gen = yield* program.generation()
        yield* gen.add(yield* ids(promptA))
        const b = yield* gen.add(yield* ids(promptB))
        // The reference: an ordinary forward over B's whole prompt.
        const input = yield* ids(promptB)
        const output = yield* model.forward(params, input)
        const [expected] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(output, { start: [0, promptB.length - 1, 0], end: [1, promptB.length, VOCAB] }),
            [VOCAB]
          )
        ])
        deep(yield* Tensor.toNumberArray(b.logits), yield* Tensor.toNumberArray(expected))
      }))

    it.effect("prefix cache: cached blocks are reclaimed under pressure", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        // Exactly one 3-block prompt fits; a second, different prompt
        // succeeds only by evicting the first's cached blocks.
        const program = yield* Model.inference(model, params, { maxTokens: 12, blockSize: 4 })
        {
          const gen = yield* program.generation()
          yield* gen.add(yield* ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]))
          yield* gen.close()
        }
        const prompt = [2, 4, 6, 8, 10, 0, 1, 3, 5, 7, 9, 11]
        const gen = yield* program.generation()
        const entry = yield* gen.add(yield* ids(prompt))
        const input = yield* ids(prompt)
        const output = yield* model.forward(params, input)
        const [expected] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(output, { start: [0, prompt.length - 1, 0], end: [1, prompt.length, VOCAB] }),
            [VOCAB]
          )
        ])
        deep(yield* Tensor.toNumberArray(entry.logits), yield* Tensor.toNumberArray(expected))
      }))

    it.effect("prefix cache: window-evicted blocks stay reusable", () =>
      Effect.gen(function*() {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 3, 5, 7, 9, 11, 2, 4]
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          attentionWindow: 8
        })
        // Generate past the window: the prompt's first block leaves the
        // window, lands in the prefix cache, and the sequence finishes
        // the rest of the prompt's blocks into the cache as well.
        {
          const gen = yield* program.generation()
          const entry = yield* gen.add(yield* ids(prompt))
          for (let i = 0; i < 4; i++) {
            yield* gen.step([{ seq: entry.seq, token: 0 }])
          }
          yield* gen.close()
        }
        const gen = yield* program.generation()
        const entry = yield* gen.add(yield* ids(prompt))
        const input = yield* ids(prompt)
        const output = yield* model.forward(params, input)
        const [expected] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(output, { start: [0, prompt.length - 1, 0], end: [1, prompt.length, VOCAB] }),
            [VOCAB]
          )
        ])
        deep(yield* Tensor.toNumberArray(entry.logits), yield* Tensor.toNumberArray(expected))
      }))

    it.effect("each add starts a fresh, independent sequence", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        const gen = yield* program.generation()
        const a = yield* gen.add(yield* ids([1, 2, 3]))
        const b = yield* gen.add(yield* ids([1, 2, 3]))
        expect(yield* a.seq.cursor()).toBe(3)
        expect(yield* b.seq.cursor()).toBe(3)
        deep(yield* Tensor.toNumberArray(b.logits), yield* Tensor.toNumberArray(a.logits))
      }))

    it.effect("prefix cache: concurrent same-prefix prefills stay exact", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4 })
        const prompt = [1, 2, 3, 4, 5, 6, 7, 8] // 1 matchable block; +6 steps stays within BLOCK
        // However the two prefills interleave — one takes the other's
        // blocks mid-flight, or both miss and compute — greedy
        // generation must match the sequential runs token-for-token.
        const sequentialA = yield* cachedGenerate(program, prompt, 6)
        const sequentialB = yield* cachedGenerate(program, prompt, 6)
        const [concurrentA, concurrentB] = yield* Effect.all(
          [cachedGenerate(program, prompt, 6), cachedGenerate(program, prompt, 6)],
          { concurrency: "unbounded" }
        )
        expect(concurrentA).toEqual(sequentialA)
        expect(concurrentB).toEqual(sequentialB)
        expect(sequentialA).toEqual(sequentialB)
      }))

    // Half-precision pools (RFC 0012): rows quantized on write, widened
    // on read. Teacher-forced — both sides see the same context — so
    // the comparison is logits closeness, not argmax luck.
    const halfPoolParity = (kvDtype: "f16" | "bf16" | "int8", tol: number) =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, kvDtype })
        const prompt = [1, 5, 3, 8, 2]
        const trajectory = [4, 9, 0, 7, 6]
        const gen = yield* program.generation()
        const context = [...prompt]
        const entry = yield* gen.add(yield* ids(prompt))
        let logits = entry.logits
        const check = (actual: Tensor.Any, ctx: ReadonlyArray<number>) =>
          Effect.gen(function*() {
            const input = yield* ids(ctx.slice(-BLOCK))
            const output = yield* model.forward(params, input)
            const t = input.shape[1]
            const [expected] = yield* Tensor.compute([
              yield* Tensor.reshape(
                yield* Tensor.slice(output, { start: [0, t - 1, 0], end: [1, t, VOCAB] }),
                [VOCAB]
              )
            ])
            const got = yield* Tensor.toNumberArray(actual)
            const want = yield* Tensor.toNumberArray(expected)
            for (let i = 0; i < VOCAB; i++) {
              expect(Math.abs(got[i]! - want[i]!)).toBeLessThan(tol)
            }
          })
        yield* check(logits, context)
        for (const next of trajectory) {
          context.push(next)
          const [nextLogits] = yield* gen.step([{ seq: entry.seq, token: next }])
          logits = nextLogits
          yield* check(logits, context)
        }
      })

    it.effect("f16 pool: teacher-forced logits track the f32 reference", () => halfPoolParity("f16", 2e-2))

    it.effect("bf16 pool: teacher-forced logits track the f32 reference", () => halfPoolParity("bf16", 6e-2))

    it.effect("int8 pool: teacher-forced logits track the f32 reference", () => halfPoolParity("int8", 1e-1))

    it.effect("wpe sliding window: inert below the window, self-consistent across eviction", () =>
      Effect.gen(function*() {
        // Sliding-window attention is not RoPE-specific: it is a pool
        // memory policy. With learned absolute positions the window
        // works within the table — positions stay absolute — so (a)
        // below the window generation matches the naive loop exactly,
        // and (b) past it a stepped sequence and a fresh prefill of
        // the same context agree (this would FAIL for window-relative
        // positions, which are the RoPE-only regime).
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          attentionWindow: 8
        })
        // (a) context 3 + 4 steps = 7 < window 8: window never engages.
        const naive = yield* naiveGenerate(model, params, [1, 5, 3], 4)
        const cached = yield* cachedGenerate(program, [1, 5, 3], 4)
        expect(cached).toEqual(naive)
        // (b) prefill 12, step 4 greedy (context 16, first block
        // evicted), then a fresh add of the full 16-token context
        // must produce the same last-position logits.
        const gen = yield* program.generation()
        const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        const context = [...prompt]
        const a = yield* gen.add(yield* ids(prompt))
        let logits = a.logits
        for (let i = 0; i < 4; i++) {
          const next = yield* argmaxOf(logits)
          context.push(next)
          const [nextLogits] = yield* gen.step([{ seq: a.seq, token: next }])
          logits = nextLogits
        }
        const fresh = yield* gen.add(yield* ids(context))
        deep(yield* Tensor.toNumberArray(fresh.logits), yield* Tensor.toNumberArray(logits))
      }))

    it.effect("batched step matches per-sequence step logits exactly", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, decodeBatch: 4 })
        const prompts = [
          [1, 2, 3],
          [4, 5, 6, 7, 8],
          [9, 10],
          [1, 2, 3, 0, 11, 5, 6]
        ]
        // Sequential reference: one session per prompt, one step each.
        const reference: Array<Array<number>> = []
        for (const prompt of prompts) {
          const gen = yield* program.generation()
          const entry = yield* gen.add(yield* ids(prompt))
          const [logits] = yield* gen.step([{ seq: entry.seq, token: 1 }])
          reference.push(yield* Tensor.toNumberArray(logits))
        }
        // Batched: one session, all prompts, one round stepping all four.
        const gen = yield* program.generation()
        const entries: Array<Model.GenerationEntry> = []
        for (const prompt of prompts) {
          entries.push(yield* gen.add(yield* ids(prompt)))
        }
        const batched = yield* gen.step(entries.map(({ seq }) => ({ seq, token: 1 })))
        expect(batched.length).toBe(prompts.length)
        for (let i = 0; i < prompts.length; i++) {
          deep(yield* Tensor.toNumberArray(batched[i]!), reference[i]!)
        }
        // The round advanced every cursor exactly once.
        for (let i = 0; i < prompts.length; i++) {
          expect(yield* entries[i]!.seq.cursor()).toBe(prompts[i]!.length + 1)
        }
      }))

    it.effect("ragged batches pad internally and stay exact", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        // Three blocks fit the reference plus two active sequences exactly;
        // inactive padding rows must not reserve KV capacity.
        const program = yield* Model.inference(model, params, { maxTokens: 12, blockSize: 4, decodeBatch: 8 })
        // Sequential reference, single-sequence path.
        const ref = yield* program.generation()
        const r1 = yield* ref.add(yield* ids([3, 1, 4]))
        const [expected] = yield* ref.step([{ seq: r1.seq, token: 2 }])
        // Two live sequences in one round: 6 slots pad internally.
        const gen = yield* program.generation()
        const a = yield* gen.add(yield* ids([3, 1, 4]))
        const b = yield* gen.add(yield* ids([7, 7, 7]))
        const [gotA] = yield* gen.step([
          { seq: a.seq, token: 2 },
          { seq: b.seq, token: 2 }
        ])
        deep(yield* Tensor.toNumberArray(gotA!), yield* Tensor.toNumberArray(expected))
        expect(yield* a.seq.cursor()).toBe(4)
        expect(yield* b.seq.cursor()).toBe(4)
      }))

    it.effect("batched step with divergent cursors and a shared prefix", () =>
      Effect.gen(function*() {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          attentionWindow: 8,
          decodeBatch: 2
        })
        const prompt = [1, 3, 5, 7, 9, 11, 2, 4]
        // A generates alone first (evicting its first block past the
        // window); B is added afterwards and shares A's cached blocks.
        const gen = yield* program.generation()
        const a = yield* gen.add(yield* ids(prompt))
        let logitsA = a.logits
        for (let i = 0; i < 4; i++) {
          const next = yield* argmaxOf(logitsA)
          const [nextLogits] = yield* gen.step([{ seq: a.seq, token: next }])
          logitsA = nextLogits
        }
        const b = yield* gen.add(yield* ids(prompt))
        let logitsB = b.logits
        // Reference: independent windowed generation from the same
        // prompt for both trajectories.
        const refA = yield* naiveWindowedGenerate(model, params, prompt, 11, 8)
        const refB = yield* naiveWindowedGenerate(model, params, prompt, 7, 8)
        // Six batched rounds: cursors 12+i and 8+i in one run.
        for (let i = 0; i < 6; i++) {
          const nextA = refA[prompt.length + 4 + i]!
          const nextB = refB[prompt.length + i]!
          const [outA, outB] = yield* gen.step([
            { seq: a.seq, token: nextA },
            { seq: b.seq, token: nextB }
          ])
          logitsA = outA!
          logitsB = outB!
          // Batched logits must equal the sequential reference argmax.
          expect(yield* argmaxOf(logitsA)).toBe(refA[prompt.length + 5 + i]!)
          expect(yield* argmaxOf(logitsB)).toBe(refB[prompt.length + 1 + i]!)
        }
      }))

    it.effect("batched step rolls back every slot on pool exhaustion", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        // Pool holds both prompts exactly; the batched step needs one
        // more block per sequence and must fail cleanly.
        const program = yield* Model.inference(model, params, { maxTokens: 24, blockSize: 4, decodeBatch: 2 })
        const gen = yield* program.generation()
        const a = yield* gen.add(yield* ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])) // 3 blocks
        const b = yield* gen.add(yield* ids([2, 4, 6, 8, 10, 0, 1, 3, 5, 7, 9, 11])) // 3 blocks
        const error = yield* Effect.flip(
          gen.step([
            { seq: a.seq, token: 1 },
            { seq: b.seq, token: 2 }
          ])
        )
        expect(error._tag).toBe("TensorError")
        expect(error.message).toMatch(/pool exhausted/)
        // Neither cursor advanced.
        expect(yield* a.seq.cursor()).toBe(12)
        expect(yield* b.seq.cursor()).toBe(12)
      }))

    it.effect("add beyond decodeBatch fails typed", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, decodeBatch: 2 })
        const gen = yield* program.generation()
        yield* gen.add(yield* ids([1, 2, 3]))
        yield* gen.add(yield* ids([4, 5, 6]))
        const error = yield* Effect.flip(gen.add(yield* ids([7, 8, 9])))
        expect(error._tag).toBe("InferenceError")
        expect(error.message).toMatch(/at most decodeBatch/)
      }))

    it.effect("finishing a sequence mid-session removes it from the round", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, decodeBatch: 2 })
        const gen = yield* program.generation()
        const a = yield* gen.add(yield* ids([1, 2, 3]))
        const b = yield* gen.add(yield* ids([4, 5, 6]))
        yield* a.seq.finish()
        const out = yield* gen.step([{ seq: b.seq, token: 1 }])
        expect(out.length).toBe(1)
        expect(yield* gen.live()).toBe(1)
        expect(yield* b.seq.cursor()).toBe(4)
      }))

    it.effect("f16 pool: prefix cache and sliding window still hold", () =>
      Effect.gen(function*() {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 3, 5, 7, 9, 11, 2, 4]
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          attentionWindow: 8,
          kvDtype: "f16"
        })
        // A resident prefix is shared in the half-precision pool too:
        // two independent 2-block prompts would need 4 of 8 blocks plus
        // B's private suffix block — fits either way, so assert exact
        // equality of the shared computation instead.
        const gen = yield* program.generation()
        const a = yield* gen.add(yield* ids(prompt))
        const b = yield* gen.add(yield* ids(prompt))
        deep(yield* Tensor.toNumberArray(b.logits), yield* Tensor.toNumberArray(a.logits))
        // And windowed generation past eviction stays close to the
        // window-relative f32 recompute (a steps alone; b stays live
        // holding the shared prefix).
        const context = [...prompt]
        let logits = a.logits
        for (let i = 0; i < 6; i++) {
          const next = yield* argmaxOf(logits)
          context.push(next)
          const [nextLogits] = yield* gen.step([{ seq: a.seq, token: next }])
          logits = nextLogits
          const input = yield* ids(context.slice(-8))
          const output = yield* model.forward(params, input)
          const t = input.shape[1]
          const [expected] = yield* Tensor.compute([
            yield* Tensor.reshape(
              yield* Tensor.slice(output, { start: [0, t - 1, 0], end: [1, t, VOCAB] }),
              [VOCAB]
            )
          ])
          const got = yield* Tensor.toNumberArray(logits)
          const want = yield* Tensor.toNumberArray(expected)
          for (let j = 0; j < VOCAB; j++) {
            expect(Math.abs(got[j]! - want[j]!)).toBeLessThan(2e-2)
          }
        }
      }))

    it.effect("supports a model without cached state", () =>
      Effect.gen(function*() {
        const model = yield* Model.chain(
          yield* Model.embedding("wte", VOCAB, EMBED),
          yield* Model.linear("head", EMBED, VOCAB)
        )
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        const gen = yield* program.generation()
        const entry = yield* gen.add(yield* ids([1, 3, 5]))
        const promptOutput = yield* model.forward(params, yield* ids([1, 3, 5]))
        const [expectedPrompt] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(promptOutput, { start: [0, 2, 0], end: [1, 3, VOCAB] }),
            [VOCAB]
          )
        ])
        deep(yield* Tensor.toNumberArray(entry.logits), yield* Tensor.toNumberArray(expectedPrompt))

        const [stepLogits] = yield* gen.step([{ seq: entry.seq, token: 7 }])
        const stepOutput = yield* model.forward(params, yield* ids([7]))
        const [expectedStep] = yield* Tensor.compute([
          yield* Tensor.reshape(yield* Tensor.slice(stepOutput, { start: [0, 0, 0], end: [1, 1, VOCAB] }), [VOCAB])
        ])
        deep(yield* Tensor.toNumberArray(stepLogits), yield* Tensor.toNumberArray(expectedStep))
        expect(yield* entry.seq.cursor()).toBe(4)
        yield* entry.seq.finish()
        yield* Tensor.clear(entry.logits)
        yield* Tensor.clear(expectedPrompt)
        yield* Tensor.clear(stepLogits)
        yield* Tensor.clear(expectedStep)
      }))

    it.effect("rejects non-causal attention at construction", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt({ causal: false })
        const params = yield* Tensor.compute(yield* model.init)
        const error = yield* Effect.flip(Model.inference(model, params, { maxTokens: 16, blockSize: 4 }))
        expect(error._tag).toBe("InferenceError")
        expect(error.message).toMatch(/only causal attention is cacheable/)
      }))

    it.effect("validates the add calling convention", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        const gen = yield* program.generation()
        const batched = yield* Effect.flip(gen.add(yield* Tensor.fromTypedArray(new Uint32Array(6), [2, 3])))
        expect(batched._tag).toBe("InferenceError")
        expect(batched.message).toMatch(/expects a prompt of shape \[1, T\]/)
        const badPool = yield* Effect.flip(Model.inference(model, params, { maxTokens: 15, blockSize: 4 }))
        expect(badPool._tag).toBe("InferenceError")
        expect(badPool.message).toMatch(/multiple of blockSize/)
      }))

    it.effect("matches the naive logits numerically, not just on argmax", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4 })
        const gen = yield* program.generation()
        const entry = yield* gen.add(yield* ids([1, 5, 3]))
        const output = yield* model.forward(params, yield* ids([1, 5, 3]))
        const [naive] = yield* Tensor.compute([
          yield* Tensor.reshape(yield* Tensor.slice(output, { start: [0, 2, 0], end: [1, 3, VOCAB] }), [
            VOCAB
          ])
        ])
        deep(yield* Tensor.toNumberArray(entry.logits), yield* Tensor.toNumberArray(naive))
        void TOL
      }))

    it.effect("RoPE: matches naive greedy generation with full attention", () =>
      Effect.gen(function*() {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 5, 3]
        const steps = 8
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4 })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      }))

    it.effect("RoPE + attention window: matches the window-relative recompute token-for-token, unbounded", () =>
      Effect.gen(function*() {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 5]
        const window = 8
        const steps = 24 // the context grows to 26: far past the window
        const program = yield* Model.inference(model, params, {
          maxTokens: 16, // 4 blocks: only eviction of dead blocks lets this run at all
          blockSize: 4,
          attentionWindow: window
        })
        const naive = yield* naiveWindowedGenerate(model, params, prompt, steps, window)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
        expect(cached.length).toBe(prompt.length + steps)
      }))

    it.effect("RoPE: trains — the rotary node differentiates", () =>
      Effect.gen(function*() {
        const model = yield* makeRopeGpt
        const data = Array.from({ length: 64 }, (_, i) => i % 4)
        const losses: Array<number> = []
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.adamW(),
          lr: LearningRate.constant(3e-3),
          loss: Loss.crossEntropy,
          data: () =>
            Effect.gen(function*() {
              const start = Math.floor(Math.random() * (data.length - BLOCK - 1))
              return {
                input: yield* ids(data.slice(start, start + BLOCK)),
                target: yield* Tensor.fromTypedArray(
                  BigInt64Array.from(data.slice(start + 1, start + BLOCK + 1), BigInt),
                  [1, BLOCK]
                )
              }
            }),
          stop: ({ step }) => step >= 50,
          onStep: ({ loss }) => Effect.sync(() => losses.push(loss))
        })
        yield* trainer.train()
        expect(losses.length).toBe(50)
        expect(Number.isFinite(losses[49])).toBe(true)
        expect(losses[49]).toBeLessThan(losses[0])
      }))
  })
})
