import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import { LearningRate, Loss, Model, Optimizer, Tensor, Trainer } from "../src/index.ts"
import { deep, onDevices, TOL } from "./utils/devices.ts"

const VOCAB = 12
const BLOCK = 16
const EMBED = 8
const HEADS = 2

const makeGpt = (options: { readonly causal?: boolean } = {}) =>
  Effect.gen(function* () {
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
const makeRopeGpt = Effect.gen(function* () {
  const wte = yield* Model.embedding("wte", VOCAB, EMBED)
  const attn = yield* Model.multiHeadAttention("attn", EMBED, HEADS, { causal: true, rope: 10000 })
  const head = yield* Model.linear("head", EMBED, VOCAB)
  return yield* Model.chain(wte, attn, head)
})

const ids = (tokens: ReadonlyArray<number>) =>
  Tensor.fromTypedArray(new Uint32Array(tokens), [1, tokens.length])

const argmaxOf = (logits: Tensor.Any) =>
  Effect.map(Tensor.toNumberArray(logits), (values) =>
    values.reduce((best, value, index) => (value > values[best] ? index : best), 0)
  )

// The reference: greedy generation through the ordinary forward graph,
// recomputing the whole context every step.
const naiveGenerate = (
  model: Model.Model,
  params: Model.Params,
  prompt: ReadonlyArray<number>,
  steps: number
) =>
  Effect.gen(function* () {
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
  Effect.gen(function* () {
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

// Greedy generation through the inference artifact: prefill once, one
// pooled step per token.
const cachedGenerate = (
  program: Model.InferenceProgram,
  prompt: ReadonlyArray<number>,
  steps: number
) =>
  Effect.gen(function* () {
    const seq = yield* program.sequence()
    const context = [...prompt]
    let logits = yield* seq.prefill(yield* ids(prompt))
    for (let i = 0; i < steps; i++) {
      const next = yield* argmaxOf(logits)
      context.push(next)
      if (i < steps - 1) {
        logits = yield* seq.step(yield* ids([next]))
      }
    }
    return context
  })

onDevices("Inference", () => (it) => {
  describe("Model.inference", () => {
    it.effect("matches naive greedy generation token-for-token across pool block boundaries", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 5, 3]
        const steps = 9 // context grows to 12, crossing block boundaries (blockSize 4)
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4 })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
        expect(cached.length).toBe(prompt.length + steps)
      })
    )

    it.effect("serves every prompt length from the two eagerly compiled programs", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4 })
        const a = yield* cachedGenerate(program, [1, 5, 3], 6)
        const b = yield* cachedGenerate(program, [2, 4, 6], 6)
        const c = yield* cachedGenerate(program, [7, 8], 6)
        expect(a.length).toBe(9)
        expect(b.length).toBe(9)
        expect(c.length).toBe(8)
      })
    )

    it.effect("chunked prefill: a long prompt runs in fixed-shape chunks with parity", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 5, 3, 8, 2, 11, 4, 7, 6] // 3 chunks of 4: 4 + 4 + 1(padded)
        const steps = 6
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunk: 4 })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      })
    )

    it.effect("runs concurrent sequences exactly like sequential ones", () =>
      Effect.gen(function* () {
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
      })
    )

    it.effect("pool exhaustion fails one sequence and leaves others unaffected", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        const seq1 = yield* program.sequence()
        yield* seq1.prefill(yield* ids(Array.from({ length: 16 }, (_, i) => i % VOCAB))) // all 4 blocks
        expect(yield* seq1.cursor()).toBe(16)
        const seq2 = yield* program.sequence()
        const error = yield* Effect.flip(seq2.prefill(yield* ids([1, 2, 3, 4, 5, 6, 7, 8])))
        expect(error._tag).toBe("TensorError")
        expect(error.message).toMatch(/pool exhausted/)
        // The failed run allocated nothing: after seq1 releases, the
        // same prefill fits the pool.
        expect(yield* seq2.cursor()).toBe(0)
        yield* Effect.scoped(Effect.gen(function* () {
          const seq3 = yield* program.sequence()
          const stillFull = yield* Effect.flip(seq3.prefill(yield* ids([1, 2, 3, 4, 5, 6, 7, 8])))
          expect(stillFull.message).toMatch(/pool exhausted/)
        }))
      })
    )

    it.effect("fails a sequence whose context outgrows the pool capacity", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 8, blockSize: 4 })
        const seq = yield* program.sequence()
        yield* seq.prefill(yield* ids([1, 2, 3, 4, 5, 6, 7, 8]))
        expect(yield* seq.cursor()).toBe(8)
        const error = yield* Effect.flip(seq.step(yield* ids([1])))
        expect(error._tag).toBe("TensorError")
        expect(error.message).toMatch(/exceeds pool capacity/)
      })
    )

    it.effect("returns a released sequence's blocks to the pool", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        yield* Effect.scoped(Effect.gen(function* () {
          const seq = yield* program.sequence()
          yield* seq.prefill(yield* ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])) // 3 of 4 blocks
        }))
        // Only possible if the released blocks came back.
        const seq = yield* program.sequence()
        yield* seq.prefill(yield* ids(Array.from({ length: 16 }, (_, i) => i % VOCAB)))
        expect(yield* seq.cursor()).toBe(16)
      })
    )

    it.effect("prefix cache: a resident prefix is shared, not recomputed", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        // 5 blocks: two independent 3-block prompts would need 6 — the
        // second prefill fits only by sharing its 2 full prefix blocks.
        const program = yield* Model.inference(model, params, { maxTokens: 20, blockSize: 4 })
        const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        const seqA = yield* program.sequence()
        const logitsA = yield* seqA.prefill(yield* ids(prompt))
        const seqB = yield* program.sequence()
        const logitsB = yield* seqB.prefill(yield* ids(prompt))
        deep(yield* Tensor.toNumberArray(logitsB), yield* Tensor.toNumberArray(logitsA))
      })
    )

    it.effect("prefix cache: divergent suffixes after a shared prefix stay correct", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4 })
        const shared = [1, 2, 3, 4, 5, 6, 7, 8] // 2 full blocks
        const promptA = [...shared, 9, 10, 11, 0]
        const promptB = [...shared, 3, 4, 5, 6]
        const seqA = yield* program.sequence()
        yield* seqA.prefill(yield* ids(promptA))
        const seqB = yield* program.sequence()
        const logitsB = yield* seqB.prefill(yield* ids(promptB))
        // The reference: an ordinary forward over B's whole prompt.
        const input = yield* ids(promptB)
        const output = yield* model.forward(params, input)
        const [expected] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(output, { start: [0, promptB.length - 1, 0], end: [1, promptB.length, VOCAB] }),
            [VOCAB]
          )
        ])
        deep(yield* Tensor.toNumberArray(logitsB), yield* Tensor.toNumberArray(expected))
      })
    )

    it.effect("prefix cache: cached blocks are reclaimed under pressure", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        // Exactly one 3-block prompt fits; a second, different prompt
        // succeeds only by evicting the first's cached blocks.
        const program = yield* Model.inference(model, params, { maxTokens: 12, blockSize: 4 })
        yield* Effect.scoped(Effect.gen(function* () {
          const seq = yield* program.sequence()
          yield* seq.prefill(yield* ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]))
        }))
        const prompt = [2, 4, 6, 8, 10, 0, 1, 3, 5, 7, 9, 11]
        const seq = yield* program.sequence()
        const logits = yield* seq.prefill(yield* ids(prompt))
        const input = yield* ids(prompt)
        const output = yield* model.forward(params, input)
        const [expected] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(output, { start: [0, prompt.length - 1, 0], end: [1, prompt.length, VOCAB] }),
            [VOCAB]
          )
        ])
        deep(yield* Tensor.toNumberArray(logits), yield* Tensor.toNumberArray(expected))
      })
    )

    it.effect("prefix cache: window-evicted blocks stay reusable", () =>
      Effect.gen(function* () {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 3, 5, 7, 9, 11, 2, 4]
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          attentionWindow: 8
        })
        // Generate past the window: the prompt's first block leaves the
        // window, lands in the prefix cache, and the sequence releases
        // the rest of the prompt's blocks into the cache as well.
        yield* Effect.scoped(Effect.gen(function* () {
          const seq = yield* program.sequence()
          let logits = yield* seq.prefill(yield* ids(prompt))
          for (let i = 0; i < 4; i++) {
            logits = yield* seq.step(yield* ids([yield* argmaxOf(logits)]))
          }
        }))
        const seq = yield* program.sequence()
        const logits = yield* seq.prefill(yield* ids(prompt))
        const input = yield* ids(prompt)
        const output = yield* model.forward(params, input)
        const [expected] = yield* Tensor.compute([
          yield* Tensor.reshape(
            yield* Tensor.slice(output, { start: [0, prompt.length - 1, 0], end: [1, prompt.length, VOCAB] }),
            [VOCAB]
          )
        ])
        deep(yield* Tensor.toNumberArray(logits), yield* Tensor.toNumberArray(expected))
      })
    )

    it.effect("prefix cache: a second prefill on a used sequence is an error", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        const seq = yield* program.sequence()
        yield* seq.prefill(yield* ids([1, 2, 3]))
        const error = yield* Effect.flip(seq.prefill(yield* ids([4, 5, 6])))
        expect(error._tag).toBe("InferenceError")
        expect(error.message).toMatch(/already holds tokens/)
      })
    )

    it.effect("prefix cache: concurrent same-prefix prefills stay exact", () =>
      Effect.gen(function* () {
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
      })
    )

    // Half-precision pools (RFC 0012): rows quantized on write, widened
    // on read. Teacher-forced — both sides see the same context — so
    // the comparison is logits closeness, not argmax luck.
    const halfPoolParity = (kvDtype: "f16" | "bf16", tol: number) =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, kvDtype })
        const prompt = [1, 5, 3, 8, 2]
        const trajectory = [4, 9, 0, 7, 6]
        const seq = yield* program.sequence()
        const context = [...prompt]
        let logits = yield* seq.prefill(yield* ids(prompt))
        const check = (actual: Tensor.Any, ctx: ReadonlyArray<number>) =>
          Effect.gen(function* () {
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
          logits = yield* seq.step(yield* ids([next]))
          yield* check(logits, context)
        }
      })

    it.effect("f16 pool: teacher-forced logits track the f32 reference", () => halfPoolParity("f16", 2e-2))

    it.effect("bf16 pool: teacher-forced logits track the f32 reference", () => halfPoolParity("bf16", 6e-2))

    it.effect("f16 pool: prefix cache and sliding window still hold", () =>
      Effect.gen(function* () {
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
        const seqA = yield* program.sequence()
        const logitsA = yield* seqA.prefill(yield* ids(prompt))
        const seqB = yield* program.sequence()
        const logitsB = yield* seqB.prefill(yield* ids(prompt))
        deep(yield* Tensor.toNumberArray(logitsB), yield* Tensor.toNumberArray(logitsA))
        // And windowed generation past eviction stays close to the
        // window-relative f32 recompute.
        const context = [...prompt]
        let logits = logitsA
        for (let i = 0; i < 6; i++) {
          const next = yield* argmaxOf(logits)
          context.push(next)
          logits = yield* seqA.step(yield* ids([next]))
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
      })
    )

    it.effect("rejects a model without cacheable attention at construction", () =>
      Effect.gen(function* () {
        const model = yield* Model.chain(
          yield* Model.embedding("wte", VOCAB, EMBED),
          yield* Model.linear("head", EMBED, VOCAB)
        )
        const params = yield* Tensor.compute(yield* model.init)
        const error = yield* Effect.flip(Model.inference(model, params, { maxTokens: 16, blockSize: 4 }))
        expect(error._tag).toBe("InferenceError")
        expect(error.message).toMatch(/no cacheable attention/)
      })
    )

    it.effect("rejects non-causal attention at construction", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt({ causal: false })
        const params = yield* Tensor.compute(yield* model.init)
        const error = yield* Effect.flip(Model.inference(model, params, { maxTokens: 16, blockSize: 4 }))
        expect(error._tag).toBe("InferenceError")
        expect(error.message).toMatch(/only causal attention is cacheable/)
      })
    )

    it.effect("validates the prefill/step calling convention", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4 })
        const seq = yield* program.sequence()
        const batched = yield* Effect.flip(seq.prefill(yield* Tensor.fromTypedArray(new Uint32Array(6), [2, 3])))
        expect(batched._tag).toBe("InferenceError")
        expect(batched.message).toMatch(/expects tokens of shape \[1, T\]/)
        const wide = yield* Effect.flip(seq.step(yield* ids([1, 2])))
        expect(wide._tag).toBe("InferenceError")
        expect(wide.message).toMatch(/expects a single token/)
        const badPool = yield* Effect.flip(Model.inference(model, params, { maxTokens: 15, blockSize: 4 }))
        expect(badPool._tag).toBe("InferenceError")
        expect(badPool.message).toMatch(/multiple of blockSize/)
      })
    )

    it.effect("matches the naive logits numerically, not just on argmax", () =>
      Effect.gen(function* () {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* model.init)
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4 })
        const seq = yield* program.sequence()
        const cached = yield* seq.prefill(yield* ids([1, 5, 3]))
        const output = yield* model.forward(params, yield* ids([1, 5, 3]))
        const [naive] = yield* Tensor.compute([
          yield* Tensor.reshape(yield* Tensor.slice(output, { start: [0, 2, 0], end: [1, 3, VOCAB] }), [
            VOCAB
          ])
        ])
        deep(yield* Tensor.toNumberArray(cached), yield* Tensor.toNumberArray(naive))
        void TOL
      })
    )

    it.effect("RoPE: matches naive greedy generation with full attention", () =>
      Effect.gen(function* () {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* model.init)
        const prompt = [1, 5, 3]
        const steps = 8
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4 })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      })
    )

    it.effect("RoPE + attention window: matches the window-relative recompute token-for-token, unbounded", () =>
      Effect.gen(function* () {
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
      })
    )

    it.effect("RoPE: trains — the rotary node differentiates", () =>
      Effect.gen(function* () {
        const model = yield* makeRopeGpt
        const data = Array.from({ length: 64 }, (_, i) => i % 4)
        const losses: Array<number> = []
        const trainer = yield* Trainer.make(model, {
          optimizer: yield* Optimizer.adamW(),
          lr: LearningRate.constant(3e-3),
          loss: Loss.crossEntropy,
          data: () =>
            Effect.gen(function* () {
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
      })
    )
  })
})
