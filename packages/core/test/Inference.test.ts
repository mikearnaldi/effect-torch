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
