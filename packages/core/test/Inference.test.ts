import { describe, expect } from "@effect/vitest"
import { Effect } from "effect"
import { LearningRate, Loss, Model, Optimizer, Runtime, Speculation, Tensor, Trainer } from "../src/index.ts"
import { MuseGlimmer } from "../src/models/index.ts"
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

// The RoPE variant has no absolute position table. Resetting a retained window
// to positions 0..window-1 preserves its relative offsets.
const makeRopeGpt = Effect.gen(function*() {
  const wte = yield* Model.embedding("wte", VOCAB, EMBED)
  const attn = yield* Model.multiHeadAttention("attn", EMBED, HEADS, { causal: true, rope: 10000 })
  const head = yield* Model.linear("head", EMBED, VOCAB)
  return yield* Model.chain(wte, attn, head)
})

const headsFirst = (hidden: Tensor.Any) =>
  Effect.gen(function*() {
    const [batch, rows] = hidden.shape
    return yield* Tensor.transpose(yield* Tensor.reshape(hidden, [batch!, rows!, 1, EMBED]), [0, 2, 1, 3])
  })

// parallelProposer builds a replayable block proposer for one named exposure
// on the base model. Different speculators can subscribe to different names.
// Setting withProbabilities adds the f32 probability rows that exercise the
// exact speculative-sampling acceptance path.
const parallelProposer = (tapName: string, withProbabilities = false) => {
  const buildBlock = (
    anchorTokens: Tensor.Any,
    tokenEmbedding: Tensor.Any,
    lmHead: Tensor.Any
  ) =>
    Effect.gen(function*() {
      const batch = anchorTokens.shape[0]!
      const tokens = yield* Tensor.concat([
        yield* Tensor.reshape(anchorTokens, [batch, 1]),
        yield* Tensor.zeros([batch, 3], { dtype: anchorTokens.dtype })
      ], { dim: 1 })
      const hidden = yield* Tensor.embedding(tokens, { weight: tokenEmbedding })
      const heads = yield* headsFirst(hidden)
      const attended = yield* Tensor.scaledDotProductAttention(heads, heads, heads, {
        causal: false,
        scale: 1 / Math.sqrt(EMBED)
      })
      const merged = yield* Tensor.reshape(yield* Tensor.transpose(attended, [0, 2, 1, 3]), [batch, 4, EMBED])
      const logits = yield* Tensor.matmul(merged, lmHead)
      return yield* Tensor.slice(logits, { start: [0, 1, 0], end: [batch, 4, VOCAB] })
    })
  const base: Omit<Speculation.ParallelBlock, "_tag" | "build" | "buildWithProbabilities"> = {
    params: [],
    vocabulary: VOCAB,
    maxDraftTokens: 3,
    hiddenTaps: [{ name: tapName, dtype: "f32", shape: ["Rows", EMBED] }],
    tokenEmbedding: { name: "token_embd.weight", dtype: "f32", shape: [VOCAB, EMBED] },
    lmHead: { name: "output.weight", dtype: "f32", shape: [EMBED, VOCAB] },
    currentBlockAttention: "Bidirectional",
    replay: (_params: Model.Params, [hidden]: ReadonlyArray<Tensor.Any>) =>
      Effect.gen(function*() {
        const heads = yield* headsFirst(hidden!)
        return [{ key: heads, value: heads }]
      })
  }
  const build = (_params: Model.Params, anchorTokens: Tensor.Any, tokenEmbedding: Tensor.Any, lmHead: Tensor.Any) =>
    Effect.gen(function*() {
      const candidates = yield* buildBlock(anchorTokens, tokenEmbedding, lmHead)
      return yield* Tensor.cast(yield* Tensor.argmax(candidates, -1), "u32")
    })
  if (!withProbabilities) {
    return Speculation.parallelBlock({ ...base, build })
  }
  return Speculation.parallelBlock({
    ...base,
    build,
    buildWithProbabilities: (_params, anchorTokens, tokenEmbedding, lmHead) =>
      Effect.gen(function*() {
        const candidates = yield* buildBlock(anchorTokens, tokenEmbedding, lmHead)
        return {
          tokenIds: yield* Tensor.cast(yield* Tensor.argmax(candidates, -1), "u32"),
          probabilityRows: yield* Tensor.softmax(candidates, { dims: [-1] })
        }
      })
  })
}

const makeParallelFixture = Effect.gen(function*() {
  const embedding = yield* Model.embedding("token_embd", VOCAB, EMBED)
  const attention = yield* Model.multiHeadAttention("attn", EMBED, 1, { causal: true, rope: 10_000 })
  const head = yield* Model.linear("output", EMBED, VOCAB)
  const embeddingCount = embedding.parameterSpecs.map(({ name }) => name).length
  const attentionCount = attention.parameterSpecs.map(({ name }) => name).length
  const model = yield* Model.define({
    parameterSpecs: [...embedding.parameterSpecs, ...attention.parameterSpecs, ...head.parameterSpecs],
    forward: (params, input) =>
      Effect.gen(function*() {
        let hidden = yield* embedding.forward(params.slice(0, embeddingCount), input)
        hidden = yield* Tensor.expose(hidden, Model.hiddenExposure(0))
        hidden = yield* attention.forward(
          params.slice(embeddingCount, embeddingCount + attentionCount),
          hidden
        )
        return yield* head.forward(params.slice(embeddingCount + attentionCount), hidden)
      })
  })
  const params = yield* Tensor.compute([
    ...yield* Model.initialize(embedding),
    ...yield* Model.initialize(attention),
    yield* Tensor.zeros([EMBED, VOCAB]),
    yield* Tensor.zeros([1, VOCAB])
  ])
  const proposer = parallelProposer(Model.hiddenExposure(0))
  return { model, params, proposer }
})

const ids = (tokens: ReadonlyArray<number>) => Tensor.fromTypedArray(new Uint32Array(tokens), [1, tokens.length])

const CONSTANT_BIAS_TARGET = [3, 2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2, -2.5, -3]

// A target whose logits are a fixed bias vector at every position, so its
// sampling distribution p = softmax(bias) is known exactly and independent of
// context. The attention layer stays live for the KV-state contract; its
// contribution to the logits is numerically negligible.
const makeConstantBiasModel = Effect.gen(function*() {
  const embedding = yield* Model.embedding("token_embd", VOCAB, EMBED)
  const attention = yield* Model.multiHeadAttention("attn", EMBED, 1, { causal: true })
  const embeddingCount = embedding.parameterSpecs.length
  const attentionCount = attention.parameterSpecs.length
  const model = yield* Model.define({
    parameterSpecs: [
      ...embedding.parameterSpecs,
      ...attention.parameterSpecs,
      { name: "output.weight", shape: [EMBED, VOCAB], initializer: { _tag: "Constant", value: 0 } },
      { name: "logit_bias.weight", shape: [VOCAB], initializer: { _tag: "Constant", value: 0 } }
    ],
    forward: (params, input) =>
      Effect.gen(function*() {
        const [batch, steps] = input.shape
        let hidden = yield* embedding.forward(params.slice(0, embeddingCount), input)
        hidden = yield* attention.forward(
          params.slice(embeddingCount, embeddingCount + attentionCount),
          hidden
        )
        hidden = yield* Tensor.expose(hidden, Model.hiddenExposure(0))
        const pooled = yield* Tensor.mean(hidden, { dims: [2], keepdims: true })
        const negligible = yield* Tensor.mul(pooled, yield* Tensor.constantLike(pooled, 1e-30))
        const bias = yield* Tensor.broadcastTo(
          yield* Tensor.reshape(params[embeddingCount + attentionCount + 1]!, [1, 1, VOCAB]),
          [batch!, steps!, VOCAB]
        )
        return yield* Tensor.add(bias, negligible)
      })
  })
  const params = yield* Tensor.compute([
    ...yield* Model.initialize(embedding),
    ...yield* Model.initialize(attention),
    yield* Tensor.zeros([EMBED, VOCAB]),
    yield* Tensor.fromTypedArray(Float32Array.from(CONSTANT_BIAS_TARGET), [VOCAB])
  ])
  return { model, params }
})

const historyLookup = (maxDraftTokens: number, minMatchTokens = 1) =>
  Effect.succeed({
    proposer: Speculation.historyLookup({
      vocabulary: VOCAB,
      maxDraftTokens,
      minMatchTokens,
      maxMatchTokens: 8
    }),
    maxDraftTokens
  })

const constantOneGpt = Effect.gen(function*() {
  const model = yield* makeGpt()
  const initialized = yield* Tensor.compute(yield* Model.initialize(model))
  const headWeight = yield* Tensor.zeros([EMBED, VOCAB])
  const biasValues = new Float32Array(VOCAB)
  biasValues[1] = 20
  const headBias = yield* Tensor.fromTypedArray(biasValues, [1, VOCAB])
  return { model, params: [...initialized.slice(0, -2), headWeight, headBias] }
})

const argmaxOf = (logits: Tensor.Any) =>
  Effect.gen(function*() {
    const values = yield* Tensor.toNumberArray(logits)
    if (Tensor.isTensor(logits)) yield* Tensor.clear(logits)
    return values.reduce((best, value, index) => (value > values[best] ? index : best), 0)
  })

// The reference performs greedy generation through the ordinary forward graph
// and recomputes the whole context at every step.
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

// This window-relative reference uses the pre-cache generation loop. Every
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

// Greedy generation through the inference artifact adds the prompt once, then
// runs one round per token with an argmax chooser.
const cachedGenerate = (
  program: Model.InferenceProgram,
  prompt: ReadonlyArray<number>,
  steps: number
) =>
  Effect.gen(function*() {
    const gen = yield* program.execution()
    const context = [...prompt]
    const entry = (yield* gen.add([yield* ids(prompt)]))[0]!
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

// Greedy generation through the native generation session uses one add, then
// one round per token. Unlike the execution session, this drives the backend's
// own chunked-prefill scheduler, exercising bucket selection on runtimes that
// honor prefill shape buckets.
const nativeGreedyGenerate = (
  program: Model.InferenceProgram,
  prompt: ReadonlyArray<number>,
  steps: number
) =>
  Effect.gen(function*() {
    const generation = yield* program.generation()
    const context = [...prompt]
    let pages = yield* generation.add([{ prompt: yield* ids(prompt) }])
    for (let i = 0; i < steps; i++) {
      context.push(...pages[0]!.tokens)
      if (i < steps - 1) {
        pages = yield* generation.step(pages.map(({ seq }) => ({ seq })))
      }
    }
    yield* generation.close()
    return context
  })

onDevices("Inference", () => (it) => {
  describe("Model.inference", () => {
    it("defines packed verification rows independently of physical batch", () => {
      const state: Runtime.DecodeStateRequest = {
        maxTokens: 64,
        blockSize: 4,
        kvDtype: "f32",
        batch: 2,
        packedCausalChains: { rowsPerSequence: 5 }
      }

      expect(state.batch).toBe(2)
      expect(state.packedCausalChains?.rowsPerSequence).toBe(5)
      expect(state.batch * state.packedCausalChains!.rowsPerSequence).toBe(10)
    })

    it.effect("preserves the last-token-row policy in the completed decode schema", () =>
      Effect.gen(function*() {
        const root = yield* Tensor.zeros([1, 2, 3])
        const compile = (lastTokenRow?: boolean) =>
          Tensor.compileDecodeProgram([root], {
            maxTokens: 4,
            blockSize: 2,
            kvDtype: "f32",
            batch: 1,
            lastTokenRow
          })

        const selected = yield* compile(true)
        expect(selected.lastTokenRow).toBe(true)
        expect(selected.handle.state?.lastTokenRow).toBe(true)
        expect(selected.outputs[0]?.shape).toEqual([3])

        const retained = yield* compile(false)
        expect(retained.lastTokenRow).toBe(false)
        expect(retained.handle.state?.lastTokenRow).toBe(false)
        expect(retained.outputs[0]?.shape).toEqual([1, 2, 3])

        const omitted = yield* compile()
        expect(omitted.lastTokenRow).toBeUndefined()
        expect(omitted.handle.state?.lastTokenRow).toBeUndefined()
      }))

    it.effect("samples add/step without publishing logits", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2
        })
        const generation = yield* program.generation()
        const reference = yield* program.execution()
        const prompts = [
          [1, 5, 3, 8, 2, 11, 4, 7, 6],
          [2, 4, 6, 8, 10, 0]
        ]
        const sampling: ReadonlyArray<Tensor.SamplingOptions> = [
          { temperature: 0, seed: 7 },
          { temperature: 0, seed: 11 }
        ]
        const referenceEntries: Array<{ readonly seq: Model.StatefulExecutionSeq; readonly logits: Tensor.Concrete }> =
          []
        const sampledEntries: Array<Model.TokenPage> = []
        const expectedAdd: Array<number> = []
        for (const [index, prompt] of prompts.entries()) {
          const expected = (yield* reference.add([yield* ids(prompt)]))[0]!
          referenceEntries.push(expected)
          expectedAdd.push(yield* Tensor.sample(expected.logits, sampling[index]!))
          yield* Tensor.clear(expected.logits)

          const actual = (yield* generation.add([{ prompt: yield* ids(prompt), sampling: sampling[index]! }]))[0]!
          sampledEntries.push(actual)
          expect(actual.tokens).toEqual([expectedAdd[index]])
          expect(yield* actual.seq.cursor()).toBe(prompt.length)
        }

        const referenceLogits = yield* reference.step(
          referenceEntries.map(({ seq }, index) => ({ seq, token: expectedAdd[index]! }))
        )
        const expectedStep: Array<number> = []
        for (const [index, logits] of referenceLogits.entries()) {
          expectedStep.push(yield* Tensor.sample(logits, sampling[index]!))
          yield* Tensor.clear(logits)
        }
        const actualStep = yield* generation.step(sampledEntries.map(({ seq }, index) => ({
          seq,
          sampling: sampling[index]!
        })))
        expect(actualStep.map((page) => page.tokens[0])).toEqual(expectedStep)
        for (const [index, entry] of sampledEntries.entries()) {
          expect(yield* entry.seq.cursor()).toBe(prompts[index]!.length + 1)
        }
        yield* reference.close()
        yield* generation.close()
      }))

    it.effect("generation owns pending tokens and terminal output policy", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2,
          sampling: { temperature: 0, seed: 17 }
        })
        const generation = yield* program.generation()
        const prompts = [[1, 2, 3], [4, 5, 6, 7]]
        const pages = yield* generation.add([
          { prompt: yield* ids(prompts[0]!), maxTokens: 1 },
          { prompt: yield* ids(prompts[1]!) }
        ])
        expect(pages).toHaveLength(2)
        expect(pages[0]!.tokens).toHaveLength(1)
        expect(pages[0]!.stopReason).toBe("maxTokens")
        expect(pages[1]!.tokens).toHaveLength(1)
        expect(yield* pages[0]!.seq.cursor()).toBe(prompts[0]!.length)
        expect(yield* pages[1]!.seq.cursor()).toBe(prompts[1]!.length)

        const before = yield* pages[1]!.seq.cursor()
        const terminal = yield* Effect.flip(generation.step([
          { seq: pages[1]!.seq },
          { seq: pages[0]!.seq }
        ]))
        expect(terminal.message).toContain("terminal")
        expect(yield* pages[1]!.seq.cursor()).toBe(before)

        const [next] = yield* generation.step([{ seq: pages[1]!.seq }])
        expect(next?.tokens).toHaveLength(1)
        expect(yield* pages[1]!.seq.cursor()).toBe(prompts[1]!.length + 1)
        yield* pages[0]!.seq.finish()
        const [eos] = yield* generation.add([{
          prompt: yield* ids(prompts[0]!),
          eosTokens: [pages[0]!.tokens[0]!]
        }])
        expect(eos?.stopReason).toBe("eos")
        expect(yield* eos!.seq.cursor()).toBe(prompts[0]!.length)
        yield* generation.close()
      }))

    it.effect("batched generation admission is all-or-nothing", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2,
          sampling: { temperature: 0, seed: 1 }
        })
        const generation = yield* program.generation()
        const error = yield* Effect.flip(generation.add([
          { prompt: yield* ids([1, 2]) },
          { prompt: yield* ids([3, 4]) },
          { prompt: yield* ids([5, 6]) }
        ]))
        expect(error.message).toContain("free lanes")
        expect(yield* generation.live()).toBe(0)
        const invalid = yield* Effect.flip(generation.add([
          { prompt: yield* ids([1, 2]) },
          { prompt: yield* Tensor.fromTypedArray(new Uint32Array([3, 4]), [2, 1]) }
        ]))
        expect(invalid.message).toContain("shape [1, T]")
        expect(yield* generation.live()).toBe(0)
        const budget = yield* Effect.flip(generation.add([{
          prompt: yield* ids([1, 2]),
          maxTokens: 0x1_0000_0000
        }]))
        expect(budget.message).toContain("unsigned 32-bit")
        expect(yield* generation.live()).toBe(0)
        const pages = yield* generation.add([
          { prompt: yield* ids([1, 2]) },
          { prompt: yield* ids([3, 4]) }
        ])
        expect(pages).toHaveLength(2)
        expect(yield* generation.live()).toBe(2)
        yield* generation.close()
      }))

    it.effect("generation sampling follows sequence identity when step order changes", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const config = {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2,
          sampling: { temperature: 1, seed: 29 }
        } as const
        const programA = yield* Model.inference(model, params, config)
        const programB = yield* Model.inference(model, params, config)
        const generationA = yield* programA.generation()
        const generationB = yield* programB.generation()
        const prompts = [yield* ids([1, 2, 3]), yield* ids([4, 5, 6])]
        const pagesA = yield* generationA.add(prompts.map((prompt) => ({ prompt })))
        const pagesB = yield* generationB.add(prompts.map((prompt) => ({ prompt })))
        expect(pagesA.map((page) => page.tokens)).toEqual(pagesB.map((page) => page.tokens))

        const nextA = yield* generationA.step(pagesA.map(({ seq }) => ({ seq })))
        const nextB = yield* generationB.step([...pagesB].reverse().map(({ seq }) => ({ seq })))
        expect(nextA[0]!.tokens).toEqual(nextB[1]!.tokens)
        expect(nextA[1]!.tokens).toEqual(nextB[0]!.tokens)
        yield* generationA.close()
        yield* generationB.close()
      }))

    it.effect("batch size one uses the sampled fixed-lane path", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 3 }
        })
        const generation = yield* program.generation()
        const [first] = yield* generation.add([{ prompt: yield* ids([1, 2, 3]) }])
        const [next] = yield* generation.step([{ seq: first!.seq }])
        expect(first!.tokens).toHaveLength(1)
        expect(next!.tokens).toHaveLength(1)
        expect(yield* first!.seq.cursor()).toBe(4)
        yield* generation.close()
      }))

    it.effect("exact chain speculation matches greedy ordinary generation", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 3 })
        const ordinaryProgram = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 7 }
        })
        const speculativeProgram = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 7 },
          speculation: { proposer, maxDraftTokens: 3 }
        })
        const ordinary = yield* ordinaryProgram.generation()
        const speculative = yield* speculativeProgram.generation()
        let ordinaryPage = (yield* ordinary.add([{ prompt: yield* ids([1, 2, 3]) }]))[0]!
        const speculativeFirst = (yield* speculative.add([{ prompt: yield* ids([1, 2, 3]) }]))[0]!
        expect(speculativeFirst.tokens).toEqual(ordinaryPage.tokens)
        const expected: Array<number> = []
        for (let index = 0; index < 4; index++) {
          ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
          expected.push(...ordinaryPage.tokens)
        }
        const speculativePage = (yield* speculative.step([{ seq: speculativeFirst.seq }]))[0]!
        expect(speculativePage.tokens).toEqual(expected)
        expect(speculativePage.tokens).toHaveLength(4)
        expect(yield* speculativeFirst.seq.cursor()).toBe(7)
        const expectedNext: Array<number> = []
        for (let index = 0; index < 4; index++) {
          ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
          expectedNext.push(...ordinaryPage.tokens)
        }
        const speculativeNext = (yield* speculative.step([{ seq: speculativeFirst.seq }]))[0]!
        expect(speculativeNext.tokens).toEqual(expectedNext)
        expect(yield* speculativeFirst.seq.cursor()).toBe(11)
        yield* ordinary.close()
        yield* speculative.close()
      }))

    it.effect("parallel replay matches greedy generation across prompt chunks and rounds", () =>
      Effect.gen(function*() {
        const { model, params, proposer } = yield* makeParallelFixture
        const base = {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 17 }
        } as const
        const ordinaryProgram = yield* Model.inference(model, params, base)
        const parallelProgram = yield* Model.inference(model, params, {
          ...base,
          speculation: { proposer, maxDraftTokens: 2 }
        })
        const ordinary = yield* ordinaryProgram.generation()
        const parallel = yield* parallelProgram.generation()
        let ordinaryPage = (yield* ordinary.add([{ prompt: yield* ids([1, 2, 3, 4, 5, 6]) }]))[0]!
        const parallelFirst = (yield* parallel.add([{ prompt: yield* ids([1, 2, 3, 4, 5, 6]) }]))[0]!
        expect(parallelFirst.tokens).toEqual(ordinaryPage.tokens)

        for (let round = 0; round < 2; round++) {
          const parallelPage = (yield* parallel.step([{ seq: parallelFirst.seq }]))[0]!
          expect(parallelPage.tokens).toHaveLength(3)
          const expected: Array<number> = []
          for (let index = 0; index < parallelPage.tokens.length; index++) {
            ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
            expected.push(...ordinaryPage.tokens)
          }
          expect(parallelPage.tokens).toEqual(expected)
        }
        expect(yield* parallelFirst.seq.cursor()).toBe(yield* ordinaryPage.seq.cursor())
        const diagnostics = yield* parallelProgram.diagnostics()
        expect(diagnostics.proposedTokens).toBe(4n)
        expect(diagnostics.acceptedTokens).toBe(4n)
        yield* ordinary.close()
        yield* parallel.close()
      }))

    it.effect("parallel target-sample matching is pathwise equal to ordinary seeded sampling", () =>
      Effect.gen(function*() {
        const { model, params, proposer } = yield* makeParallelFixture
        const base = {
          maxTokens: 128,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0.8, topK: 8, topP: 0.9, seed: 0x5eed_1234n }
        } as const
        const ordinary = yield* (yield* Model.inference(model, params, base)).generation()
        const parallelProgram = yield* Model.inference(model, params, {
          ...base,
          speculation: { proposer, maxDraftTokens: 2 }
        })
        const parallel = yield* parallelProgram.generation()
        let ordinaryPage = (yield* ordinary.add([{ prompt: yield* ids([1, 2, 3, 4, 5, 6]) }]))[0]!
        let parallelPage = (yield* parallel.add([{ prompt: yield* ids([1, 2, 3, 4, 5, 6]) }]))[0]!
        expect(parallelPage.tokens).toEqual(ordinaryPage.tokens)

        for (let round = 0; round < 8; round++) {
          parallelPage = (yield* parallel.step([{ seq: parallelPage.seq }]))[0]!
          const expected: Array<number> = []
          while (expected.length < parallelPage.tokens.length) {
            ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
            expected.push(...ordinaryPage.tokens)
          }
          expect(parallelPage.tokens).toEqual(expected)
        }
        expect(yield* parallelPage.seq.cursor()).toBe(yield* ordinaryPage.seq.cursor())
        const diagnostics = yield* parallelProgram.diagnostics()
        expect(diagnostics.proposedTokens).toBeGreaterThan(0n)
        expect(diagnostics.acceptedTokens).toBeLessThan(diagnostics.proposedTokens)
        yield* ordinary.close()
        yield* parallel.close()
      }))

    it.effect("history lookup is pathwise equal to ordinary seeded sampling", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const speculation = yield* historyLookup(3)
        const base = {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0.8, topK: 8, topP: 0.9, seed: 0x1234_5678n }
        } as const
        const ordinary = yield* (yield* Model.inference(model, params, base)).generation()
        const lookup = yield* (yield* Model.inference(model, params, { ...base, speculation })).generation()
        let ordinaryPage = (yield* ordinary.add([{ prompt: yield* ids([1, 2, 1, 2]) }]))[0]!
        let lookupPage = (yield* lookup.add([{ prompt: yield* ids([1, 2, 1, 2]) }]))[0]!
        expect(lookupPage.tokens).toEqual(ordinaryPage.tokens)
        for (let round = 0; round < 5; round++) {
          lookupPage = (yield* lookup.step([{ seq: lookupPage.seq }]))[0]!
          const expected: Array<number> = []
          while (expected.length < lookupPage.tokens.length) {
            ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
            expected.push(...ordinaryPage.tokens)
          }
          expect(lookupPage.tokens).toEqual(expected)
        }
        yield* ordinary.close()
        yield* lookup.close()
      }))

    it.effect("history lookup emits one token without a match and multiple tokens for repetition", () =>
      Effect.gen(function*() {
        const { model, params } = yield* constantOneGpt
        const noMatchProgram = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 9 },
          speculation: yield* historyLookup(3, 3)
        })
        const noMatch = yield* noMatchProgram.generation()
        const noMatchFirst = (yield* noMatch.add([{ prompt: yield* ids([2, 3, 4]) }]))[0]!
        const noMatchPage = (yield* noMatch.step([{ seq: noMatchFirst.seq }]))[0]!
        expect(noMatchPage.tokens).toEqual([1])
        yield* noMatch.close()

        const repeatedProgram = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 9 },
          speculation: yield* historyLookup(3)
        })
        const repeated = yield* repeatedProgram.generation()
        const repeatedFirst = (yield* repeated.add([{ prompt: yield* ids([1, 1, 1, 1]) }]))[0]!
        const repeatedPage = (yield* repeated.step([{ seq: repeatedFirst.seq }]))[0]!
        // The longest prior suffix has one token following it, then verification
        // contributes the ordinary target bonus.
        expect(repeatedPage.tokens).toEqual([1, 1])
        yield* repeated.close()
      }))

    it.effect("history lookup supports ragged batches, sparse lanes, and output budgets", () =>
      Effect.gen(function*() {
        const { model, params } = yield* constantOneGpt
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2,
          sampling: { temperature: 0, seed: 13 },
          speculation: yield* historyLookup(3)
        })
        const generation = yield* program.generation()
        const first = yield* generation.add([
          { prompt: yield* ids([1, 1, 1]), maxTokens: 3 },
          { prompt: yield* ids([2, 3, 4]), maxTokens: 8 }
        ])
        const pages = yield* generation.step(first.map(({ seq }) => ({ seq })))
        expect(pages[0]!.tokens).toEqual([1, 1])
        expect(pages[0]!.stopReason).toBe("maxTokens")
        expect(pages[1]!.tokens).toEqual([1])
        const idleCursor = yield* pages[1]!.seq.cursor()
        const sparse = (yield* generation.step([{ seq: pages[1]!.seq }]))[0]!
        expect(sparse.tokens.length).toBeGreaterThan(0)
        expect(yield* pages[1]!.seq.cursor()).toBe(idleCursor + sparse.tokens.length)
        yield* generation.close()
      }))

    it.effect("speculative pages stop exactly at the remaining output budget", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 4 })
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 11 },
          speculation: { proposer, maxDraftTokens: 4 }
        })
        const generation = yield* program.generation()
        const first = (yield* generation.add([{ prompt: yield* ids([1, 2]), maxTokens: 3 }]))[0]!
        const page = (yield* generation.step([{ seq: first.seq }]))[0]!
        expect(page.tokens).toHaveLength(2)
        expect(page.stopReason).toBe("maxTokens")
        expect(yield* first.seq.cursor()).toBe(4)
        yield* generation.close()
      }))

    it.effect("one speculative batch publishes different nonzero page lengths", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 3 })
        const program = yield* Model.inference(model, params, {
          maxTokens: 128,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2,
          sampling: { temperature: 0, seed: 13 },
          speculation: { proposer, maxDraftTokens: 3 }
        })
        const generation = yield* program.generation()
        const first = yield* generation.add([
          { prompt: yield* ids([1, 2, 3]), maxTokens: 2 },
          { prompt: yield* ids([4, 5, 6]), maxTokens: 4 }
        ])
        const pages = yield* generation.step(first.map(({ seq }) => ({ seq })))
        expect(pages.map((page) => page.tokens.length)).toEqual([1, 3])
        expect(pages.map((page) => page.stopReason)).toEqual(["maxTokens", "maxTokens"])
        const terminal = yield* Effect.flip(generation.step([{ seq: pages[0]!.seq }]))
        expect(terminal.message).toMatch(/terminal/)
        expect(yield* pages[0]!.seq.cursor()).toBe(4)
        expect(yield* pages[1]!.seq.cursor()).toBe(6)
        yield* generation.close()
      }))

    it.effect("speculative EOS cuts discard the computed suffix and make the lane terminal", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 3 })
        const speculativeProgram = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 17 },
          speculation: { proposer, maxDraftTokens: 3 }
        })
        let selectedPrompt: ReadonlyArray<number> | undefined
        let baselinePage: Model.TokenPage | undefined
        let eosIndex = -1
        for (let start = 1; start <= 32 && baselinePage === undefined; start++) {
          const prompt = [start % VOCAB, (start + 1) % VOCAB, (start + 2) % VOCAB]
          const baseline = yield* speculativeProgram.generation()
          const baselineFirst = (yield* baseline.add([{ prompt: yield* ids(prompt) }]))[0]!
          const candidate = (yield* baseline.step([{ seq: baselineFirst.seq }]))[0]!
          const index = candidate.tokens
            .slice(0, -1)
            .findIndex((token) => token !== baselineFirst.tokens[0])
          yield* baseline.close()
          if (index >= 0) {
            selectedPrompt = prompt
            baselinePage = candidate
            eosIndex = index
          }
        }
        expect(selectedPrompt).toBeDefined()
        expect(baselinePage).toBeDefined()
        if (selectedPrompt === undefined || baselinePage === undefined) {
          throw new Error("test model did not produce an interior EOS token")
        }
        const eos = baselinePage.tokens[eosIndex]!

        const speculative = yield* speculativeProgram.generation()
        const first = (yield* speculative.add([{ prompt: yield* ids(selectedPrompt), eosTokens: [eos] }]))[0]!
        const page = (yield* speculative.step([{ seq: first.seq }]))[0]!
        expect(page.tokens).toEqual(baselinePage.tokens.slice(0, eosIndex + 1))
        expect(page.stopReason).toBe("eos")
        const cursor = 3 + page.tokens.length
        expect(yield* first.seq.cursor()).toBe(cursor)
        expect(cursor).toBeLessThan(3 + baselinePage.tokens.length)
        const terminal = yield* Effect.flip(speculative.step([{ seq: first.seq }]))
        expect(terminal.message).toMatch(/terminal/)
        expect(yield* first.seq.cursor()).toBe(cursor)
        yield* speculative.close()
      }))

    it.effect("reports native speculative phase, acceptance, and pool diagnostics", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 2 })
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 19n },
          speculation: { proposer, maxDraftTokens: 2 }
        })
        const generation = yield* program.generation()
        const first = (yield* generation.add([{ prompt: yield* ids([1, 2, 3]) }]))[0]!
        yield* generation.step([{ seq: first.seq }])
        const diagnostics = yield* program.diagnostics()
        expect(diagnostics.roundsStarted).toBe(2n)
        expect(diagnostics.roundsCompleted).toBe(2n)
        expect(diagnostics.speculativeRounds).toBe(1n)
        expect(diagnostics.proposedTokens).toBe(2n)
        expect(diagnostics.acceptedTokens).toBe(2n)
        expect(diagnostics.emittedTokens).toBe(4n)
        expect(diagnostics.draftNanos).toBeGreaterThan(0n)
        expect(diagnostics.verificationNanos).toBeGreaterThan(0n)
        expect(diagnostics.acceptedLengthHistogram[2]).toBe(1n)
        expect(diagnostics.targetPoolHighWaterBlocks).toBeGreaterThan(0n)
        expect(diagnostics.proposerPoolHighWaterBlocks).toBeGreaterThan(0n)
        yield* generation.close()
      }))

    it.effect("positive-temperature speculative replay ignores request order", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 3 })
        const config = {
          maxTokens: 128,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2,
          sampling: { temperature: 0.8, topK: 8, topP: 0.9, seed: 37 },
          speculation: { proposer, maxDraftTokens: 3 }
        } as const
        const programA = yield* Model.inference(model, params, config)
        const programB = yield* Model.inference(model, params, config)
        const generationA = yield* programA.generation()
        const generationB = yield* programB.generation()
        const firstA = yield* generationA.add([
          { prompt: yield* ids([1, 2, 3]) },
          { prompt: yield* ids([4, 5, 6]) }
        ])
        const firstB = yield* generationB.add([
          { prompt: yield* ids([1, 2, 3]) },
          { prompt: yield* ids([4, 5, 6]) }
        ])
        const pagesA = yield* generationA.step(firstA.map(({ seq }) => ({ seq })))
        const pagesB = yield* generationB.step([...firstB].reverse().map(({ seq }) => ({ seq })))
        expect(pagesA[0]!.tokens).toEqual(pagesB[1]!.tokens)
        expect(pagesA[1]!.tokens).toEqual(pagesB[0]!.tokens)
        expect(pagesA[0]!.tokens.length).toBeGreaterThan(0)
        expect(pagesA[1]!.tokens.length).toBeGreaterThan(0)
        yield* generationA.close()
        yield* generationB.close()
      }))

    it.effect("round sampling overrides preserve unspecified artifact defaults", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const config = {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 1, topK: 0, topP: 1, seed: 7n }
        } as const
        const programA = yield* Model.inference(model, params, config)
        const programB = yield* Model.inference(model, params, config)
        const generationA = yield* programA.generation()
        const generationB = yield* programB.generation()
        let pageA = (yield* generationA.add([{ prompt: yield* ids([1, 2, 3]) }]))[0]!
        let pageB = (yield* generationB.add([{ prompt: yield* ids([1, 2, 3]) }]))[0]!
        expect(pageA.tokens).toEqual(pageB.tokens)
        for (let round = 0; round < 4; round++) {
          pageA = (yield* generationA.step([{ seq: pageA.seq, sampling: { topK: 4 } }]))[0]!
          pageB = (yield* generationB.step([{
            seq: pageB.seq,
            sampling: { ...config.sampling, topK: 4 }
          }]))[0]!
          expect(pageA.tokens).toEqual(pageB.tokens)
        }
        yield* generationA.close()
        yield* generationB.close()
      }))

    it.effect("preserves high seed bits through native generation", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const generate = (seed: bigint) =>
          Effect.gen(function*() {
            const program = yield* Model.inference(model, params, {
              maxTokens: 64,
              blockSize: 4,
              prefillChunks: [4],
              batchSize: 1,
              sampling: { temperature: 1, seed }
            })
            const generation = yield* program.generation()
            let page = (yield* generation.add([{ prompt: yield* ids([1, 2, 3]) }]))[0]!
            const tokens = [...page.tokens]
            for (let round = 0; round < 7; round++) {
              page = (yield* generation.step([{ seq: page.seq }]))[0]!
              tokens.push(...page.tokens)
            }
            yield* generation.close()
            return tokens
          })
        const low = 1n
        const high = (1n << 63n) | low
        const lowA = yield* generate(low)
        const lowB = yield* generate(low)
        const highA = yield* generate(high)
        const highB = yield* generate(high)
        expect(lowA).toEqual(lowB)
        expect(highA).toEqual(highB)
        expect(lowA).not.toEqual(highA)
      }))

    it.effect("speculative rounds accept sparse physical lanes", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 2 })
        const program = yield* Model.inference(model, params, {
          maxTokens: 128,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2,
          sampling: { temperature: 0, seed: 19 },
          speculation: { proposer, maxDraftTokens: 2 }
        })
        const generation = yield* program.generation()
        const first = yield* generation.add([
          { prompt: yield* ids([1, 2, 3]) },
          { prompt: yield* ids([4, 5, 6]) }
        ])
        const secondCursor = yield* first[1]!.seq.cursor()
        const page = (yield* generation.step([{ seq: first[0]!.seq }]))[0]!
        expect(page.tokens).toHaveLength(3)
        expect(yield* first[0]!.seq.cursor()).toBe(6)
        expect(yield* first[1]!.seq.cursor()).toBe(secondCursor)
        yield* generation.close()
      }))

    it.effect("speculation shortens a page at the remaining state capacity", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 4 })
        const program = yield* Model.inference(model, params, {
          maxTokens: 8,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 23 },
          speculation: { proposer, maxDraftTokens: 4 }
        })
        const generation = yield* program.generation()
        const first = (yield* generation.add([{ prompt: yield* ids([1, 2, 3, 4, 5]) }]))[0]!
        const page = (yield* generation.step([{ seq: first.seq }]))[0]!
        expect(page.tokens).toHaveLength(3)
        expect(yield* first.seq.cursor()).toBe(8)
        yield* generation.close()
      }))

    it.effect("speculation supports i64 token programs", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 2 })
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          tokenDtype: "i64",
          sampling: { temperature: 0, seed: 31 },
          speculation: { proposer, maxDraftTokens: 2 }
        })
        const generation = yield* program.generation()
        const prompt = yield* Tensor.fromTypedArray(BigInt64Array.from([1n, 2n, 3n]), [1, 3])
        const first = (yield* generation.add([{ prompt }]))[0]!
        const page = (yield* generation.step([{ seq: first.seq }]))[0]!
        expect(page.tokens).toHaveLength(3)
        expect(yield* first.seq.cursor()).toBe(6)
        yield* generation.close()
      }))

    it.effect("greedy correction from a different draft preserves target tokens", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const targetParams = yield* Tensor.compute(yield* Model.initialize(model))
        const draftParams = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, draftParams, { vocabulary: VOCAB, maxDraftTokens: 3 })
        const ordinaryProgram = yield* Model.inference(model, targetParams, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 5 }
        })
        const speculativeProgram = yield* Model.inference(model, targetParams, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 5 },
          speculation: { proposer, maxDraftTokens: 3 }
        })
        const ordinary = yield* ordinaryProgram.generation()
        const speculative = yield* speculativeProgram.generation()
        let ordinaryPage = (yield* ordinary.add([{ prompt: yield* ids([2, 3, 4]) }]))[0]!
        const speculativeFirst = (yield* speculative.add([{ prompt: yield* ids([2, 3, 4]) }]))[0]!
        const expected: Array<number> = []
        for (let index = 0; index < 4; index++) {
          ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
          expected.push(...ordinaryPage.tokens)
        }
        const actual = (yield* speculative.step([{ seq: speculativeFirst.seq }]))[0]!
        expect(actual.tokens).toEqual(expected.slice(0, actual.tokens.length))
        expect(yield* speculativeFirst.seq.cursor()).toBe(3 + actual.tokens.length)
        yield* ordinary.close()
        yield* speculative.close()
      }))

    it.effect("rejects unsupported speculative schedules and draft limits before compilation", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const proposer = Speculation.autoregressive(model, params, { vocabulary: VOCAB, maxDraftTokens: 2 })
        const limit = yield* Effect.flip(Model.inference(model, params, {
          maxTokens: 32,
          prefillChunks: [4],
          speculation: { proposer, maxDraftTokens: 3 }
        }))
        expect(limit.message).toMatch(/maxDraftTokens/)
        const adaptive = yield* Effect.flip(Model.inference(model, params, {
          maxTokens: 32,
          prefillChunks: [4],
          speculation: { proposer, maxDraftTokens: 2, schedule: "adaptive" }
        }))
        expect(adaptive.message).toMatch(/adaptive/)
      }))

    it.effect("lane refill does not change a replacement sequence's RNG identity", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const config = {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2,
          sampling: { temperature: 1, seed: 41 }
        } as const
        const programA = yield* Model.inference(model, params, config)
        const programB = yield* Model.inference(model, params, config)
        const generationA = yield* programA.generation()
        const generationB = yield* programB.generation()
        const promptsA = yield* generationA.add([
          { prompt: yield* ids([1, 2]) },
          { prompt: yield* ids([3, 4]) }
        ])
        const promptsB = yield* generationB.add([
          { prompt: yield* ids([1, 2]) },
          { prompt: yield* ids([3, 4]) }
        ])
        yield* promptsA[0]!.seq.finish()
        yield* promptsB[1]!.seq.finish()
        const [replacementA] = yield* generationA.add([{ prompt: yield* ids([5, 6, 7]) }])
        const [replacementB] = yield* generationB.add([{ prompt: yield* ids([5, 6, 7]) }])
        expect(replacementA!.tokens).toEqual(replacementB!.tokens)
        const [nextA] = yield* generationA.step([{ seq: replacementA!.seq }])
        const [nextB] = yield* generationB.step([{ seq: replacementB!.seq }])
        expect(nextA!.tokens).toEqual(nextB!.tokens)
        yield* generationA.close()
        yield* generationB.close()
      }))

    it.effect("legacy window retains history for mixed local/full attention", () =>
      Effect.gen(function*() {
        const qExemplar = yield* Tensor.zeros([1, 4, 4, 1])
        const kExemplar = yield* Tensor.zeros([1, 2, 4, 1])
        const vExemplar = yield* Tensor.zeros([1, 2, 4, 1])
        const q = yield* Tensor.makeInput(0, qExemplar)
        const k = yield* Tensor.makeInput(1, kExemplar)
        const v = yield* Tensor.makeInput(2, vExemplar)
        const local = yield* Tensor.scaledDotProductAttention(q, k, v, { causal: true, window: 2 })
        const full = yield* Tensor.scaledDotProductAttention(q, k, v, { causal: true, window: null })
        const program = yield* Tensor.compileDecodeProgram([local, full], {
          maxTokens: 8,
          blockSize: 4,
          kvDtype: "f32",
          window: 4,
          batch: 1
        })
        expect(program.kvHeads).toBe(2)
        expect(program.headDim).toBe(1)
        expect(program.window).toBeUndefined()
        expect(program.handle.state?.window).toBeUndefined()

        const pool = yield* Tensor.makeKvPool(
          program.layers,
          program.kvHeads,
          program.headDim,
          program.maxTokens,
          program.blockSize
        )
        const sequence = yield* Tensor.makeKvSequence(pool)
        const tensor = (values: ReadonlyArray<number>, shape: ReadonlyArray<number>) =>
          Tensor.fromTypedArray(new Float32Array(values), shape)
        const zeroQ = () => tensor(Array(16).fill(0), [1, 4, 4, 1])
        const zeroK = () => tensor(Array(8).fill(0), [1, 2, 4, 1])

        const first = yield* Tensor.runDecodeProgram(
          program,
          [yield* zeroQ(), yield* zeroK(), yield* tensor([1, 2, 4, 8, 10, 20, 40, 80], [1, 2, 4, 1])],
          sequence,
          [1, 2, 3, 4]
        )
        for (const output of first) yield* Tensor.clear(output)

        const second = yield* Tensor.runDecodeProgram(
          program,
          [yield* zeroQ(), yield* zeroK(), yield* tensor([16, 0, 0, 0, 160, 0, 0, 0], [1, 2, 4, 1])],
          sequence,
          [5]
        )
        const localValues = yield* Tensor.toNumberArray(second[0])
        const fullValues = yield* Tensor.toNumberArray(second[1])
        deep([localValues[0], localValues[4], localValues[8], localValues[12]], [12, 12, 120, 120])
        deep([fullValues[0], fullValues[4], fullValues[8], fullValues[12]], [6.2, 6.2, 62, 62])
        for (const output of second) yield* Tensor.clear(output)

        const third = yield* Tensor.runDecodeProgram(
          program,
          [yield* zeroQ(), yield* zeroK(), yield* tensor(Array(8).fill(0), [1, 2, 4, 1])],
          sequence,
          [6, 7, 8]
        )
        for (const output of third) yield* Tensor.clear(output)
        const capacityError = yield* Effect.flip(
          Tensor.runDecodeProgram(
            program,
            [yield* zeroQ(), yield* zeroK(), yield* tensor(Array(8).fill(0), [1, 2, 4, 1])],
            sequence,
            [9]
          )
        )
        expect(capacityError.message).toContain("capacity")
        yield* Tensor.releaseKvSequence(sequence)
      }))

    it.effect("matches naive greedy generation token-for-token across pool block boundaries", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 5, 3]
        const steps = 9 // context grows to 12, crossing block boundaries (blockSize 4)
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4, prefillChunks: [4] })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
        expect(cached.length).toBe(prompt.length + steps)
      }))

    it.effect("serves every prompt length from the two eagerly compiled programs", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunks: [4] })
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 5, 3, 8, 2, 11, 4, 7, 6] // 3 chunks of 4: 4 + 4 + 1(padded)
        const steps = 6
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunks: [4] })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      }))

    it.effect("compiles one prefill program per chunk shape with generation parity", () =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const requests: Array<Runtime.InferenceCompileRequest> = []
        const recording: Runtime.RuntimeService = {
          ...runtime,
          extensions: {
            ...runtime.extensions,
            inference: {
              ...runtime.extensions.inference,
              compile: (request) => {
                requests.push(request)
                return runtime.extensions.inference.compile(request)
              }
            }
          }
        }
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        // Chunk widths are validated, deduplicated, and sorted ascending.
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4, 2, 2],
          sampling: { temperature: 0, seed: 7 }
        }).pipe(Effect.provideService(Runtime.Runtime, recording))
        expect(requests.at(-1)?.target.prefill).toHaveLength(2)
        // Multi-chunk greedy generation matches the ordinary forward
        // reference through both the execution and generation sessions.
        const prompt = [1, 5, 3, 8, 2, 11, 4, 7, 6]
        const steps = 6
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
        const generated = yield* nativeGreedyGenerate(program, prompt, steps)
        expect(generated).toEqual(naive)
        // Short prompts that fit one small chunk generate identically.
        const shortPrompt = [2, 4]
        const shortNaive = yield* naiveGenerate(model, params, shortPrompt, steps)
        const shortGenerated = yield* nativeGreedyGenerate(program, shortPrompt, steps)
        expect(shortGenerated).toEqual(shortNaive)

        requests.length = 0
        yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunks: [4] }).pipe(
          Effect.provideService(Runtime.Runtime, recording)
        )
        expect(requests.at(-1)?.target.prefill).toHaveLength(1)
      }))

    it.effect("parallel speculation preserves the exact target distribution", () =>
      Effect.gen(function*() {
        // p = softmax(bias) is constant per position by construction; the
        // emitted sequence must be an exact draw from p whether the draft
        // supplies a causal q or uses target-sample matching. Identical seeds
        // must replay identically in both branches.
        const { model, params } = yield* makeConstantBiasModel
        const sampling = { temperature: 1, topK: 8, topP: 0.9, seed: 0x5eed_5eedn } as const
        const base = {
          maxTokens: 2048,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling
        } as const
        const samples = 600
        // The engine's effective distribution: top-k truncate, temperature
        // softmax, top-p cutoff at cumulative >= topP of the pre-cutoff
        // total, renormalize. Replicated exactly from the sampler contract.
        const sorted = CONSTANT_BIAS_TARGET
          .map((value, token) => ({ token, value }))
          .sort((a, b) => b.value - a.value)
          .slice(0, sampling.topK)
        const max = sorted[0]!.value
        const weights = sorted.map(({ token, value }) => ({
          token,
          weight: Math.exp((value - max) / sampling.temperature)
        }))
        const total = weights.reduce((sum, { weight }) => sum + weight, 0)
        let cumulative = 0
        let retained = weights.length
        for (const [index, { weight }] of weights.entries()) {
          cumulative += weight
          if (cumulative >= sampling.topP * total) {
            retained = index + 1
            break
          }
        }
        const effective = new Array<number>(VOCAB).fill(0)
        const retainedTotal = weights.slice(0, retained).reduce((sum, { weight }) => sum + weight, 0)
        for (const { token, weight } of weights.slice(0, retained)) {
          effective[token] = weight / retainedTotal
        }

        for (const withProbabilities of [false, true]) {
          const proposer = parallelProposer(Model.hiddenExposure(0), withProbabilities)
          const generate = Effect.gen(function*() {
            const generation = yield* (yield* Model.inference(model, params, {
              ...base,
              speculation: { proposer, maxDraftTokens: 3 }
            })).generation()
            const tokens: Array<number> = []
            let page = (yield* generation.add([{ prompt: yield* ids([1, 2, 3]) }]))[0]!
            tokens.push(...page.tokens)
            while (tokens.length < samples) {
              page = (yield* generation.step([{ seq: page.seq }]))[0]!
              tokens.push(...page.tokens)
            }
            yield* generation.close()
            return tokens.slice(0, samples)
          })
          const first = yield* generate
          const second = yield* generate
          expect(second).toEqual(first)

          const histogram = new Array<number>(VOCAB).fill(0)
          for (const token of first) {
            histogram[token]! += 1
          }
          const chiSquare = effective.reduce((statistic, probability, token) => {
            const expected = samples * probability
            return probability === 0 ? statistic : statistic + (histogram[token]! - expected) ** 2 / expected
          }, 0)
          // A correct sampler sits far below this loose deterministic bound;
          // a misrouted or incorrectly normalized branch fails systematically.
          expect(chiSquare).toBeLessThan(60)
        }
      }))

    it.effect("compiles one replay prefill per chunk shape for parallel-block speculation", () =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        let observed: Runtime.InferenceCompileRequest | undefined
        const recording: Runtime.RuntimeService = {
          ...runtime,
          extensions: {
            ...runtime.extensions,
            inference: {
              ...runtime.extensions.inference,
              compile: (request) => {
                observed = request
                return runtime.extensions.inference.compile(request)
              }
            }
          }
        }
        const { model, params, proposer } = yield* makeParallelFixture
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [2, 4],
          sampling: { temperature: 0, seed: 3 },
          speculation: { proposer, maxDraftTokens: 2 }
        }).pipe(Effect.provideService(Runtime.Runtime, recording))
        expect(observed?.target.prefill).toHaveLength(2)
        expect(observed?.generalizedProposer?.replay?.prefill).toHaveLength(2)
        // Bucketed parallel replay matches ordinary token-for-token.
        const ordinary = yield* (yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [2, 4],
          sampling: { temperature: 0, seed: 3 }
        })).generation()
        const parallel = yield* program.generation()
        let ordinaryPage = (yield* ordinary.add([{ prompt: yield* ids([1, 5, 3, 8, 2, 11]) }]))[0]!
        const parallelFirst = (yield* parallel.add([{ prompt: yield* ids([1, 5, 3, 8, 2, 11]) }]))[0]!
        expect(parallelFirst.tokens).toEqual(ordinaryPage.tokens)
        for (let round = 0; round < 2; round++) {
          const parallelPage = (yield* parallel.step([{ seq: parallelFirst.seq }]))[0]!
          const expected: Array<number> = []
          for (let index = 0; index < parallelPage.tokens.length; index++) {
            ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
            expected.push(...ordinaryPage.tokens)
          }
          expect(parallelPage.tokens).toEqual(expected)
        }
        yield* ordinary.close()
        yield* parallel.close()
      }))

    it.effect("exposures survive chain composition for proposer taps", () =>
      Effect.gen(function*() {
        // The exposure is a node in the graph, so combinator composition
        // (which never saw a trace callback even at inference-trace time)
        // cannot drop it.
        const embedding = yield* Model.embedding("token_embd", VOCAB, EMBED)
        const attention = yield* Model.multiHeadAttention("attn", EMBED, 1, { causal: true, rope: 10_000 })
        const head = yield* Model.linear("output", EMBED, VOCAB)
        const exposingEmbed = yield* Model.define({
          parameterSpecs: embedding.parameterSpecs,
          forward: (params, input) =>
            Effect.flatMap(embedding.forward(params, input), (hidden) => Tensor.expose(hidden, "layers.0.hidden"))
        })
        const chained = yield* Model.chain(exposingEmbed, attention, head)
        const params = yield* Tensor.compute(yield* Model.initialize(chained))
        const { proposer } = yield* makeParallelFixture
        const base = {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 17 }
        } as const
        const ordinary = yield* (yield* Model.inference(chained, params, base)).generation()
        const parallel = yield* (yield* Model.inference(chained, params, {
          ...base,
          speculation: { proposer, maxDraftTokens: 2 }
        })).generation()
        let ordinaryPage = (yield* ordinary.add([{ prompt: yield* ids([1, 2, 3, 4, 5, 6]) }]))[0]!
        const parallelFirst = (yield* parallel.add([{ prompt: yield* ids([1, 2, 3, 4, 5, 6]) }]))[0]!
        expect(parallelFirst.tokens).toEqual(ordinaryPage.tokens)
        const parallelPage = (yield* parallel.step([{ seq: parallelFirst.seq }]))[0]!
        const expected: Array<number> = []
        for (let index = 0; index < parallelPage.tokens.length; index++) {
          ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
          expected.push(...ordinaryPage.tokens)
        }
        expect(parallelPage.tokens).toEqual(expected)
        yield* ordinary.close()
        yield* parallel.close()
      }))

    it.effect("one base model serves multiple speculative proposers", () =>
      Effect.gen(function*() {
        // The model publishes every exposure once; each proposer subscribes
        // to its own name and compiles an independent program set.
        const embedding = yield* Model.embedding("token_embd", VOCAB, EMBED)
        const attention = yield* Model.multiHeadAttention("attn", EMBED, 1, { causal: true, rope: 10_000 })
        const head = yield* Model.linear("output", EMBED, VOCAB)
        const embeddingCount = embedding.parameterSpecs.length
        const attentionCount = attention.parameterSpecs.length
        const model = yield* Model.define({
          parameterSpecs: [...embedding.parameterSpecs, ...attention.parameterSpecs, ...head.parameterSpecs],
          forward: (params, input) =>
            Effect.gen(function*() {
              let hidden = yield* embedding.forward(params.slice(0, embeddingCount), input)
              hidden = yield* Tensor.expose(hidden, Model.hiddenExposure(0))
              hidden = yield* attention.forward(
                params.slice(embeddingCount, embeddingCount + attentionCount),
                hidden
              )
              hidden = yield* Tensor.expose(hidden, Model.hiddenExposure(1))
              return yield* head.forward(params.slice(embeddingCount + attentionCount), hidden)
            })
        })
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const base = {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 1,
          sampling: { temperature: 0, seed: 17 }
        } as const
        const ordinary = yield* (yield* Model.inference(model, params, base)).generation()
        const prompt = [1, 2, 3, 4, 5, 6]
        let ordinaryPage = (yield* ordinary.add([{ prompt: yield* ids(prompt) }]))[0]!
        const expected: Array<number> = [...ordinaryPage.tokens]
        for (let round = 0; round < 6; round++) {
          ordinaryPage = (yield* ordinary.step([{ seq: ordinaryPage.seq }]))[0]!
          expected.push(...ordinaryPage.tokens)
        }
        yield* ordinary.close()
        // Two different speculators on the same base model, interleaved.
        for (const tapName of [Model.hiddenExposure(0), Model.hiddenExposure(1)]) {
          const proposer = parallelProposer(tapName)
          const speculative = yield* (yield* Model.inference(model, params, {
            ...base,
            speculation: { proposer, maxDraftTokens: 2 }
          })).generation()
          let page = (yield* speculative.add([{ prompt: yield* ids(prompt) }]))[0]!
          const actual: Array<number> = [...page.tokens]
          while (actual.length < expected.length) {
            page = (yield* speculative.step([{ seq: page.seq }]))[0]!
            actual.push(...page.tokens)
          }
          expect(actual.slice(0, expected.length)).toEqual(expected)
          yield* speculative.close()
        }
      }))

    it.effect("Muse-Glimmer compiles and generates with parity", () =>
      Effect.gen(function*() {
        const model = yield* MuseGlimmer.create(
          new Map<string, unknown>([
            ["block_count", 1],
            ["embedding_length", EMBED],
            ["feed_forward_length", 16],
            ["context_length", BLOCK],
            ["attention.head_count", HEADS],
            ["attention.head_count_kv", 1],
            ["attention.key_length", 4],
            ["attention.value_length", 4],
            ["attention.layer_norm_rms_epsilon", 1e-4],
            ["attention.sliding_window", BLOCK],
            ["attention.sliding_window_pattern", 2],
            ["rope.freq_base", 10000],
            ["logit_scale", 0.25],
            ["final_logit_softcapping", 10],
            ["vocab_size", VOCAB]
          ])
        )
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 5, 3, 8, 2, 11, 4, 7, 6] // 3 chunks of 4: 4 + 4 + 1(padded)
        const steps = 6
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunks: [4] })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      }))

    it.effect("runs concurrent sessions exactly like sequential ones", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunks: [4] })
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] })
        const gen = yield* program.execution()
        yield* gen.add([yield* ids(Array.from({ length: 16 }, (_, i) => i % VOCAB))]) // all 4 blocks
        expect(yield* gen.live()).toBe(1)
        // A failed admission must roll back its temporary sequence and blocks;
        // the original live sequence remains the sole pool owner.
        const error = yield* Effect.flip(gen.add([yield* ids([1, 2, 3, 4, 5, 6, 7, 8])]))
        expect(error.message).toMatch(/pool exhausted/)
        expect(yield* gen.live()).toBe(1)
      }))

    it.effect("fails a sequence whose context outgrows the pool capacity", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 8, blockSize: 4, prefillChunks: [4] })
        const gen = yield* program.execution()
        const entry = (yield* gen.add([yield* ids([1, 2, 3, 4, 5, 6, 7, 8])]))[0]!
        expect(yield* entry.seq.cursor()).toBe(8)
        const error = yield* Effect.flip(gen.step([{ seq: entry.seq, token: 1 }]))
        expect(error._tag).toBe("TensorError")
        expect(error.message).toMatch(/exceeds pool capacity/)
      }))

    it.effect("returns a finished sequence's blocks to the pool", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] })
        const gen = yield* program.execution()
        const entry = (yield* gen.add([yield* ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])]))[0]! // 3 of 4 blocks
        yield* entry.seq.finish()
        expect(yield* gen.live()).toBe(0)
        // Only possible if the finished blocks came back.
        const full = (yield* gen.add([yield* ids(Array.from({ length: 16 }, (_, i) => i % VOCAB))]))[0]!
        expect(yield* full.seq.cursor()).toBe(16)
      }))

    it.effect("prefix cache: a resident prefix is shared, not recomputed", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        // Five blocks are available. Two independent three-block prompts would
        // need six, so the second prefill fits only by sharing two full blocks.
        const program = yield* Model.inference(model, params, { maxTokens: 20, blockSize: 4, prefillChunks: [4] })
        const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids(prompt)]))[0]!
        const b = (yield* gen.add([yield* ids(prompt)]))[0]!
        deep(yield* Tensor.toNumberArray(b.logits), yield* Tensor.toNumberArray(a.logits))
      }))

    it.effect("prefix cache: divergent suffixes after a shared prefix stay correct", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunks: [4] })
        const shared = [1, 2, 3, 4, 5, 6, 7, 8] // 2 full blocks
        const promptA = [...shared, 9, 10, 11, 0]
        const promptB = [...shared, 3, 4, 5, 6]
        const gen = yield* program.execution()
        yield* gen.add([yield* ids(promptA)])
        const b = (yield* gen.add([yield* ids(promptB)]))[0]!
        // The reference runs an ordinary forward pass over B's whole prompt.
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        // Exactly one 3-block prompt fits; a second, different prompt
        // succeeds only by evicting the first's cached blocks.
        const program = yield* Model.inference(model, params, { maxTokens: 12, blockSize: 4, prefillChunks: [4] })
        {
          const gen = yield* program.execution()
          yield* gen.add([yield* ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])])
          yield* gen.close()
        }
        const prompt = [2, 4, 6, 8, 10, 0, 1, 3, 5, 7, 9, 11]
        const gen = yield* program.execution()
        const entry = (yield* gen.add([yield* ids(prompt)]))[0]!
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 3, 5, 7, 9, 11, 2, 4]
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          attentionWindow: 8
        })
        // Generate past the window. The prompt's first block leaves the window,
        // lands in the prefix cache, and the sequence finishes
        // the rest of the prompt's blocks into the cache as well.
        {
          const gen = yield* program.execution()
          const entry = (yield* gen.add([yield* ids(prompt)]))[0]!
          for (let i = 0; i < 4; i++) {
            yield* gen.step([{ seq: entry.seq, token: 0 }])
          }
          yield* gen.close()
        }
        const gen = yield* program.execution()
        const entry = (yield* gen.add([yield* ids(prompt)]))[0]!
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] })
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids([1, 2, 3])]))[0]!
        const b = (yield* gen.add([yield* ids([1, 2, 3])]))[0]!
        expect(yield* a.seq.cursor()).toBe(3)
        expect(yield* b.seq.cursor()).toBe(3)
        deep(yield* Tensor.toNumberArray(b.logits), yield* Tensor.toNumberArray(a.logits))
      }))

    it.effect("prefix cache: concurrent same-prefix prefills stay exact", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 64, blockSize: 4, prefillChunks: [4] })
        const prompt = [1, 2, 3, 4, 5, 6, 7, 8] // 1 matchable block; +6 steps stays within BLOCK
        // The two prefills can interleave. One may take the other's blocks
        // mid-flight, or both may miss and compute. Greedy
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

    // Reduced-precision pools quantize rows on write and widen on read. Both
    // sides are teacher-forced through identical contexts, avoiding argmax
    // instability; bounds widen from f16 through bf16 to per-row int8 as cache
    // quantization error accumulates in later logits.
    const halfPoolParity = (kvDtype: "f16" | "bf16" | "int8", tol: number) =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          kvDtype
        })
        const prompt = [1, 5, 3, 8, 2]
        const trajectory = [4, 9, 0, 7, 6]
        const gen = yield* program.execution()
        const context = [...prompt]
        const entry = (yield* gen.add([yield* ids(prompt)]))[0]!
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
        // Sliding-window attention is a pool memory policy, not a RoPE-specific
        // operation. With learned absolute positions, the window
        // works within the table and positions stay absolute. Thus (a)
        // below the window generation matches the naive loop exactly,
        // and (b) past it a stepped sequence and a fresh prefill of
        // the same context agree (this would FAIL for window-relative
        // positions, which are the RoPE-only regime).
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          attentionWindow: 8
        })
        // (a) context 3 + 4 steps = 7 < window 8: window never engages.
        const naive = yield* naiveGenerate(model, params, [1, 5, 3], 4)
        const cached = yield* cachedGenerate(program, [1, 5, 3], 4)
        expect(cached).toEqual(naive)
        // (b) prefill 12, step 4 greedy (context 16, first block
        // evicted), then a fresh add of the full 16-token context
        // must produce the same last-position logits.
        const gen = yield* program.execution()
        const prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        const context = [...prompt]
        const a = (yield* gen.add([yield* ids(prompt)]))[0]!
        let logits = a.logits
        for (let i = 0; i < 4; i++) {
          const next = yield* argmaxOf(logits)
          context.push(next)
          const [nextLogits] = yield* gen.step([{ seq: a.seq, token: next }])
          logits = nextLogits
        }
        const fresh = (yield* gen.add([yield* ids(context)]))[0]!
        deep(yield* Tensor.toNumberArray(fresh.logits), yield* Tensor.toNumberArray(logits))
      }))

    it.effect("batched step matches per-sequence step logits exactly", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 4
        })
        const prompts = [
          [1, 2, 3],
          [4, 5, 6, 7, 8],
          [9, 10],
          [1, 2, 3, 0, 11, 5, 6]
        ]
        // The sequential reference uses one session and one step per prompt.
        const reference: Array<Array<number>> = []
        for (const prompt of prompts) {
          const gen = yield* program.execution()
          const entry = (yield* gen.add([yield* ids(prompt)]))[0]!
          const [logits] = yield* gen.step([{ seq: entry.seq, token: 1 }])
          reference.push(yield* Tensor.toNumberArray(logits))
        }
        // The batched run uses one session and one round for all four prompts.
        const gen = yield* program.execution()
        const promptTensors: Array<Tensor.Any> = []
        for (const prompt of prompts) promptTensors.push(yield* ids(prompt))
        const entries = yield* gen.add(promptTensors)
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        // Three blocks fit the reference plus two active sequences exactly;
        // inactive padding rows must not reserve KV capacity.
        const program = yield* Model.inference(model, params, {
          maxTokens: 12,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 8
        })
        // Sequential reference, single-sequence path.
        const ref = yield* program.execution()
        const r1 = (yield* ref.add([yield* ids([3, 1, 4])]))[0]!
        const [expected] = yield* ref.step([{ seq: r1.seq, token: 2 }])
        // Two live sequences in one round: 6 slots pad internally.
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids([3, 1, 4])]))[0]!
        const b = (yield* gen.add([yield* ids([7, 7, 7])]))[0]!
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          attentionWindow: 8,
          batchSize: 2
        })
        const prompt = [1, 3, 5, 7, 9, 11, 2, 4]
        // A generates alone first (evicting its first block past the
        // window); B is added afterwards and shares A's cached blocks.
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids(prompt)]))[0]!
        let logitsA = a.logits
        for (let i = 0; i < 4; i++) {
          const next = yield* argmaxOf(logitsA)
          const [nextLogits] = yield* gen.step([{ seq: a.seq, token: next }])
          logitsA = nextLogits
        }
        const b = (yield* gen.add([yield* ids(prompt)]))[0]!
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        // Pool holds both prompts exactly; the batched step needs one
        // more block per sequence and must fail cleanly.
        const program = yield* Model.inference(model, params, {
          maxTokens: 24,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2
        })
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])]))[0]! // 3 blocks
        const b = (yield* gen.add([yield* ids([2, 4, 6, 8, 10, 0, 1, 3, 5, 7, 9, 11])]))[0]! // 3 blocks
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

    it.effect("add beyond batchSize fails with InferenceError", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2
        })
        const gen = yield* program.execution()
        yield* gen.add([yield* ids([1, 2, 3])])
        yield* gen.add([yield* ids([4, 5, 6])])
        const error = yield* Effect.flip(gen.add([yield* ids([7, 8, 9])]))
        expect(error._tag).toBe("InferenceError")
        expect(error.message).toMatch(/free lanes/)
      }))

    it.effect("finishing a sequence mid-session removes it from the round", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2
        })
        const reference = yield* program.execution()
        const expectedEntry = (yield* reference.add([yield* ids([4, 5, 6])]))[0]!
        const [expected] = yield* reference.step([{ seq: expectedEntry.seq, token: 1 }])
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids([1, 2, 3])]))[0]!
        const b = (yield* gen.add([yield* ids([4, 5, 6])]))[0]!
        yield* a.seq.finish()
        const [out] = yield* gen.step([{ seq: b.seq, token: 1 }])
        deep(yield* Tensor.toNumberArray(out!), yield* Tensor.toNumberArray(expected!))
        const [refill] = yield* gen.add([yield* ids([7, 8, 9, 10])])
        const reversed = yield* gen.step([
          { seq: b.seq, token: 2 },
          { seq: refill!.seq, token: 3 }
        ])
        expect(reversed).toHaveLength(2)
        expect(yield* gen.live()).toBe(2)
        expect(yield* b.seq.cursor()).toBe(5)
        expect(yield* refill!.seq.cursor()).toBe(5)
      }))

    it.effect("f16 pool: prefix cache and sliding window still hold", () =>
      Effect.gen(function*() {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 3, 5, 7, 9, 11, 2, 4]
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          attentionWindow: 8,
          kvDtype: "f16"
        })
        // A resident prefix is shared in the half-precision pool too:
        // two independent 2-block prompts would need 4 of 8 blocks plus
        // B's private suffix block. It fits either way, so assert exact
        // equality of the shared computation instead.
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids(prompt)]))[0]!
        const b = (yield* gen.add([yield* ids(prompt)]))[0]!
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] })
        const gen = yield* program.execution()
        const entry = (yield* gen.add([yield* ids([1, 3, 5])]))[0]!
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const error = yield* Effect.flip(
          Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] })
        )
        expect(error._tag).toBe("InferenceError")
        expect(error.message).toMatch(/only causal attention is cacheable/)
      }))

    it.effect("releases its computed parameter generation when construction fails", () =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const diagnostics = runtime.extensions.diagnostics
        const model = yield* makeGpt({ causal: false })
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        // The baseline includes caller-owned params. Returning to it proves the
        // failed artifact released only its retained generation; readability
        // below proves it did not consume the caller's handles.
        const before = yield* diagnostics.externalMemoryBytes
        yield* Effect.flip(Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] }))
        expect(yield* diagnostics.externalMemoryBytes).toBe(before)
        expect((yield* Tensor.toNumberArray(params[0])).length).toBeGreaterThan(0)
      }))

    it.effect("releases lazily materialized parameters when construction fails", () =>
      Effect.gen(function*() {
        const runtime = yield* Runtime.Runtime
        const diagnostics = runtime.extensions.diagnostics
        const model = yield* makeGpt({ causal: false })
        const params = yield* Model.initialize(model)
        // Lazy params own no storage at this baseline; inference materializes a
        // private generation that must be wholly released on construction error.
        const before = yield* diagnostics.externalMemoryBytes
        yield* Effect.flip(Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] }))
        expect(yield* diagnostics.externalMemoryBytes).toBe(before)
      }))

    it.effect("returns direct logits rows for single and batched decode", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 64,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2
        })
        const gen = yield* program.execution()
        const a = (yield* gen.add([yield* ids([1, 5, 3])]))[0]!
        const b = (yield* gen.add([yield* ids([2, 4])]))[0]!
        expect(a.logits.shape).toEqual([VOCAB])
        expect(b.logits.shape).toEqual([VOCAB])
        const [single] = yield* gen.step([{ seq: a.seq, token: 1 }])
        const batched = yield* gen.step([
          { seq: a.seq, token: 2 },
          { seq: b.seq, token: 3 }
        ])
        expect(single.shape).toEqual([VOCAB])
        expect(batched.map((logits) => logits.shape)).toEqual([[VOCAB], [VOCAB]])
        yield* Tensor.clearAll([a.logits, b.logits, single, ...batched])
        yield* gen.close()
      }))

    it.effect("rejects a model whose output is not batch-by-time-by-vocab logits", () =>
      Effect.gen(function*() {
        const model = yield* Model.define({
          parameterSpecs: [],
          forward: (_, input) => Tensor.cast(input, "f32")
        })
        const error = yield* Effect.flip(
          Model.inference(model, [], { maxTokens: 16, blockSize: 4, prefillChunks: [4] })
        )
        expect(error._tag).toBe("InferenceError")
        expect(error.message).toMatch(/model output must be \[8, 4, vocab\]/)
      }))

    it.effect("validates step entries and keeps finished logits owned by the caller", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          batchSize: 2
        })
        const gen = yield* program.execution()
        const entry = (yield* gen.add([yield* ids([1, 5, 3])]))[0]!
        const before = yield* Tensor.toNumberArray(entry.logits)
        yield* entry.seq.finish()
        yield* entry.seq.finish()
        expect(yield* Tensor.toNumberArray(entry.logits)).toEqual(before)
        expect(yield* gen.live()).toBe(0)
        const finished = yield* Effect.flip(gen.step([{ seq: entry.seq, token: 1 }]))
        expect(finished._tag).toBe("InferenceError")
        expect(finished.message).toMatch(/not a live sequence/)
        const empty = yield* Effect.flip(gen.step([]))
        expect(empty.message).toMatch(/at least one entry/)
        yield* Tensor.clear(entry.logits)
        yield* gen.close()
        yield* gen.close()
      }))

    it.effect("validates inference config and prompt dtype before compilation or execution", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const badConfigs = [
          [{ maxTokens: 16, blockSize: 0, prefillChunks: [4] }, /blockSize/],
          [{ maxTokens: 16, blockSize: 4, attentionWindow: 17, prefillChunks: [4] }, /attentionWindow/],
          [{ maxTokens: 16, blockSize: 4, prefillChunks: [0] }, /prefillChunks/],
          [{ maxTokens: 16, blockSize: 4, prefillChunks: [] }, /prefillChunks/],
          [{ maxTokens: 16, blockSize: 4, prefillChunks: [4, 0] }, /prefillChunks/],
          [{ maxTokens: 16, blockSize: 4, prefillChunks: [2.5] }, /prefillChunks/],
          [{ maxTokens: 16, blockSize: 4, prefillChunks: [4], batchSize: 0 }, /batchSize/],
          [{ maxTokens: 16, blockSize: 4, prefillChunks: [4], sampling: { seed: 1n << 64n } }, /unsigned 64-bit/]
        ] as const
        for (const [config, message] of badConfigs) {
          const error = yield* Effect.flip(Model.inference(model, params, config))
          expect(error._tag).toBe("InferenceError")
          expect(error.message).toMatch(message)
        }
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] })
        const gen = yield* program.execution()
        const wrongDtype = yield* Effect.flip(
          gen.add([yield* Tensor.fromTypedArray(BigInt64Array.of(1n, 2n), [1, 2])])
        )
        expect(wrongDtype._tag).toBe("InferenceError")
        expect(wrongDtype.message).toMatch(/prompt dtype must be u32/)
      }))

    it.effect("runs i64 token programs without truncating token state", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, {
          maxTokens: 32,
          blockSize: 4,
          prefillChunks: [4],
          tokenDtype: "i64"
        })
        const gen = yield* program.execution()
        const entry = (yield* gen.add([yield* Tensor.fromTypedArray(BigInt64Array.of(1n, 5n, 3n), [1, 3])]))[0]!
        const output = yield* model.forward(params, yield* ids([1, 5, 3]))
        const [expected] = yield* Tensor.compute([
          yield* Tensor.reshape(yield* Tensor.slice(output, { start: [0, 2, 0], end: [1, 3, VOCAB] }), [
            VOCAB
          ])
        ])
        deep(yield* Tensor.toNumberArray(entry.logits), yield* Tensor.toNumberArray(expected))
        const [next] = yield* gen.step([{ seq: entry.seq, token: 7 }])
        expect(next.shape).toEqual([VOCAB])
        expect(yield* entry.seq.cursor()).toBe(4)
        yield* Tensor.clearAll([entry.logits, expected, next])
        yield* gen.close()
      }))

    it.effect("validates the add calling convention", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 16, blockSize: 4, prefillChunks: [4] })
        const gen = yield* program.execution()
        const batched = yield* Effect.flip(gen.add([yield* Tensor.fromTypedArray(new Uint32Array(6), [2, 3])]))
        expect(batched._tag).toBe("InferenceError")
        expect(batched.message).toMatch(/expects a prompt of shape \[1, T\]/)
        const badPool = yield* Effect.flip(
          Model.inference(model, params, { maxTokens: 15, blockSize: 4, prefillChunks: [4] })
        )
        expect(badPool._tag).toBe("InferenceError")
        expect(badPool.message).toMatch(/multiple of blockSize/)
      }))

    it.effect("matches the naive logits numerically, not just on argmax", () =>
      Effect.gen(function*() {
        const model = yield* makeGpt()
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4, prefillChunks: [4] })
        const gen = yield* program.execution()
        const entry = (yield* gen.add([yield* ids([1, 5, 3])]))[0]!
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
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 5, 3]
        const steps = 8
        const program = yield* Model.inference(model, params, { maxTokens: 32, blockSize: 4, prefillChunks: [4] })
        const naive = yield* naiveGenerate(model, params, prompt, steps)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
      }))

    it.effect("RoPE + attention window: matches the window-relative recompute token-for-token, unbounded", () =>
      Effect.gen(function*() {
        const model = yield* makeRopeGpt
        const params = yield* Tensor.compute(yield* Model.initialize(model))
        const prompt = [1, 5]
        const window = 8
        const steps = 24 // the context grows to 26: far past the window
        const program = yield* Model.inference(model, params, {
          maxTokens: 16, // 4 blocks: only eviction of dead blocks lets this run at all
          blockSize: 4,
          prefillChunks: [4],
          attentionWindow: window
        })
        const naive = yield* naiveWindowedGenerate(model, params, prompt, steps, window)
        const cached = yield* cachedGenerate(program, prompt, steps)
        expect(cached).toEqual(naive)
        expect(cached.length).toBe(prompt.length + steps)
      }))

    it.effect("RoPE trains because the rotary node differentiates", () =>
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
        yield* trainer.train(yield* Model.initialize(trainer.model))
        expect(losses.length).toBe(50)
        expect(Number.isFinite(losses[49])).toBe(true)
        expect(losses[49]).toBeLessThan(losses[0])
      }))
  })
})
