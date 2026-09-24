import { Diffusion } from "@effect-torch/core"
import * as Gemma from "@effect-torch/models/DiffusionGemma"
import { describe, expect, it } from "@effect/vitest"
import { Deferred, Effect, Exit, Fiber } from "effect"
import { readFileSync } from "node:fs"

interface TensorRecord {
  readonly values: ReadonlyArray<number>
  readonly dtype: string
}

interface ReferenceStep {
  readonly remaining: number
  readonly canvas: ReadonlyArray<number>
  readonly positions: ReadonlyArray<number>
  readonly feedback_in: TensorRecord | null
  readonly feedback_out: TensorRecord
  readonly raw_logits: TensorRecord
  readonly processed_logits: TensorRecord
  readonly exponentials: ReadonlyArray<number>
  readonly probabilities: ReadonlyArray<number>
  readonly sampled_tokens: ReadonlyArray<number>
  readonly argmax_tokens: ReadonlyArray<number>
  readonly token_entropy: ReadonlyArray<number>
  readonly entropy_order: ReadonlyArray<number>
  readonly mean_entropy: number
  readonly accepted_mask: ReadonlyArray<boolean>
  readonly accepted_canvas: ReadonlyArray<number>
  readonly random_canvas: ReadonlyArray<number>
  readonly next_canvas: ReadonlyArray<number>
  readonly done: boolean
}

interface ReferenceRun {
  readonly dtype: string
  readonly prompt: ReadonlyArray<number>
  readonly canvas_length: number
  readonly vocab_size: number
  readonly sequences: ReadonlyArray<number>
  readonly config: {
    readonly max_new_tokens: number
    readonly max_denoising_steps: number
    readonly sampler_config: { readonly entropy_bound: number }
    readonly t_min: number
    readonly t_max: number
    readonly stability_threshold: number
    readonly confidence_threshold: number
  }
  readonly encoder_calls: ReadonlyArray<
    {
      readonly tokens: ReadonlyArray<number>
      readonly positions: ReadonlyArray<number>
    }
  >
  readonly blocks: ReadonlyArray<{
    readonly initial_canvas: ReadonlyArray<number>
    readonly steps: ReadonlyArray<ReferenceStep>
  }>
}

const fixture: {
  readonly runs: ReadonlyArray<ReferenceRun>
  readonly stopping: ReadonlyArray<{
    readonly argmax_tokens: ReadonlyArray<number>
    readonly mean_entropy: number
    readonly done: boolean
  }>
} = JSON.parse(readFileSync(new URL("./fixtures/diffusion-gemma-generation.json", import.meta.url), "utf8"))

const prediction = (step: ReferenceStep): Gemma.GenerationPrediction => ({
  sampledTokens: Uint32Array.from(step.sampled_tokens),
  argmaxTokens: Uint32Array.from(step.argmax_tokens),
  tokenEntropy: Float32Array.from(step.token_entropy),
  entropyOrder: Uint32Array.from(step.entropy_order),
  meanEntropy: step.mean_entropy
})

const config = (run: ReferenceRun): Gemma.GenerationSamplerConfig => ({
  canvasLength: run.canvas_length,
  maxSteps: run.config.max_denoising_steps,
  entropyBound: run.config.sampler_config.entropy_bound,
  stabilityThreshold: run.config.stability_threshold,
  confidenceThreshold: run.config.confidence_threshold,
  eosTokenIds: [],
  padTokenId: 0
})

describe("pinned DiffusionGemma sampler", () => {
  for (const run of fixture.runs) {
    it(`${run.dtype} replays recorded random inputs through every refinement`, () => {
      for (const block of run.blocks) {
        let state: Gemma.GenerationSamplerState = { history: [] }

        for (const step of block.steps) {
          // PyTorch num_samples=1 uses an exponential race. Replay actual draws,
          // rather than expecting JavaScript and Python seeds to share an RNG.
          const sampled = new Uint32Array(run.canvas_length)

          for (let row = 0; row < sampled.length; row++) {
            let best = -Infinity

            for (let token = 0; token < run.vocab_size; token++) {
              const offset = row * run.vocab_size + token
              const score = Math.fround(step.probabilities[offset] / step.exponentials[offset])

              if (score > best) {
                best = score
                sampled[row] = token
              }
            }
          }

          expect(Array.from(sampled)).toEqual(step.sampled_tokens)
          const stats = { ...prediction(step), sampledTokens: sampled }

          const result = Gemma.sampleGenerationCanvas(
            Uint32Array.from(step.canvas),
            stats,
            Uint32Array.from(step.random_canvas),
            config(run).entropyBound
          )

          expect(Array.from(result.acceptedMask, Boolean)).toEqual(step.accepted_mask)
          expect(Array.from(result.acceptedCanvas)).toEqual(step.accepted_canvas)
          expect(Array.from(result.canvas)).toEqual(step.next_canvas)

          const stopped = Gemma.stopGeneration(
            state,
            stats,
            config(run).stabilityThreshold,
            config(run).confidenceThreshold
          )

          expect(stopped.done).toBe(step.done)
          state = stopped.state

          const temperature = Gemma.generationTemperature(
            run.config.t_min,
            run.config.t_max,
            config(run).maxSteps,
            step.remaining
          )

          expect(step.raw_logits.values.map((value) => Math.fround(value / temperature)))
            .toEqual(step.processed_logits.values)
        }
      }
    })
  }

  it("acceptance is recomputed rather than accumulated, using cumulative entropy minus its maximum", () => {
    const stats: Gemma.GenerationPrediction = {
      sampledTokens: new Uint32Array([4, 5, 6]),
      argmaxTokens: new Uint32Array([1, 1, 1]),
      tokenEntropy: new Float32Array([0.01, 0.2, 0.8]),
      entropyOrder: new Uint32Array([0, 1, 2]),
      meanEntropy: 0.337
    }

    const first = Gemma.sampleGenerationCanvas(new Uint32Array([1, 2, 3]), stats, new Uint32Array([9, 9, 9]), 0.1)
    expect(Array.from(first.acceptedMask)).toEqual([1, 1, 0])
    expect(Array.from(first.canvas)).toEqual([4, 5, 9])

    const second = Gemma.sampleGenerationCanvas(
      first.canvas,
      {
        ...stats,
        tokenEntropy: new Float32Array([0.8, 0.2, 0.01]),
        entropyOrder: new Uint32Array([2, 1, 0])
      },
      new Uint32Array([8, 8, 8]),
      0.1
    )

    expect(Array.from(second.acceptedMask)).toEqual([0, 1, 1])
    expect(Array.from(second.canvas)).toEqual([8, 5, 6])
  })

  it("requires the complete argmax history and a strict confidence inequality", () => {
    let state: Gemma.GenerationSamplerState = { history: [] }
    const base = prediction(fixture.runs[0].blocks[0].steps[0])
    const observed: Array<boolean> = []

    for (const step of fixture.stopping) {
      const result = Gemma.stopGeneration(
        state,
        {
          ...base,
          argmaxTokens: Uint32Array.from(step.argmax_tokens),
          meanEntropy: step.mean_entropy
        },
        2,
        0.005
      )

      expect(result.done).toBe(step.done)
      observed.push(result.done)
      state = result.state
    }

    expect(observed).toEqual([false, false, false, false, true])
    expect(Gemma.stopGeneration({ history: [] }, { ...base, meanEntropy: 0.125 }, 0, 0.125).done).toBe(false)
    expect(Gemma.stopGeneration({ history: [] }, { ...base, meanEntropy: 0.124 }, 0, 0.125).done).toBe(true)
  })

  it("keeps the first EOS and follows the reference padding rule", () => {
    const draft = new Uint32Array([4, 9, 3, 8])
    expect(Gemma.finishGenerationCanvas(draft, [8, 9], 0)).toEqual({
      tokens: new Uint32Array([4, 9, 0, 0]),
      stop: true
    })
    expect(Gemma.finishGenerationCanvas(draft, [8, 9], undefined)).toEqual({ tokens: draft, stop: true })
    expect(Gemma.finishGenerationCanvas(draft, [], 0)).toEqual({ tokens: draft, stop: false })
    expect(Array.from(draft)).toEqual([4, 9, 3, 8])
  })
})

interface Resource<A> {
  readonly value: A
  live: boolean
}

const replay = (run: ReferenceRun) => {
  const resources: Array<Resource<ReadonlyArray<number> | TensorRecord | null>> = []
  const released: Array<Resource<ReadonlyArray<number> | TensorRecord | null>> = []

  const own = <A extends ReadonlyArray<number> | TensorRecord | null>(value: A): Diffusion.Owned<Resource<A>> => {
    const resource: Resource<A> = { value, live: true }
    resources.push(resource)

    return {
      value: resource,
      release: Effect.sync(() => {
        expect(resource.live).toBe(true)
        resource.live = false
        released.push(resource)
      })
    }
  }

  let current = run.blocks[0].steps[0]
  let commits = 0
  const pages: Array<Array<number>> = []
  const progress: Array<boolean> = []

  const callbacks: Diffusion.Callbacks<
    Resource<ReadonlyArray<number>>,
    Resource<TensorRecord>,
    Gemma.GenerationPrediction,
    string,
    never
  > = {
    encode: (tokens) => Effect.sync(() => own(Array.from(tokens))),
    initialize: (block) =>
      Effect.sync(() => ({
        value: {
          canvas: Uint32Array.from(run.blocks[block.index].initial_canvas),
          feedback: { _tag: "Initial" as const }
        },
        release: own(null).release
      })),
    evaluate: ({ prefix, canvas, feedback, block, step }) =>
      Effect.sync(() => {
        current = run.blocks[block.index].steps[step.index]
        expect(prefix.live).toBe(true)
        expect(resources.filter((resource) => resource.live)).toHaveLength(2)
        expect(prefix.value).toEqual(run.sequences.slice(0, block.position))
        expect(Array.from(canvas)).toEqual(current.canvas)
        expect(current.positions).toEqual(
          Array.from({ length: run.canvas_length }, (_, index) => block.position + index)
        )
        expect(step.remaining).toBe(current.remaining)

        if (feedback._tag === "Initial") {
          expect(current.feedback_in).toBeNull()
        } else {
          expect(feedback.value.live).toBe(true)
          expect(feedback.value.value).toEqual(current.feedback_in)
        }

        const output = own(current.feedback_out)

        return { value: { prediction: prediction(current), feedback: output.value }, release: output.release }
      }),
    commit: (prefix, tokens, block) =>
      Effect.sync(() => {
        expect(prefix.live).toBe(true)
        commits++
        expect(resources.filter((resource) => resource.live)).toHaveLength(1)
        expect(Array.from(tokens)).toEqual(run.encoder_calls[commits].tokens)
        expect(run.encoder_calls[commits].positions[0]).toBe(block.position)

        return own([...prefix.value, ...tokens])
      })
  }

  const options: Diffusion.GenerationOptions<
    Resource<ReadonlyArray<number>>,
    Resource<TensorRecord>,
    Gemma.GenerationPrediction,
    Gemma.GenerationSamplerState,
    string,
    never
  > = {
    prompt: Uint32Array.from(run.prompt),
    maxNewTokens: run.config.max_new_tokens,
    outputLimit: "whole-block",
    callbacks,
    policy: Gemma.generationPolicy(config(run), () => Effect.sync(() => Uint32Array.from(current.random_canvas))),
    onProgress: ({ done }) =>
      Effect.sync(() => {
        progress.push(done)
        expect(pages.length).toBe(Math.floor((progress.length - 1) / config(run).maxSteps))
      }),
    onPage: (tokens) =>
      Effect.sync(() => {
        pages.push(Array.from(tokens))
        tokens.fill(99) // A consumer cannot corrupt the next encoder commit.
      })
  }

  return {
    options,
    resources,
    released,
    pages,
    progress,
    commitCount: () => commits
  }
}

describe("diffusion generation scheduler", () => {
  for (const run of fixture.runs) {
    for (const outputLimit of ["whole-block", "exact"] as const) {
      it.effect(`${run.dtype} carries feedback, commits argmax blocks and handles ${outputLimit} output`, () =>
        Effect.gen(function*() {
          const test = replay(run)
          const result = yield* Diffusion.runGeneration({ ...test.options, outputLimit })
          const count = outputLimit === "exact" ? run.config.max_new_tokens : run.canvas_length * run.blocks.length
          expect(test.pages.flat()).toEqual(run.sequences.slice(run.prompt.length, run.prompt.length + count))
          expect(test.pages.map((page) => page.length)).toEqual([3, outputLimit === "exact" ? 2 : 3])
          expect(test.progress).toEqual([false, false, true, false, false, true])
          expect(result).toEqual({ generatedTokens: count, blocks: 2, refinements: 6, stop: "length" })
          expect(test.commitCount()).toBe(1)
          expect(test.resources.every((resource) => !resource.live)).toBe(true)
          expect(test.released).toHaveLength(test.resources.length)
        }))
    }
  }

  it.effect("does no work for a zero-token request", () =>
    Effect.gen(function*() {
      const test = replay(fixture.runs[0])
      expect(yield* Diffusion.runGeneration({ ...test.options, maxNewTokens: 0 })).toEqual({
        generatedTokens: 0,
        blocks: 0,
        refinements: 0,
        stop: "length"
      })
      expect(test.resources).toEqual([])
      expect(test.pages).toEqual([])
    }))

  it.effect("converges early, still draws renoising inputs, and stops on EOS without a final encoder commit", () =>
    Effect.gen(function*() {
      const run = fixture.runs[0]
      const test = replay(run)
      const first = run.blocks[0].steps[0]
      let draws = 0

      const policy = Gemma.generationPolicy({
        ...config(run),
        stabilityThreshold: 0,
        confidenceThreshold: 100,
        eosTokenIds: [first.argmax_tokens[1]]
      }, () =>
        Effect.sync(() => {
          draws++

          return Uint32Array.from(first.random_canvas)
        }))

      const result = yield* Diffusion.runGeneration({ ...test.options, policy, maxNewTokens: 12 })
      expect(result).toEqual({ generatedTokens: 3, blocks: 1, refinements: 1, stop: "policy" })
      expect(test.pages).toEqual([
        Array.from(
          Gemma.finishGenerationCanvas(Uint32Array.from(first.argmax_tokens), [first.argmax_tokens[1]], 0).tokens
        )
      ])
      expect(draws).toBe(1)
      expect(test.progress).toEqual([true])
      expect(test.commitCount()).toBe(0)
      expect(test.resources.every((resource) => !resource.live)).toBe(true)
    }))

  for (const stage of ["encode", "evaluate", "commit"] as const) {
    for (const outcome of ["failure", "interrupt"] as const) {
      it.effect(`${outcome} during ${stage} releases request-owned state and permits another session`, () =>
        Effect.gen(function*() {
          const test = replay(fixture.runs[0])
          const reached = yield* Deferred.make<void>()
          let partialLive = false

          // An acquisition owns cleanup of its own partial work. The driver owns
          // only successful results, including the previous refinement feedback.
          const pending = Effect.gen(function*() {
            partialLive = true
            yield* Deferred.succeed(reached, undefined)

            return yield* outcome === "failure" ? Effect.fail("injected") : Effect.never
          }).pipe(Effect.ensuring(Effect.sync(() => {
            partialLive = false
          })))

          const callbacks = { ...test.options.callbacks }

          if (stage === "encode") callbacks.encode = () => pending

          if (stage === "evaluate") {
            const evaluate = callbacks.evaluate
            callbacks.evaluate = (input) => input.step.index === 1 ? pending : evaluate(input)
          }

          if (stage === "commit") callbacks.commit = () => pending

          const execution = Diffusion.runGeneration({ ...test.options, callbacks })

          if (outcome === "failure") {
            expect(yield* Effect.flip(execution)).toBe("injected")
          } else {
            const fiber = yield* Effect.forkChild(execution)
            yield* Deferred.await(reached)
            yield* Fiber.interrupt(fiber)
            expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
          }

          expect(partialLive).toBe(false)
          expect(test.resources.every((resource) => !resource.live)).toBe(true)
          expect(test.released).toHaveLength(test.resources.length)
          const other = replay(fixture.runs[0])
          expect((yield* Diffusion.runGeneration(other.options)).generatedTokens).toBe(6)
          expect(other.resources.every((resource) => !resource.live)).toBe(true)
        }))
    }
  }

  for (const boundary of ["previous feedback", "final feedback", "previous prefix"] as const) {
    it.effect(`interruption during ${boundary} release finishes cleanup without publishing another page`, () =>
      Effect.gen(function*() {
        const run = fixture.runs[0]
        const test = replay(run)
        const releasing = yield* Deferred.make<void>()
        const resume = yield* Deferred.make<void>()
        let releaseFinished = false

        const pauseRelease = <A>(owned: Diffusion.Owned<A>): Diffusion.Owned<A> => ({
          value: owned.value,
          release: Effect.gen(function*() {
            yield* Deferred.succeed(releasing, undefined)
            yield* Deferred.await(resume)
            yield* owned.release
            releaseFinished = true
          })
        })

        const callbacks = { ...test.options.callbacks }

        if (boundary === "previous prefix") {
          const encode = callbacks.encode
          callbacks.encode = (tokens) => Effect.map(encode(tokens), pauseRelease)
        } else {
          const evaluate = callbacks.evaluate
          const releasedStep = boundary === "previous feedback" ? 0 : config(run).maxSteps - 1
          callbacks.evaluate = (input) =>
            Effect.map(
              evaluate(input),
              (owned) => input.block.index === 0 && input.step.index === releasedStep ? pauseRelease(owned) : owned
            )
        }

        const fiber = yield* Effect.forkChild(Diffusion.runGeneration({ ...test.options, callbacks }))
        yield* Effect.raceFirst(Deferred.await(releasing), Fiber.join(fiber))
        const pagesAtRelease = test.pages.map((page) => page.slice())
        const progressAtRelease = test.progress.slice()
        const interruption = yield* Fiber.interrupt(fiber).pipe(Effect.forkChild({ startImmediately: true }))
        yield* Deferred.succeed(resume, undefined)
        yield* Fiber.join(interruption)
        expect(Exit.isFailure(yield* Fiber.await(fiber))).toBe(true)
        expect(releaseFinished).toBe(true)
        expect(test.resources.every((resource) => !resource.live)).toBe(true)
        expect(test.released).toHaveLength(test.resources.length)
        expect(test.pages).toEqual(pagesAtRelease)
        expect(test.progress).toEqual(progressAtRelease)
        expect(test.pages).toHaveLength(boundary === "previous prefix" ? 1 : 0)
        expect(test.commitCount()).toBe(boundary === "previous prefix" ? 1 : 0)
      }))
  }

  it.effect("cleans both previous and newly acquired feedback if the policy fails", () =>
    Effect.gen(function*() {
      const test = replay(fixture.runs[0])
      const policy = { ...test.options.policy, refine: () => Effect.fail("sampler failed") }
      expect(yield* Effect.flip(Diffusion.runGeneration({ ...test.options, policy }))).toBe("sampler failed")
      expect(test.resources).toHaveLength(3)
      expect(test.resources.every((resource) => !resource.live)).toBe(true)
      expect(test.pages).toEqual([])
    }))

  it.effect("releases all resources even when a release defects", () =>
    Effect.gen(function*() {
      const test = replay(fixture.runs[0])
      const evaluate = test.options.callbacks.evaluate

      const execution = Diffusion.runGeneration({
        ...test.options,
        callbacks: {
          ...test.options.callbacks,
          evaluate: (input) =>
            Effect.map(evaluate(input), (result) => ({
              value: result.value,
              release: result.release.pipe(Effect.andThen(Effect.die("release failed")))
            }))
        },
        policy: { ...test.options.policy, refine: () => Effect.fail("sampler failed") }
      })

      expect(Exit.isFailure(yield* Effect.exit(execution))).toBe(true)
      expect(test.resources.every((resource) => !resource.live)).toBe(true)
      expect(test.released).toHaveLength(test.resources.length)
    }))
})
