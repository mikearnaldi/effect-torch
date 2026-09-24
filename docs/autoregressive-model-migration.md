# Autoregressive model API migration

RFC 0027 moves autoregressive inference from `Model` to
`AutoRegressive`. Import the new namespace from `@effect-torch/core`.
The old names have no compatibility aliases.

## Symbol map

| Previous symbol | Current symbol |
| --- | --- |
| `Model.inference` | `AutoRegressive.compile` |
| `Model.InferenceError` | `AutoRegressive.InferenceError` |
| `Model.CompileOptions` | `AutoRegressive.CompileOptions` |
| `Model.Artifact` | `AutoRegressive.Artifact` |
| `Model.Generation` | `AutoRegressive.Generation` |
| `Model.GenerationSeq` | `AutoRegressive.GenerationSeq` |
| `Model.GenerationSamplingOptions` | `AutoRegressive.GenerationSamplingOptions` |
| `Model.GenerationAdd` | `AutoRegressive.GenerationAdd` |
| `Model.GenerationStep` | `AutoRegressive.GenerationStep` |
| `Model.TokenPage` | `AutoRegressive.TokenPage` |
| `Model.StatefulExecution` | `AutoRegressive.StatefulExecution` |
| `Model.StatefulExecutionSeq` | `AutoRegressive.StatefulExecutionSeq` |

For example:

```ts
import { AutoRegressive, type Model } from "@effect-torch/core"

const prepare = (model: Model.Definition, params: Model.Parameters) =>
  AutoRegressive.compile(model, params, {
    maxTokens: 256,
    blockSize: 16,
    prefillChunks: [16, 64],
    batchSize: 8,
    sampling: { seed: 0, temperature: 0 }
  })
```

`Model` still supplies model definitions, parameter specifications,
initialization, composition, ordinary execution, `ModelError`, and
`hiddenExposure`. `Model.executeLayers` remains available for layer-by-layer
reference evaluation. The interim `Model.PrefixModel` and `Model.executor`
are removed; compiled diffusion consumers use `Diffusion.compile`.

`Speculation` continues to describe proposers. DFlash remains a parallel-block
proposer configured on `AutoRegressive.compile`. Chat accepts an
`AutoRegressive.Artifact`. Session methods, sampled nonempty
token pages, error tags, sampling defaults, and caller-owned logits retain
their existing contracts.

## Chat generation consumers

`Chat.stream(options)` retains its autoregressive program and sampling options,
including custom host logits samplers. `Chat.streamWith(generate, options)`
accepts a required generation operation from either family. Both use one
template, token parser, and event driver. Shared options are `Chat.ChatOptions`;
`Chat.ChatStreamOptions` adds the autoregressive program and sampling controls.

The `Chat.ChatGeneration<E, R>` callback receives `prompt: Uint32Array`,
`maxTokens: number | undefined`, `eosTokens: ReadonlyArray<number>`, and
`onPage(page: Chat.ChatTokenPage): Effect<void>`. It publishes sequential
nonempty pages and returns `Effect<"stop" | "maxTokens", E, R>`. Each page has
`tokens: ReadonlyArray<number> | Uint32Array` and may carry a terminal
`stopReason: "eos" | "maxTokens"`.

Await `onPage` before continuing. Tokens are borrowed until it completes.
Chat can finish or be cancelled within a page; it interrupts the generator
and waits for its scoped cleanup. No empty refinement pages or synthetic
terminal events appear after failure or cancellation. Generation errors and
Effect requirements remain in the returned stream's type.

## Shared parameter preparation

`Model.withParameters` prepares one parameter generation for compilation:

```ts
Model.withParameters<A, E, R>(
  sourceParams: ReadonlyArray<Tensor.Any>,
  use: (params: ReadonlyArray<Tensor.Concrete>) => Effect.Effect<A, E, R>
): Effect.Effect<
  A,
  E | Model.ModelError | Tensor.TensorError,
  R | Runtime.Runtime
>
```

It materializes distinct dense source tensors together once, deduplicating by
tensor identity, then restores the original array order. Packed concrete
parameters are borrowed. The callback can compile multiple entry points against
the same parameter generation. Native programs must retain the constants before
the callback returns.

The helper records acquired handles in the acquisition's success exit handler
and clears all dense temporaries on success, failure, or interruption. Acquisition
inherits the caller's interruptibility; native adapters own partial and late
results. Caller-supplied handles remain live. A packed parameter that is not
concrete fails with `Model.ModelError`, with `op: "withParameters"`.

`Model.sameStateGeometry(left, right)` compares the persistent K/V, KDA, and
convolution layouts of two `Runtime.DecodeStateSchema` values. Ordered K/V
layer identities, shapes, dtypes, and retention windows must agree. Entry points
can have different query widths, output selections, or append/read-only access.

`Model.makeStatePool(schema)` allocates a `Tensor.KvPool` from the compiled
schema, including its token capacity, block size, and recurrent geometry. It
returns `Effect<Tensor.KvPool, Tensor.TensorError, Runtime.Runtime>`. Both
families can use these public operations directly. Autoregressive programs
request explicit `Append` access.
