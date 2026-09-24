# @effect-torch/serve

Effect HttpApi routes for generation and independent decision inference. A model
registration can provide both operations. Serving consumes model-owned token
generation and a `Decision.Scorer`. Core derives decision scorers from
either model family; serving does not select or derive model families.

## Endpoints

| Endpoint                  | Request                   | Response                           |
| ------------------------- | ------------------------- | ---------------------------------- |
| GET /v1/models            | None                      | OpenAI model list                  |
| POST /v1/completions      | One text prompt           | OpenAI completion or SSE           |
| POST /v1/chat/completions | Text messages             | OpenAI chat completion or SSE      |
| POST /v1/decisions        | State and named questions | Answers, distributions and usage   |
| GET /openapi.json         | None                      | Generated HttpApi OpenAPI document |

Generation accepts `model`, `max_tokens`, `seed`, `stream`,
`stream_options.include_usage`, `n: 1` and `user`. Chat also accepts
`max_completion_tokens` instead of `max_tokens`. Autoregressive models accept
`temperature` and `top_p`. DiffusionGemma uses its checkpoint's denoising
policy and rejects these two overrides.

This is a text-only, single-choice OpenAI-compatible API. Message roles are
system, user and assistant. Unsupported fields, including tools, stop strings,
logprobs, penalties, multimodal content and prompt batching, return HTTP 400.
The completions endpoint encodes its prompt literally, without a chat template.
Use the chat endpoint to render the model's conversation template.
Generation defaults to 256 new tokens. Streaming sends committed text deltas,
a finish chunk, optional usage, then `[DONE]`. Diffusion emits committed
blocks, so it does not publish intermediate noisy canvases. A failed stream
emits an error frame and closes without a success terminator.

## Run DiffusionGemma

Build the CUDA and tokenizer addons first. From the workspace root:

```sh
pnpm --filter @effect-torch/examples serve-cuda \
  /path/to/diffusiongemma-26B-A4B-it 8080
```

The CUDA entry point listens on 127.0.0.1 and loads the checkpoint once. It
imports only the CUDA backend. `DiffusionGemma.load` validates the checkpoint,
compiles one artifact, and registers generation and decisions from it.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128,"stream":true}'

curl http://127.0.0.1:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma","state":"The customer was charged twice.","questions":{"department":{"type":"choice","criteria":{"billing":"Charges and refunds","sales":"New purchases"}}},"reads":4,"seed":"example"}'
```

## Compose with Effect

The package compiles existing core definitions into prepared registrations. It does
not interpret checkpoints, select a backend, or own a listening socket.

```ts
import { type Model, Server } from "@effect-torch/serve"
import { NodeHttpServer } from "@effect/platform-node"
import { Layer } from "effect"
import { HttpRouter } from "effect/unstable/http"
import { createServer } from "node:http"

export const serve = (models: ReadonlyArray<Model.Registration>) =>
  HttpRouter.serve(Server.layer({ models, maxConcurrentRequests: 1 })).pipe(
    Layer.provide(NodeHttpServer.layer(createServer, { port: 8080 }))
  )
```

Model integrations expose one scoped loading operation. An application can load
any number of registrations and pass them to the server as plain values:

```ts
import { DiffusionGemma } from "@effect-torch/models"
import { Server } from "@effect-torch/serve"
import { Effect, Layer } from "effect"

const routes = Layer.unwrap(Effect.gen(function*() {
  const a = yield* DiffusionGemma.load({
    directory: "/models/diffusiongemma",
    id: "diffusiongemma"
  })

  return Server.layer({ models: [a] })
}))
```

The runnable CUDA entry point is
`packages/examples/diffusion-gemma/serve-cuda.ts`. Its application layer loads
the checkpoint before opening the listener. On shutdown, server and decision
engine finalizers drain calls before the application releases the artifact and
its parameters.

- `Model.fromGeneration` binds a token generator, tokenizer, and template.
  The application supplies EOS IDs, channel markers, and supported sampling
  overrides. It works with either family without family-specific serving code.
- `AutoRegressive.generate` owns a single-prompt generation session and
  closes it on completion, failure, or interruption. Diffusion generation uses
  the artifact's generation driver with its model-owned denoising policy.
- `Model.fromDecision` binds a `Decision.Scorer`, verified prompt preparation,
  scoped prefix acquisition, and conversion of each independent read into the
  scorer's input. The adapter owns selected tensors through readback.
- `Decision.fromAutoRegressive(program)` scores the next token after each
  prompt at row zero through an independent execution session.
- `Decision.fromDiffusion(program)` scores selected canvas rows against
  an immutable prefix with fresh initial prediction state for every read.
- `Model.load` compiles a core definition once. Its generation and decision
  factories receive the same artifact and return one registration under one ID.
- `Server.toWebHandler` returns a Fetch handler and asynchronous disposal.
  Request aborts and canceled response readers interrupt calls.

The core family adapters provide scoring, not prompt construction. Model loaders
verify answer tokens against their own templates. Causal adapters prepare the
prompt up to the answer and select row zero; diffusion adapters prepare the
canvas and select its verified answer slot. `Model.load` composes both tasks over one compiled artifact.

The server admits at most 16 requests by default, including active calls, and
runs one model call at a time across all registrations. Excess requests receive
429. Streaming producers use bounded queues. Server shutdown interrupts active
and queued model calls and waits for finalizers. Model execution failures use
the OpenAI error envelope without exposing backend diagnostics in the response.

## Decision semantics

The request carries an application-owned model ID, JSON state, and a nonempty
map of questions. State can be a string, object or array. Instructions and
criteria descriptions can contain nested JSON.

- noul returns the probability of true, from the ordered false/true readout.
- choice supports 1 to 255 named options and returns a selected name, its
  confidence and the complete distribution in option enumeration order.
- score accepts 2 to 10 ordered criteria. It returns the expected zero-based
  score, legend, distribution and distance-based confidence.

Choice confidence measures the modal probability above uniform, normalized to
one. Score confidence compares expected distance from the first modal level
with the uniform distribution's distance from the center. These preserve the
existing decision consumer's formulas; confidence is not the selected probability.

Questions are rendered independently. Caller IDs route results and never enter
the model prompt or semantic noise seed. Renaming, reordering or adding sibling
questions does not change an existing question's read. Exact duplicate questions
execute once. One or four independent reads are supported; four-read results
average probabilities, not logits. The default seed is the string "0".

The DiffusionGemma scaffold checks full-prompt tokenization for every answer code
and corrupts only its verified answer slot. The engine shares immutable encoded
prefixes with exact token equality, keeping at most two cached prefixes for
60 seconds idle. Each decision prompt is limited to 4096 tokens. The loader's
default shared state pool is 16384 tokens. Applications binding their own
artifact must budget for both cached prefixes and generation state.

Decision usage counts input tokens and evaluated canvas rows per distinct
question, including cache hits. Generation usage counts prompt IDs and consumed
output IDs, including channel controls and EOS, excluding post-EOS padding.

## Validation

```sh
pnpm --filter @effect-torch/serve test
pnpm --filter @effect-torch/serve typecheck
pnpm --filter @effect-torch/models exec vitest run test/DiffusionGemmaDecision.test.ts
```

Unit tests cover request validation, question independence, prefix ownership,
failure cleanup, cancellation, HTTP/SSE behavior, and real CPU execution for
autoregressive and diffusion artifacts. The models package DiffusionGemma tests use
`EFFECT_TORCH_DIFFUSION_GEMMA_TOKENIZER_ASSETS` when set to the checkpoint
directory. These checks do not measure serving capacity or decision calibration.
