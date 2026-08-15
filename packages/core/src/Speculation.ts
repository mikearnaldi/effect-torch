/**
 * High-level speculative decoding artifacts.
 *
 * TypeScript describes the proposer and builds its Tensor graphs. Model
 * inference traces those graphs and lowers the selected variant to Runtime
 * wire plans; native runtimes own execution, state, sampling, and acceptance.
 */
import type { Effect } from "effect"
import type { Model, ModelError, Params } from "./Model.ts"
import type * as Runtime from "./Runtime.ts"
import type * as Tensor from "./Tensor.ts"

/** One layer of authoritative proposer K/V rows rebuilt after acceptance. */
export interface KeyValue {
  readonly key: Tensor.Lazy
  readonly value: Tensor.Lazy
}

/** A target residual activation consumed by a replayable parallel block. */
export interface HiddenTap {
  readonly layer: number
  readonly dtype: Runtime.DType
  readonly shape: ReadonlyArray<number | "Rows">
}

/** A target weight shared with a replayable parallel block. */
export interface SharedWeight {
  readonly name: string
  readonly dtype: Runtime.DType
  readonly shape: ReadonlyArray<number>
}

/** Exact autoregressive draft model with the same token vocabulary as the target. */
export interface Autoregressive {
  readonly _tag: "Autoregressive"
  readonly model: Model
  readonly params: Params
  readonly vocabulary: number
  readonly maxDraftTokens: number
}

/** Deterministic suffix n-gram lookup over each sequence's committed history. */
export interface HistoryLookup {
  readonly _tag: "HistoryLookup"
  readonly vocabulary: number
  readonly maxDraftTokens: number
  readonly minMatchTokens: number
  readonly maxMatchTokens: number
}

export interface ParallelBlockOutput {
  readonly tokenIds: Tensor.Lazy
  readonly probabilityRows?: Tensor.Lazy
}

/** One replayable fixed-width parallel proposal graph, as used by DFlash. */
export interface ParallelBlock {
  readonly _tag: "ParallelBlock"
  readonly params: Params
  readonly vocabulary: number
  readonly maxDraftTokens: number
  readonly hiddenTaps: ReadonlyArray<HiddenTap>
  readonly tokenEmbedding: SharedWeight
  readonly lmHead: SharedWeight
  readonly currentBlockAttention?: "Causal" | "Bidirectional"
  readonly attentionWindow?: number
  readonly build: (
    params: Params,
    anchorTokens: Tensor.Any,
    tokenEmbedding: Tensor.Any,
    lmHead: Tensor.Any,
    maxDraftTokens: number
  ) => Effect.Effect<Tensor.Lazy, ModelError | Tensor.TensorError, Runtime.Runtime>
  readonly buildWithProbabilities?: (
    params: Params,
    anchorTokens: Tensor.Any,
    tokenEmbedding: Tensor.Any,
    lmHead: Tensor.Any,
    maxDraftTokens: number
  ) => Effect.Effect<ParallelBlockOutput, ModelError | Tensor.TensorError, Runtime.Runtime>
  readonly replay: (
    params: Params,
    targetRows: ReadonlyArray<Tensor.Any>
  ) => Effect.Effect<ReadonlyArray<KeyValue>, ModelError | Tensor.TensorError, Runtime.Runtime>
}

/** The complete public speculation language. */
export type Artifact = Autoregressive | HistoryLookup | ParallelBlock

/** Constructs an exact autoregressive proposer. */
export const autoregressive = (
  model: Model,
  params: Params,
  options: { readonly vocabulary: number; readonly maxDraftTokens: number }
): Autoregressive => ({ _tag: "Autoregressive", model, params, ...options })

/** Constructs a deterministic suffix n-gram history proposer. */
export const historyLookup = (options: Omit<HistoryLookup, "_tag">): HistoryLookup => ({
  _tag: "HistoryLookup",
  ...options
})

/** Constructs one replayable fixed-width parallel-block proposer. */
export const parallelBlock = (options: Omit<ParallelBlock, "_tag">): ParallelBlock => ({
  _tag: "ParallelBlock",
  ...options
})
