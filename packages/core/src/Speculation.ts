/**
 * Defines autoregressive, history lookup, and parallel block proposers for
 * speculative decoding.
 *
 * TypeScript describes the proposer and builds its Tensor graphs. Model
 * inference traces those graphs and lowers the selected variant to Runtime
 * wire plans; native runtimes own execution, state, sampling, and acceptance.
 *
 * @since 0.1.0
 */
import type { Effect } from "effect"
import type { Model, ModelError, Params } from "./Model.ts"
import type * as Runtime from "./Runtime.ts"
import type * as Tensor from "./Tensor.ts"

/**
 * One layer of authoritative proposer key/value rows rebuilt after acceptance.
 *
 * @since 0.1.0
 * @category models
 */
export interface KeyValue {
  /** Key rows in the layer's attention layout. */
  readonly key: Tensor.Lazy
  /** Value rows in the layer's attention layout. */
  readonly value: Tensor.Lazy
}

/**
 * A target residual activation consumed by a replayable parallel block.
 *
 * @since 0.1.0
 * @category models
 */
export interface HiddenTap {
  /**
   * Name of the target-model exposure routed to the proposer, matching a
   * `Tensor.expose` call in the target's `forward` (see
   * {@link Model.hiddenExposure} for the per-layer residual contract).
   */
  readonly name: string
  /** Required residual element dtype. */
  readonly dtype: Runtime.DType
  /** Logical shape, with `"Rows"` denoting the runtime-selected row extent. */
  readonly shape: ReadonlyArray<number | "Rows">
}

/**
 * A target weight shared with a replayable parallel block.
 *
 * @since 0.1.0
 * @category models
 */
export interface SharedWeight {
  /** Exact target parameter name. */
  readonly name: string
  /** Required logical dtype. */
  readonly dtype: Runtime.DType
  /** Required logical shape. */
  readonly shape: ReadonlyArray<number>
}

/**
 * Exact autoregressive draft model with the same token vocabulary as the target.
 *
 * @since 0.1.0
 * @category models
 */
export interface Autoregressive {
  /** Discriminates an autoregressive proposer. */
  readonly _tag: "Autoregressive"
  /** Draft model traced into prefill and decode programs. */
  readonly model: Model
  /** Parameters supplied to the draft model. */
  readonly params: Params
  /** Shared target and proposer vocabulary size. */
  readonly vocabulary: number
  /** Maximum candidate tokens proposed per speculative round. */
  readonly maxDraftTokens: number
}

/**
 * Deterministic suffix n-gram lookup over each sequence's committed history.
 *
 * @since 0.1.0
 * @category models
 */
export interface HistoryLookup {
  /** Discriminates a history-lookup proposer. */
  readonly _tag: "HistoryLookup"
  /** Target vocabulary size used to validate proposed token ids. */
  readonly vocabulary: number
  /** Maximum candidate tokens returned by one lookup. */
  readonly maxDraftTokens: number
  /** Minimum suffix length eligible for a match. */
  readonly minMatchTokens: number
  /** Maximum suffix length searched for a match. */
  readonly maxMatchTokens: number
}

/**
 * Token candidates and optional normalized distributions produced by a
 * parallel-block graph.
 *
 * @since 0.1.0
 * @category models
 */
export interface ParallelBlockOutput {
  /** Candidate token ids for each draft row. */
  readonly tokenIds: Tensor.Lazy
  /** Optional target-vocabulary probability row for each candidate. */
  readonly probabilityRows?: Tensor.Lazy
}

/**
 * One replayable fixed-width parallel proposal graph, as used by DFlash.
 *
 * @since 0.1.0
 * @category models
 */
export interface ParallelBlock {
  /** Discriminates a replayable parallel-block proposer. */
  readonly _tag: "ParallelBlock"
  /** Proposer-owned parameters consumed by graph builders. */
  readonly params: Params
  /** Target vocabulary size of emitted token ids and probability rows. */
  readonly vocabulary: number
  /** Maximum candidate tokens built by one parallel block. */
  readonly maxDraftTokens: number
  /** Ordered target residual activations required by {@link replay}. */
  readonly hiddenTaps: ReadonlyArray<HiddenTap>
  /** Target token-embedding parameter shared with the proposal graph. */
  readonly tokenEmbedding: SharedWeight
  /** Target language-model head shared with the proposal graph. */
  readonly lmHead: SharedWeight
  /** Visibility policy for rows inside the current proposal block. */
  readonly currentBlockAttention?: "Causal" | "Bidirectional"
  /** Optional attention window used by the proposal block. */
  readonly attentionWindow?: number
  /** Builds candidate token ids from anchor tokens and shared target weights. */
  readonly build: (
    params: Params,
    anchorTokens: Tensor.Any,
    tokenEmbedding: Tensor.Any,
    lmHead: Tensor.Any,
    maxDraftTokens: number
  ) => Effect.Effect<Tensor.Lazy, ModelError | Tensor.TensorError, Runtime.Runtime>
  /** Builds candidate token ids together with their probability rows. */
  readonly buildWithProbabilities?: (
    params: Params,
    anchorTokens: Tensor.Any,
    tokenEmbedding: Tensor.Any,
    lmHead: Tensor.Any,
    maxDraftTokens: number
  ) => Effect.Effect<ParallelBlockOutput, ModelError | Tensor.TensorError, Runtime.Runtime>
  /** Rebuilds authoritative proposer key/value rows from target hidden taps. */
  readonly replay: (
    params: Params,
    targetRows: ReadonlyArray<Tensor.Any>
  ) => Effect.Effect<ReadonlyArray<KeyValue>, ModelError | Tensor.TensorError, Runtime.Runtime>
}

/**
 * The complete public speculative-proposer language accepted by model inference.
 *
 * @since 0.1.0
 * @category models
 */
export type Artifact = Autoregressive | HistoryLookup | ParallelBlock

/**
 * Constructs an exact autoregressive proposer without tracing or compiling it.
 *
 * @since 0.1.0
 * @category constructors
 */
export const autoregressive = (
  model: Model,
  params: Params,
  options: { readonly vocabulary: number; readonly maxDraftTokens: number }
): Autoregressive => ({ _tag: "Autoregressive", model, params, ...options })

/**
 * Constructs a deterministic suffix n-gram history proposer.
 *
 * @since 0.1.0
 * @category constructors
 */
export const historyLookup = (options: Omit<HistoryLookup, "_tag">): HistoryLookup => ({
  _tag: "HistoryLookup",
  ...options
})

/**
 * Constructs one replayable fixed-width parallel-block proposer.
 *
 * @since 0.1.0
 * @category constructors
 */
export const parallelBlock = (options: Omit<ParallelBlock, "_tag">): ParallelBlock => ({
  _tag: "ParallelBlock",
  ...options
})
