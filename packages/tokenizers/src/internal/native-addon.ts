/**
 * Compile-time declarations for the package-local Node-API addon.
 *
 * This file contains no runtime implementation. `native.ts` imports these
 * declarations with `import type`, then casts the object returned by `require`.
 * Consequently TypeScript cannot verify this ABI against the Rust exports; the
 * names, argument shapes, defaults, sync/async behavior, and ownership rules
 * below must be kept in lockstep with the addon. The package build deletes the
 * generated internal declaration files after compiling, leaving `index.d.ts`
 * as the public declaration boundary.
 *
 * @internal
 */

/** Native training-source object copied from the facade at the NAPI call. @internal */
export interface NativeTrainSource {
  /** Runtime discriminator; native code accepts only `Files` or `Texts`. */
  readonly tag: string
  /** Required when `tag` is `Files`; other field combinations fail natively. */
  readonly paths?: Array<string>
  /** Required when `tag` is `Texts`; other field combinations fail natively. */
  readonly texts?: Array<string>
}

/** Native trainer input. Arrays and strings are converted to Rust-owned values. @internal */
export interface NativeTrainConfig {
  /** One of `BPE`, `WordPiece`, `Unigram`, or `WordLevel`. */
  readonly model: string
  /** Node-API converts this JavaScript number to a Rust `u32`. */
  readonly vocabSize: number
  /** Node-API converts this JavaScript number to a Rust `u32`. */
  readonly minFrequency: number
  readonly specialTokens: Array<string>
  readonly source: NativeTrainSource
}

/**
 * ABI declaration for the constructor actually exported by the `.node` file.
 * The declared class emits no JavaScript value. Native instances own immutable
 * CPU tokenizer state; promises retain native references needed by background
 * work, and Node-API finalization releases state after it becomes unreachable.
 *
 * @internal
 */
export declare class NativeTokenizer {
  /** Synchronously reads and parses a `tokenizer.json` file. */
  static fromFile(path: string, parseSpecials: boolean): NativeTokenizer
  /** Synchronously parses an in-memory `tokenizer.json` document. */
  static fromJson(json: string, parseSpecials: boolean): NativeTokenizer
  /**
   * Runs corpus consumption and training on a blocking native worker. Progress
   * callbacks are posted non-blockingly to JavaScript and do not cancel work.
   */
  static train(
    config: NativeTrainConfig,
    parseSpecials: boolean,
    progress: (event: [number, number]) => void,
    progressEveryBytes: number
  ): Promise<NativeTokenizer>
  /** Includes model, added, and special tokens. */
  get vocabSize(): number
  /** Exact model-or-added-vocabulary lookup. */
  tokenToId(token: string): number | null
  /** Exact unsigned-32-bit id lookup. */
  idToToken(id: number): string | null
  /** Synchronously serializes to a directly created or truncated file. */
  save(path: string): void
  /**
   * Synchronously encodes one sequence. The returned writable array is no
   * longer retained by native tokenizer state.
   */
  encode(text: string, addSpecialTokens?: boolean): Uint32Array
  /**
   * Encodes texts in parallel on a blocking native worker, preserves order,
   * and always passes `addSpecialTokens = true` internally.
   */
  encodeBatch(texts: Array<string>): Promise<Array<Uint32Array>>
  /** Synchronously decodes one flat id sequence. */
  decode(ids: Array<number>, skipSpecialTokens?: boolean): string
  /** Synchronously decodes independent rows, potentially in native parallel. */
  decodeBatch(ids: Array<Array<number>>, skipSpecialTokens?: boolean): Array<string>
  /** Synchronously renders `template` against the parsed JSON context object. */
  applyChatTemplate(template: string, contextJson: string): string
}

/** Runtime export map expected from the selected addon binary. @internal */
export interface NativeAddon {
  readonly NativeTokenizer: typeof NativeTokenizer
}
