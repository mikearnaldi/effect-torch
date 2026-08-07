/** @internal */
export interface NativeTrainSource {
  readonly tag: string
  readonly paths?: Array<string>
  readonly texts?: Array<string>
}

/** @internal */
export interface NativeTrainConfig {
  readonly model: string
  readonly vocabSize: number
  readonly minFrequency: number
  readonly specialTokens: Array<string>
  readonly source: NativeTrainSource
}

/** @internal */
export declare class NativeTokenizer {
  static fromFile(path: string, parseSpecials: boolean): NativeTokenizer
  static fromJson(json: string, parseSpecials: boolean): NativeTokenizer
  static train(
    config: NativeTrainConfig,
    parseSpecials: boolean,
    progress: (event: [number, number]) => void,
    progressEveryBytes: number
  ): Promise<NativeTokenizer>
  get vocabSize(): number
  tokenToId(token: string): number | null
  idToToken(id: number): string | null
  save(path: string): void
  encode(text: string): Uint32Array
  encodeBatch(texts: Array<string>): Promise<Array<Uint32Array>>
  decode(ids: Array<number>): string
  decodeBatch(ids: Array<Array<number>>): Array<string>
}

/** @internal */
export interface NativeAddon {
  readonly NativeTokenizer: typeof NativeTokenizer
}
