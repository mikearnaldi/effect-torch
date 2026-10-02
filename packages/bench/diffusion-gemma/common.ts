import { Schema } from "effect"
import { createHash } from "node:crypto"
import * as fs from "node:fs"

export interface PromptSpec {
  readonly id: string
  readonly question: string
  readonly offset: number
  readonly baseTokens: number
}

const ManifestSchema = Schema.Struct({
  schemaVersion: Schema.Literal(1),
  model: Schema.Struct({
    id: Schema.String,
    revision: Schema.String,
    precision: Schema.Literal("bfloat16"),
    tokenizerSha256: Schema.String,
    chatTemplateSha256: Schema.String,
    initializedStateSha256: Schema.String
  }),
  generation: Schema.Struct({
    canvasLength: Schema.Number,
    maxNewTokens: Schema.Number,
    maxSteps: Schema.Number,
    entropyBound: Schema.Number,
    minTemperature: Schema.Number,
    maxTemperature: Schema.Number,
    stabilityThreshold: Schema.Number,
    confidenceThreshold: Schema.Number,
    eosTokenIds: Schema.Array(Schema.Number),
    padTokenId: Schema.Number,
    enableThinking: Schema.Boolean,
    seed: Schema.Number
  }),
  target: Schema.Struct({
    engine: Schema.Literal("vllm"),
    minimumVersion: Schema.String,
    dockerImage: Schema.String,
    imageCommit: Schema.String,
    wheelUrl: Schema.String,
    wheelSha256: Schema.String,
    supportMergeCommit: Schema.String
  }),
  hardware: Schema.Struct({
    gpu: Schema.String,
    minimumMemoryBytes: Schema.Number,
    gpuCount: Schema.Literal(1)
  }),
  deployment: Schema.Struct({
    maxTokens: Schema.Number,
    gpuMemoryUtilization: Schema.Number
  }),
  matrix: Schema.Struct({
    promptTargets: Schema.Array(Schema.Number),
    outputTokens: Schema.Array(Schema.Number),
    concurrencies: Schema.Array(Schema.Number),
    warmupRuns: Schema.Number,
    measuredRuns: Schema.Number,
    cooldownMilliseconds: Schema.Number
  }),
  promptPaddingOverheadTokens: Schema.Number,
  promptWords: Schema.Array(Schema.String),
  prompts: Schema.Array(
    Schema.Struct({ id: Schema.String, question: Schema.String, offset: Schema.Number, baseTokens: Schema.Number })
  )
})

export interface Manifest {
  readonly schemaVersion: 1
  readonly model: {
    readonly id: string
    readonly revision: string
    readonly precision: "bfloat16"
    readonly tokenizerSha256: string
    readonly chatTemplateSha256: string
    readonly initializedStateSha256: string
  }
  readonly generation: {
    readonly canvasLength: number
    readonly maxNewTokens: number
    readonly maxSteps: number
    readonly entropyBound: number
    readonly minTemperature: number
    readonly maxTemperature: number
    readonly stabilityThreshold: number
    readonly confidenceThreshold: number
    readonly eosTokenIds: ReadonlyArray<number>
    readonly padTokenId: number
    readonly enableThinking: boolean
    readonly seed: number
  }
  readonly target: {
    readonly engine: "vllm"
    readonly minimumVersion: string
    readonly dockerImage: string
    readonly imageCommit: string
    readonly wheelUrl: string
    readonly wheelSha256: string
    readonly supportMergeCommit: string
  }
  readonly hardware: {
    readonly gpu: string
    readonly minimumMemoryBytes: number
    readonly gpuCount: 1
  }
  readonly deployment: {
    readonly maxTokens: number
    readonly gpuMemoryUtilization: number
  }
  readonly matrix: {
    readonly promptTargets: ReadonlyArray<number>
    readonly outputTokens: ReadonlyArray<number>
    readonly concurrencies: ReadonlyArray<number>
    readonly warmupRuns: number
    readonly measuredRuns: number
    readonly cooldownMilliseconds: number
  }
  readonly promptPaddingOverheadTokens: number
  readonly promptWords: ReadonlyArray<string>
  readonly prompts: ReadonlyArray<PromptSpec>
}

export interface PromptCase {
  readonly id: string
  readonly targetTokens: number
  readonly messages: ReadonlyArray<{ readonly role: "user"; readonly content: string }>
  readonly contentSha256: string
}

export const loadManifest = (path: string): Manifest =>
  Schema.decodeUnknownSync(ManifestSchema)(JSON.parse(fs.readFileSync(path, "utf8")))

export const promptCases = (manifest: Manifest): ReadonlyArray<PromptCase> =>
  manifest.matrix.promptTargets.flatMap((targetTokens) =>
    manifest.prompts.map((prompt) => {
      const paddingTokens = targetTokens - prompt.baseTokens - manifest.promptPaddingOverheadTokens
      if (paddingTokens < 1) {
        throw new Error(`prompt target ${targetTokens} is too small for ${prompt.id}`)
      }
      const padding = Array.from(
        { length: paddingTokens },
        (_, index) => manifest.promptWords[(prompt.offset + index) % manifest.promptWords.length]
      ).join(" ")
      const content = `${prompt.question}\n\nContext: ${padding}`

      return {
        id: `${prompt.id}-p${targetTokens}`,
        targetTokens,
        messages: [{ role: "user" as const, content }],
        contentSha256: createHash("sha256").update(content).digest("hex")
      }
    })
  )

export const percentile = (values: ReadonlyArray<number>, quantile: number): number => {
  if (values.length === 0) return Number.NaN
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[Math.min(sorted.length - 1, Math.floor(quantile * sorted.length))]!
}
