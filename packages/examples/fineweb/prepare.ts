import * as Tokenizer from "@effect-torch/tokenizers"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { parquetMetadataAsync, parquetReadObjects } from "hyparquet"
import { compressors } from "hyparquet-compressors"
import fs from "node:fs"

// FineWeb-Edu (sample-10BT, shard 000) → GPT-2 BPE token bins. Each
// document is encoded and terminated by <|endoftext|> (id 50256, parsed
// as a special token because the config is specialTokens: "Always");
// ids fit u16 (vocab 50257), so the bins are plain little-endian u16 —
// the same flat format nano-gpt trains from. Split is positional: the
// first 99% of rows is train, the last 1% val (the shard is a random
// sample, so position is an unbiased cut).

const PARQUET = new URL("../data/fineweb-10BT-000.parquet", import.meta.url).pathname
const TOKENIZER_JSON = new URL("../data/gpt2-tokenizer.json", import.meta.url).pathname
const TRAIN_BIN = new URL("../data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("../data/fineweb-val.bin", import.meta.url).pathname
const EOT = "<|endoftext|>"
const VAL_FRACTION = 0.01

// hyparquet reads through an AsyncBuffer; backed by the file descriptor
// so the 2.15GB shard never materializes in memory — each slice() reads
// exactly the column chunks of the requested rows.
const parquetFile = (path: string) => {
  const fd = fs.openSync(path, "r")
  return {
    byteLength: fs.fstatSync(fd).size,
    slice(start: number, end: number = fs.fstatSync(fd).size) {
      const length = end - start
      const buffer = Buffer.allocUnsafe(length)
      fs.readSync(fd, buffer, 0, length, start)
      return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + length)
    }
  }
}

const program = Effect.gen(function*() {
  const file = parquetFile(PARQUET)
  const metadata = yield* Effect.tryPromise(() => parquetMetadataAsync(file))
  const rows = Number(metadata.num_rows)
  const splitRow = Math.floor(rows * (1 - VAL_FRACTION))
  // One parquet row group per step: reads and tokenization stay chunked.
  const group = Math.ceil(rows / metadata.row_groups.length)
  yield* Effect.log(`fineweb-prepare: ${rows} documents, split at row ${splitRow}`)

  const tokenizer = yield* Tokenizer.fromFile(TOKENIZER_JSON, {
    padding: Tokenizer.paddingNone,
    truncation: Tokenizer.truncationNone,
    specialTokens: "Always"
  })
  expect50256(tokenizer)

  const trainFd = fs.openSync(TRAIN_BIN, "w")
  const valFd = fs.openSync(VAL_BIN, "w")
  let trainTokens = 0
  let valTokens = 0
  const started = Date.now()
  for (let start = 0; start < rows; start += group) {
    const end = Math.min(start + group, rows)
    const documents = yield* Effect.tryPromise(() =>
      parquetReadObjects({ file, compressors, columns: ["text"], rowStart: start, rowEnd: end })
    )
    const flat = yield* tokenizer.encodeBatchConcat(documents.map((document) => document.text + EOT))
    const ids = flat.data
    const u16 = new Uint16Array(ids.length)
    for (let i = 0; i < ids.length; i++) u16[i] = ids[i]
    const bytes = Buffer.from(u16.buffer, u16.byteOffset, u16.byteLength)
    if (start >= splitRow) {
      fs.writeSync(valFd, bytes)
      valTokens += ids.length
    } else {
      fs.writeSync(trainFd, bytes)
      trainTokens += ids.length
    }
    if (start % (group * 50) === 0 || end === rows) {
      yield* Effect.log(
        `row ${end}/${rows}  train ${(trainTokens / 1e6).toFixed(0)}M tokens  ${
          ((Date.now() - started) / 1000).toFixed(0)
        }s`
      )
    }
  }
  fs.closeSync(trainFd)
  fs.closeSync(valFd)
  yield* Effect.log(
    `done: ${(trainTokens / 1e6).toFixed(1)}M train tokens, ${(valTokens / 1e6).toFixed(1)}M val tokens in ${
      ((Date.now() - started) / 1000).toFixed(0)
    }s`
  )
})

const expect50256 = (tokenizer: Tokenizer.Tokenizer) => {
  const id = tokenizer.tokenToId(EOT)
  if (id._tag !== "Some" || id.value !== 50256) {
    throw new Error(`expected ${EOT} to be id 50256, got ${id._tag === "Some" ? id.value : "none"}`)
  }
}

NodeRuntime.runMain(program)
