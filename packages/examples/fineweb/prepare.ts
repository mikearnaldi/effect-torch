import * as Tokenizer from "@effect-torch/tokenizers"
import { NodeRuntime } from "@effect/platform-node"
import { Effect } from "effect"
import { parquetMetadataAsync, parquetReadObjects } from "hyparquet"
import { compressors } from "hyparquet-compressors"
import fs from "node:fs"

// Converts FineWeb-Edu sample-10BT shard 000 into the flat token streams
// consumed by the training examples. The script encodes each document
// separately and appends <|endoftext|> (id 50256). The tokenizer file must
// already register that sentinel. `specialTokens: "Always"` enables matching
// it as a special token. GPT-2's 50,257 IDs fit in u16, so the output uses
// headerless little-endian u16 values.
//
// The positional split targets 99/1, but output switches only at processing
// chunk boundaries. If `splitRow` falls inside a chunk, the whole chunk goes to
// training. With the bundled 1,000-row groups, validation starts at row 719,000
// instead of row 718,740. The script truncates output files in place instead of
// replacing them atomically. An interrupted run leaves partial bins with no
// embedded length or completion marker.

const PARQUET = new URL("../data/fineweb-10BT-000.parquet", import.meta.url).pathname
const TOKENIZER_JSON = new URL("../data/gpt2-tokenizer.json", import.meta.url).pathname
const TRAIN_BIN = new URL("../data/fineweb-train.bin", import.meta.url).pathname
const VAL_BIN = new URL("../data/fineweb-val.bin", import.meta.url).pathname
const EOT = "<|endoftext|>"
const VAL_FRACTION = 0.01

// This random-access AsyncBuffer lets hyparquet read byte ranges from the file
// descriptor without loading the multi-GB parquet file into JavaScript memory.
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
  // The bundled shard has equal-sized row groups, so this count aligns each
  // read and tokenization batch with one group.
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
