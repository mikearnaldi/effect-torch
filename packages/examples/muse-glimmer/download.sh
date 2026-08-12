#!/bin/sh
# Downloads the pinned text GGUF and matching tokenizer into examples/data/.
# curl resumes partial transfers; completed files are always size/hash checked.
set -eu

cd "$(dirname "$0")/.."
mkdir -p data

MODEL_REPOSITORY="unsloth/Muse-Glimmer-30B-GGUF"
MODEL_REVISION="faa5b025c584459c13febfa5c59883516710ae39"
MODEL_FILE="Muse-Glimmer-30B-UD-Q2_K_XL.gguf"
MODEL_BYTES="12444212256"
MODEL_SHA256="3d63a1daff23fdc2a6927316151e855cacffe89b5cb9b9397a5aec0c412ec08d"

TOKENIZER_REPOSITORY="unsloth/Muse-Glimmer-30B"
TOKENIZER_REVISION="55a49b91a33d176bc99db7569f678d4c64cd91a1"
TOKENIZER_FILE="muse-glimmer-tokenizer.json"
TOKENIZER_BYTES="28129897"
TOKENIZER_SHA256="c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

size() {
  wc -c < "$1" | tr -d ' '
}

download() {
  repository="$1"
  revision="$2"
  remote_file="$3"
  output="$4"
  expected_bytes="$5"
  expected_sha256="$6"

  if [ -f "$output" ] && [ "$(size "$output")" = "$expected_bytes" ] && [ "$(sha256 "$output")" = "$expected_sha256" ]; then
    echo "verified: $output"
    return
  fi

  if [ -f "$output" ] && [ "$(size "$output")" -ge "$expected_bytes" ]; then
    echo "discarding invalid completed file: $output"
    rm "$output"
  fi

  echo "downloading $remote_file to $output..."
  curl --fail --location --retry 5 --retry-all-errors --continue-at - \
    --output "$output" \
    "https://huggingface.co/$repository/resolve/$revision/$remote_file"

  actual_bytes="$(size "$output")"
  if [ "$actual_bytes" != "$expected_bytes" ]; then
    echo "size mismatch for $output: expected $expected_bytes, got $actual_bytes" >&2
    exit 1
  fi
  actual_sha256="$(sha256 "$output")"
  if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "SHA-256 mismatch for $output: expected $expected_sha256, got $actual_sha256" >&2
    exit 1
  fi
  echo "verified: $output"
}

download \
  "$MODEL_REPOSITORY" \
  "$MODEL_REVISION" \
  "$MODEL_FILE" \
  "data/$MODEL_FILE" \
  "$MODEL_BYTES" \
  "$MODEL_SHA256"

download \
  "$TOKENIZER_REPOSITORY" \
  "$TOKENIZER_REVISION" \
  "tokenizer.json" \
  "data/$TOKENIZER_FILE" \
  "$TOKENIZER_BYTES" \
  "$TOKENIZER_SHA256"

echo "Muse-Glimmer artifacts are ready in packages/examples/data/"
