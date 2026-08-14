#!/bin/sh
# Downloads the FineWeb pre-training inputs into examples/data/
# (gitignored): one FineWeb-Edu sample-10BT parquet shard (~2.15GB,
# ~750M tokens across 726k documents) and the pretrained GPT-2 BPE
# tokenizer. prepare.ts turns these into u16 token bins.
set -e
cd "$(dirname "$0")/.."
mkdir -p data

if [ ! -f data/fineweb-10BT-000.parquet ]; then
  echo "downloading FineWeb-Edu sample-10BT shard 000 (2.15GB)..."
  curl -L -o data/fineweb-10BT-000.parquet \
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet"
fi

if [ ! -f data/gpt2-tokenizer.json ]; then
  echo "downloading GPT-2 tokenizer..."
  curl -L -o data/gpt2-tokenizer.json \
    "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json"
fi

echo "done: $(du -h data/fineweb-10BT-000.parquet data/gpt2-tokenizer.json | tr '\n' ' ')"
