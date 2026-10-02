#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
manifest=${MANIFEST:-"${script_directory}/manifest.json"}
environment=${1:-"${HOME}/.cache/effect-torch/vllm-0.24.0"}

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  printf 'The pinned vLLM wheel requires Linux x86_64\n' >&2
  exit 1
fi

if [[ -e ${environment} ]]; then
  printf 'Refusing to overwrite %s\n' "${environment}" >&2
  exit 1
fi

target=()
while IFS= read -r value; do
  target+=("${value}")
done < <(python3 - "${manifest}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    manifest = json.load(source)
print(manifest["target"]["wheelUrl"])
print(manifest["target"]["wheelSha256"])
print(manifest["target"]["minimumVersion"])
PY
)

if [[ ${#target[@]} -ne 3 ]]; then
  printf 'Cannot read the pinned vLLM target from %s\n' "${manifest}" >&2
  exit 1
fi

url=${target[0]}
sha256=${target[1]}
version=${target[2]}

cleanup() {
  rm -rf -- "${environment}"
}
trap cleanup ERR INT TERM

uv venv --python python3.12 "${environment}"
uv pip install \
  --python "${environment}/bin/python" \
  "vllm @ ${url}#sha256=${sha256}"

"${environment}/bin/python" - "${url}" "${sha256}" "${version}" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

url, sha256, version = sys.argv[1:]
distribution = importlib.metadata.distribution("vllm")
if distribution.version != version:
    raise SystemExit(f"expected vLLM {version}, got {distribution.version}")
direct = json.loads(distribution.read_text("direct_url.json"))
if direct.get("url") != url:
    raise SystemExit("installed vLLM wheel URL differs")
archive = direct.get("archive_info") or {}
hashes = archive.get("hashes") or {}
actual = hashes.get("sha256") or str(archive.get("hash", "")).removeprefix("sha256=")
if actual and actual != sha256:
    raise SystemExit("installed vLLM wheel SHA-256 differs")
marker = {"url": url, "sha256": sha256, "version": version}
(Path(sys.prefix) / "effect-torch-vllm.json").write_text(
    json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
)
PY

trap - ERR INT TERM
printf 'vLLM %s environment ready: %s\n' "${version}" "${environment}"
printf 'Use VLLM_PYTHON=%s/bin/python\n' "${environment}"
