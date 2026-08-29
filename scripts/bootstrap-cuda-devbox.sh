#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "${script_directory}/.." && pwd)

fail() {
  printf 'CUDA devbox bootstrap: %s\n' "$1" >&2
  exit 1
}

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  fail "the CUDA shell supports only x86_64-linux"
fi

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available"
nvidia-smi -L >/dev/null || fail "the NVIDIA driver cannot enumerate a GPU"

if [[ -f /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]]; then
  # This path is installed by Nix.
  # shellcheck disable=SC1091
  . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
fi
export PATH="/nix/var/nix/profiles/default/bin:${PATH}"

if ! command -v nix >/dev/null 2>&1; then
  [[ $(id -u) -eq 0 ]] || fail "installing Nix in a container requires root"
  if ! command -v curl >/dev/null 2>&1; then
    command -v apt-get >/dev/null 2>&1 || fail "curl is required to install Nix"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl
  fi

  curl --proto '=https' --tlsv1.2 -fsSL https://install.determinate.systems/nix |
    sh -s -- install linux \
      --diagnostic-endpoint "" \
      --extra-conf "sandbox = false" \
      --init none \
      --no-confirm

  # The installer creates this profile script.
  # shellcheck disable=SC1091
  . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
fi

if [[ ${EFFECT_TORCH_CUDA_SHELL:-0} != 1 ]]; then
  exec nix develop "${repository}#cuda" --command \
    env EFFECT_TORCH_CUDA_SHELL=1 bash "${BASH_SOURCE[0]}"
fi

cd "${repository}"

# A tar upload has no Git metadata. Mark its source files as tracked so later
# flake evaluations continue to exclude node_modules and other ignored outputs.
if [[ ! -d .git ]]; then
  git init --quiet
  git add .
fi

printf 'Installing workspace dependencies...\n'
pnpm install --frozen-lockfile
cargo fetch --locked

temporary_directory=$(mktemp -d)
trap 'rm -rf "${temporary_directory}"' EXIT

printf 'Compiling the CUDA toolchain check for %s...\n' "${EFFECT_TORCH_CUDA_ARCH}"
nvcc \
  -std=c++17 \
  -arch="${EFFECT_TORCH_CUDA_ARCH}" \
  -I"${CUDA_PATH}/include" \
  -L"${CUDA_PATH}/lib" \
  -Xlinker "-rpath=${CUDA_PATH}/lib" \
  scripts/check-cuda-devbox.cu \
  -lcublasLt \
  -lnvrtc \
  -o "${temporary_directory}/check-cuda-devbox"

"${temporary_directory}/check-cuda-devbox" "${EFFECT_TORCH_CUDA_DEVICE:-0}"

printf '\nCUDA devbox ready. Re-enter it with: nix develop .#cuda\n'
