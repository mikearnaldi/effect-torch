#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "${script_directory}/.." && pwd)
config=${EFFECT_TORCH_CUDA_DEVBOX_CONFIG:-"${repository}/.cuda-devbox.env"}

usage() {
  cat <<'USAGE'
Usage: scripts/cuda-devbox.sh <command> [arguments]

Commands:
  key                  Create and register the repository-local SSH key
  template             Create or update the prebuilt RunPod template
  create               Create a RunPod devbox and save its SSH endpoint
  destroy              Delete the configured RunPod devbox
  show                 Print the resolved devbox configuration
  check                Verify the SSH connection and NVIDIA device
  sync                 Upload the current non-ignored worktree
  bootstrap            Upload the worktree and run the CUDA bootstrap
  ssh                  Open an interactive SSH session
  run <command> [...]  Run a command from the remote repository
USAGE
}

fail() {
  printf 'CUDA devbox: %s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    fail "$1 is unavailable; enter the Nix development shell"
}

enabled() {
  case $1 in
    1|true|TRUE|yes|YES) return 0 ;;
    0|false|FALSE|no|NO) return 1 ;;
    *) fail "expected a boolean value, got: $1" ;;
  esac
}

command=${1:-}
case ${command} in
  key|template|create|destroy|show|check|sync|bootstrap|ssh|run) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

if [[ -f ${config} ]]; then
  # The caller chooses this local configuration file.
  # shellcheck disable=SC1090
  . "${config}"
fi
if [[ -n ${RUNPOD_API_KEY:-} ]]; then
  export RUNPOD_API_KEY
fi

pod_id=${CUDA_DEVBOX_POD_ID:-}
address=${CUDA_DEVBOX_ADDRESS:-}
port=${CUDA_DEVBOX_PORT:-22}
remote_directory=${CUDA_DEVBOX_DIRECTORY:-/root/effect-torch}
known_hosts=${CUDA_DEVBOX_KNOWN_HOSTS_FILE:-"${repository}/.cuda-devbox-known-hosts"}
identity_file=${CUDA_DEVBOX_IDENTITY_FILE:-}
managed_identity_file=${CUDA_DEVBOX_MANAGED_IDENTITY_FILE:-"${repository}/.cuda-devbox-ssh-key"}
managed_public_key=${managed_identity_file}.pub

pod_name=${CUDA_DEVBOX_NAME:-effect-torch-cuda}
gpu_id=${CUDA_DEVBOX_GPU_ID:-NVIDIA RTX PRO 6000 Blackwell Server Edition}
base_template_id=${CUDA_DEVBOX_TEMPLATE_ID:-runpod-torch-v21}
managed_template_id=${CUDA_DEVBOX_MANAGED_TEMPLATE_ID:-}
template_id=${managed_template_id:-${base_template_id}}
template_image=${CUDA_DEVBOX_TEMPLATE_IMAGE:-ghcr.io/mikearnaldi/effect-torch:cuda-devbox}
template_image_digest=${CUDA_DEVBOX_TEMPLATE_IMAGE_DIGEST:-}
template_name=${CUDA_DEVBOX_TEMPLATE_NAME:-effect-torch-cuda}
registry_auth_id=${CUDA_DEVBOX_REGISTRY_AUTH_ID:-}
image=${CUDA_DEVBOX_IMAGE:-}
cloud_type=${CUDA_DEVBOX_CLOUD_TYPE:-SECURE}
data_center_ids=${CUDA_DEVBOX_DATA_CENTER_IDS:-}
public_ip=${CUDA_DEVBOX_PUBLIC_IP:-0}
gpu_count=${CUDA_DEVBOX_GPU_COUNT:-1}
container_disk_gb=${CUDA_DEVBOX_CONTAINER_DISK_GB:-40}
minimum_cuda_version=${CUDA_DEVBOX_MIN_CUDA_VERSION:-12.9}
wait_seconds=${CUDA_DEVBOX_WAIT_SECONDS:-600}

write_connection_state() {
  local new_pod_id=$1
  local new_address=$2
  local new_port=$3
  local config_directory temporary

  config_directory=$(dirname -- "${config}")
  [[ -d ${config_directory} ]] || fail "config directory does not exist: ${config_directory}"
  temporary=$(mktemp "${config}.tmp.XXXXXX")
  if [[ -f ${config} ]]; then
    grep -Ev '^(CUDA_DEVBOX_POD_ID|CUDA_DEVBOX_ADDRESS|CUDA_DEVBOX_PORT)=' \
      "${config}" > "${temporary}" || true
  fi
  printf 'CUDA_DEVBOX_POD_ID=%q\n' "${new_pod_id}" >> "${temporary}"
  if [[ -n ${new_address} ]]; then
    printf 'CUDA_DEVBOX_ADDRESS=%q\n' "${new_address}" >> "${temporary}"
    printf 'CUDA_DEVBOX_PORT=%q\n' "${new_port}" >> "${temporary}"
  fi
  mv "${temporary}" "${config}"
}

write_identity_state() {
  local new_identity_file=$1
  local config_directory temporary

  config_directory=$(dirname -- "${config}")
  [[ -d ${config_directory} ]] || fail "config directory does not exist: ${config_directory}"
  temporary=$(mktemp "${config}.tmp.XXXXXX")
  if [[ -f ${config} ]]; then
    grep -Ev '^CUDA_DEVBOX_IDENTITY_FILE=' "${config}" > "${temporary}" || true
  fi
  printf 'CUDA_DEVBOX_IDENTITY_FILE=%q\n' "${new_identity_file}" >> "${temporary}"
  mv "${temporary}" "${config}"
}

ensure_managed_key() {
  local identity_directory key_data key_type public_key temporary

  require_command ssh-keygen
  identity_directory=$(dirname -- "${managed_identity_file}")
  mkdir -p "${identity_directory}"
  if [[ ! -f ${managed_identity_file} ]]; then
    [[ ! -e ${managed_public_key} ]] ||
      fail "public key exists without its private key: ${managed_public_key}"
    umask 077
    ssh-keygen -q -t ed25519 -N '' -C effect-torch-cuda-devbox -f "${managed_identity_file}"
    printf 'Generated unencrypted SSH key: %s\n' "${managed_identity_file}"
  fi
  chmod 600 "${managed_identity_file}"
  if ! public_key=$(ssh-keygen -y -P '' -f "${managed_identity_file}" 2>/dev/null); then
    fail "managed identity must be an unencrypted private key: ${managed_identity_file}"
  fi
  read -r key_type key_data _ <<< "${public_key}"
  [[ -n ${key_type} && -n ${key_data} ]] ||
    fail "could not read the managed public key: ${managed_identity_file}"
  temporary=$(mktemp "${managed_public_key}.tmp.XXXXXX")
  printf '%s %s effect-torch-cuda-devbox\n' "${key_type}" "${key_data}" > "${temporary}"
  chmod 644 "${temporary}"
  mv "${temporary}" "${managed_public_key}"
}

register_managed_key() {
  local fingerprint response

  require_command runpodctl
  require_command jq
  read -r _ fingerprint _ < <(ssh-keygen -lf "${managed_public_key}" -E sha256)
  response=$(runpodctl ssh list-keys --output json)
  if printf '%s' "${response}" |
    jq -e --arg fingerprint "${fingerprint}" \
      '(.keys // [])[] | select(.fingerprint == $fingerprint)' >/dev/null; then
    printf 'RunPod already has SSH key %s.\n' "${fingerprint}"
    return 0
  fi
  runpodctl ssh add-key --key-file "${managed_public_key}" --output json
  printf 'Registered SSH key %s with RunPod.\n' "${fingerprint}"
}

clear_connection_state() {
  local temporary

  [[ -f ${config} ]] || return 0
  temporary=$(mktemp "${config}.tmp.XXXXXX")
  grep -Ev '^(CUDA_DEVBOX_POD_ID|CUDA_DEVBOX_ADDRESS|CUDA_DEVBOX_PORT)=' \
    "${config}" > "${temporary}" || true
  mv "${temporary}" "${config}"
}

write_template_state() {
  local new_template_id=$1
  local new_image_digest=$2
  local config_directory temporary

  config_directory=$(dirname -- "${config}")
  [[ -d ${config_directory} ]] || fail "config directory does not exist: ${config_directory}"
  temporary=$(mktemp "${config}.tmp.XXXXXX")
  if [[ -f ${config} ]]; then
    grep -Ev '^(CUDA_DEVBOX_MANAGED_TEMPLATE_ID|CUDA_DEVBOX_TEMPLATE_IMAGE_DIGEST)=' \
      "${config}" > "${temporary}" || true
  fi
  printf 'CUDA_DEVBOX_MANAGED_TEMPLATE_ID=%q\n' "${new_template_id}" >> "${temporary}"
  printf 'CUDA_DEVBOX_TEMPLATE_IMAGE_DIGEST=%q\n' "${new_image_digest}" >> "${temporary}"
  mv "${temporary}" "${config}"
}

manage_template() {
  local digest pinned_image response new_template_id
  local -a registry_arguments=()

  require_command runpodctl
  require_command jq
  require_command crane

  printf 'Resolving prebuilt image %s...\n' "${template_image}"
  if ! digest=$(crane digest "${template_image}"); then
    fail "could not resolve ${template_image}; publish the image and make the GHCR package public"
  fi
  [[ ${digest} =~ ^sha256:[0-9a-f]{64}$ ]] || fail "registry returned an invalid image digest: ${digest}"
  pinned_image="${template_image}@${digest}"
  if [[ -n ${registry_auth_id} ]]; then
    registry_arguments=(--registry-auth-id "${registry_auth_id}")
  fi

  if [[ -n ${managed_template_id} ]]; then
    printf 'Updating RunPod template %s...\n' "${managed_template_id}"
    runpodctl template update "${managed_template_id}" \
      --name "${template_name}" \
      --image "${pinned_image}" \
      --container-disk-in-gb "${container_disk_gb}" \
      --ports 22/tcp \
      --port-labels 22=SSH \
      "${registry_arguments[@]}" \
      --output json
    new_template_id=${managed_template_id}
  else
    printf 'Creating RunPod template %s...\n' "${template_name}"
    response=$(runpodctl template create \
      --name "${template_name}" \
      --image "${pinned_image}" \
      --container-disk-in-gb "${container_disk_gb}" \
      --ports 22/tcp \
      --port-labels 22=SSH \
      "${registry_arguments[@]}" \
      --output json)
    if ! new_template_id=$(printf '%s' "${response}" | jq -er '.id | select(type == "string" and length > 0)'); then
      printf '%s\n' "${response}" >&2
      fail "RunPod created a template but returned no template ID"
    fi
  fi

  managed_template_id=${new_template_id}
  template_id=${new_template_id}
  template_image_digest=${digest}
  write_template_state "${managed_template_id}" "${template_image_digest}"
  printf 'RunPod template ready: %s using %s\n' "${managed_template_id}" "${pinned_image}"
}

wait_for_connection() {
  local response ip resolved_port runtime_status runtime_reason
  local startup_failure startup_failure_count startup_failure_message
  local deadline=$((SECONDS + wait_seconds))

  printf 'Waiting up to %s seconds for the direct SSH endpoint...\n' "${wait_seconds}"
  while ((SECONDS < deadline)); do
    if response=$(runpodctl ssh info "${pod_id}" --output json 2>/dev/null); then
      ip=$(printf '%s' "${response}" | jq -r '.ip // empty')
      resolved_port=$(printf '%s' "${response}" | jq -r '.port // empty')
      if [[ -n ${ip} && ${resolved_port} =~ ^[0-9]+$ && ${resolved_port} -ge 1 && ${resolved_port} -le 65535 ]]; then
        address="root@${ip}"
        port=${resolved_port}
        write_connection_state "${pod_id}" "${address}" "${port}"
        rm -f "${known_hosts}"
        printf 'Devbox ready: %s:%s (pod %s)\n' "${address}" "${port}" "${pod_id}"
        return 0
      fi
    fi

    if response=$(runpodctl pod get "${pod_id}" --output json 2>/dev/null); then
      runtime_status=$(printf '%s' "${response}" | jq -r '.runtimeStatus // empty')
      runtime_reason=$(printf '%s' "${response}" | jq -r '.runtimeStatusReason // .lastStatusChange // empty')
      case ${runtime_status} in
        stopped|terminated)
          fail "pod ${pod_id} became ${runtime_status}: ${runtime_reason:-reason unavailable}"
          ;;
      esac
    fi

    if response=$(runpodctl pod logs "${pod_id}" --source system --tail 100 --max-wait 2s 2>/dev/null); then
      startup_failure=$(printf '%s' "${response}" | jq -rs '
        [.[] | select(.source == "system" and (.line | startswith("error creating container:")))] as $errors
        | select($errors | length >= 3)
        | [($errors | length), $errors[-1].line]
        | @tsv
      ')
      if [[ -n ${startup_failure} ]]; then
        startup_failure_count=${startup_failure%%$'\t'*}
        startup_failure_message=${startup_failure#*$'\t'}
        fail "pod ${pod_id} failed to create its container ${startup_failure_count} times: ${startup_failure_message}"
      fi
    fi
    sleep 5
  done

  fail "pod ${pod_id} was created but SSH was not ready; inspect it with runpodctl pod get ${pod_id} or delete it with this script"
}

create_devbox() {
  local response new_pod_id
  local -a arguments=(
    pod create
    --name "${pod_name}"
    --gpu-id "${gpu_id}"
    --gpu-count "${gpu_count}"
    --container-disk-in-gb "${container_disk_gb}"
    --cloud-type "${cloud_type}"
    --ports 22/tcp
    --min-cuda-version "${minimum_cuda_version}"
    --output json
  )

  require_command runpodctl
  require_command jq
  [[ -z ${pod_id} && -z ${address} ]] ||
    fail "a devbox is already configured; destroy it before creating another"
  [[ ${gpu_count} =~ ^[0-9]+$ && ${gpu_count} -ge 1 ]] ||
    fail "CUDA_DEVBOX_GPU_COUNT must be a positive integer"
  [[ ${container_disk_gb} =~ ^[0-9]+$ && ${container_disk_gb} -ge 1 ]] ||
    fail "CUDA_DEVBOX_CONTAINER_DISK_GB must be a positive integer"
  [[ ${wait_seconds} =~ ^[0-9]+$ && ${wait_seconds} -ge 1 ]] ||
    fail "CUDA_DEVBOX_WAIT_SECONDS must be a positive integer"
  case ${cloud_type} in
    SECURE|COMMUNITY) ;;
    *) fail "CUDA_DEVBOX_CLOUD_TYPE must be SECURE or COMMUNITY" ;;
  esac

  configure_managed_key

  if [[ -n ${image} ]]; then
    arguments+=(--image "${image}")
  else
    arguments+=(--template-id "${template_id}")
  fi
  if [[ -n ${data_center_ids} ]]; then
    arguments+=(--data-center-ids "${data_center_ids}")
  fi
  if enabled "${public_ip}"; then
    arguments+=(--public-ip)
  fi

  printf 'Creating RunPod devbox %s with %s...\n' "${pod_name}" "${gpu_id}"
  response=$(runpodctl "${arguments[@]}")
  if ! new_pod_id=$(printf '%s' "${response}" | jq -er '.id | select(type == "string" and length > 0)'); then
    printf '%s\n' "${response}" >&2
    fail "RunPod created a pod but returned no pod ID"
  fi

  pod_id=${new_pod_id}
  write_connection_state "${pod_id}" "" ""
  printf 'Created pod %s. Its ID is saved even if SSH setup times out.\n' "${pod_id}"
  wait_for_connection
}

destroy_devbox() {
  require_command runpodctl
  [[ -n ${pod_id} ]] || fail "CUDA_DEVBOX_POD_ID is not configured"

  printf 'Deleting RunPod pod %s...\n' "${pod_id}"
  runpodctl pod delete "${pod_id}" --output json
  clear_connection_state
  rm -f "${known_hosts}"
  printf 'Deleted pod %s and cleared its local connection state.\n' "${pod_id}"
}

ssh_arguments=()
quoted_remote_directory=

prepare_ssh() {
  [[ -n ${address} ]] || fail "CUDA_DEVBOX_ADDRESS is not configured; create a devbox first"
  [[ ${port} =~ ^[0-9]+$ && ${port} -ge 1 && ${port} -le 65535 ]] ||
    fail "CUDA_DEVBOX_PORT must be an integer in 1..65535"
  [[ ${address} != -* ]] || fail "CUDA_DEVBOX_ADDRESS cannot begin with '-'"
  [[ ${address} != *$'\n'* && ${remote_directory} != *$'\n'* ]] ||
    fail "connection values cannot contain newlines"
  printf -v quoted_remote_directory '%q' "${remote_directory}"

  ssh_arguments=(
    -p "${port}"
    -o "ConnectTimeout=15"
    -o "ServerAliveInterval=30"
    -o "StrictHostKeyChecking=accept-new"
    -o "UserKnownHostsFile=${known_hosts}"
  )
  if [[ -n ${identity_file} ]]; then
    [[ -f ${identity_file} ]] ||
      fail "identity file does not exist: ${identity_file}"
    ssh_arguments+=(-o "IdentitiesOnly=yes" -i "${identity_file}")
  fi
}

configure_managed_key() {
  local public_key

  ensure_managed_key
  register_managed_key
  if [[ -n ${address} ]]; then
    prepare_ssh
    public_key=$(<"${managed_public_key}")
    printf '%s\n' "${public_key}" |
      ssh "${ssh_arguments[@]}" "${address}" \
        'umask 077; mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"; touch "$HOME/.ssh/authorized_keys"; chmod 600 "$HOME/.ssh/authorized_keys"; IFS= read -r key; grep -qxF -- "$key" "$HOME/.ssh/authorized_keys" || printf "\n%s\n" "$key" >> "$HOME/.ssh/authorized_keys"'
  fi

  identity_file=${managed_identity_file}
  if [[ -n ${address} ]]; then
    prepare_ssh
    ssh "${ssh_arguments[@]}" -o "BatchMode=yes" "${address}" true
  fi
  write_identity_state "${identity_file}"
  printf 'CUDA devbox identity: %s\n' "${identity_file}"
}

run_remote() {
  local remote_command
  printf -v remote_command '%q ' "$@"
  # The repository and command are quoted before crossing the SSH boundary.
  # shellcheck disable=SC2029
  ssh "${ssh_arguments[@]}" "${address}" \
    "cd ${quoted_remote_directory} && ${remote_command}"
}

sync_worktree() {
  local deleted_files=() file quoted_deleted_files

  printf 'Uploading worktree to %s:%s...\n' "${address}" "${remote_directory}"
  while IFS= read -r -d '' file; do
    deleted_files+=("${file}")
  done < <(git -C "${repository}" ls-files --deleted -z)
  if [[ ${#deleted_files[@]} -gt 0 ]]; then
    printf -v quoted_deleted_files '%q ' "${deleted_files[@]}"
    # Values are quoted before crossing the SSH boundary.
    # shellcheck disable=SC2029
    ssh "${ssh_arguments[@]}" "${address}" \
      "mkdir -p ${quoted_remote_directory} && cd ${quoted_remote_directory} && rm -f -- ${quoted_deleted_files}"
  fi
  # The repository is quoted before crossing the SSH boundary.
  # shellcheck disable=SC2029
  git -C "${repository}" ls-files --cached --others --exclude-standard -z |
    while IFS= read -r -d '' file; do
      if [[ -e ${repository}/${file} || -L ${repository}/${file} ]]; then
        printf '%s\0' "${file}"
      fi
    done |
    tar --null -czf - -C "${repository}" -T - |
    ssh "${ssh_arguments[@]}" "${address}" \
      "mkdir -p ${quoted_remote_directory} && tar -xzf - -C ${quoted_remote_directory}"
}

case ${command} in
  key)
    [[ $# -eq 1 ]] || fail "key does not accept arguments"
    configure_managed_key
    ;;
  template)
    [[ $# -eq 1 ]] || fail "template does not accept arguments"
    manage_template
    ;;
  create)
    [[ $# -eq 1 ]] || fail "create does not accept arguments"
    create_devbox
    ;;
  destroy)
    [[ $# -eq 1 ]] || fail "destroy does not accept arguments"
    destroy_devbox
    ;;
  show)
    printf 'pod_id=%s\ntemplate_id=%s\ntemplate_image=%s\ntemplate_digest=%s\nregistry_auth_id=%s\ndata_center_ids=%s\naddress=%s\nport=%s\ndirectory=%s\nidentity_file=%s\nknown_hosts=%s\n' \
      "${pod_id}" "${template_id}" "${template_image}" "${template_image_digest}" "${registry_auth_id}" \
      "${data_center_ids}" "${address}" "${port}" "${remote_directory}" "${identity_file}" "${known_hosts}"
    ;;
  check)
    prepare_ssh
    ssh "${ssh_arguments[@]}" "${address}" nvidia-smi -L
    ;;
  sync)
    prepare_ssh
    sync_worktree
    ;;
  bootstrap)
    prepare_ssh
    sync_worktree
    run_remote ./scripts/bootstrap-cuda-devbox.sh
    ;;
  ssh)
    prepare_ssh
    exec ssh "${ssh_arguments[@]}" "${address}"
    ;;
  run)
    shift
    [[ $# -gt 0 ]] || fail "run requires a command"
    prepare_ssh
    run_remote nix develop .#cuda --command "$@"
    ;;
esac
