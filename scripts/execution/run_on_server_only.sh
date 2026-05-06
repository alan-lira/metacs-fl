#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Run one server-only MetaCS-FL experiment on the server node.
# ----------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  run_on_server_only.sh --nodes-file FILE --script FILE [options]

Required:
  --nodes-file FILE
      Shared node-description file.

  --script FILE
      Python script to run on the server node, relative to the project root
      unless an absolute path is provided.

Options:
  --config-file FILE
      Optional config file passed as:
        --config-file <FILE>

  --remote-project-dir DIR
      Project directory on the target/server machine.
      Default: /root/metacs-fl

  --remote-venv-activate FILE
      Virtual environment activation script on target/server machine.
      Default: <remote-project-dir>/.venv/bin/activate

  --python-bin BIN
      Python executable.
      Default: python3

  --run-id ID
      Run identifier.
      Default: current timestamp.

  --local-log-root DIR
      Local log root.
      Default: server_only_logs

  --remote-output-dir DIR
      Optional output directory on the server node to collect after execution.
      This should be relative to --remote-project-dir unless absolute.

  --local-output-root DIR
      Local root where server-only outputs are collected.
      Final output is <local-output-root>/<run-id>/.
      Default: server_only_results

  --clean-local-output true|false
      Whether to remove local collected output before rsync/copy.
      Default: true

  --ssh-options "OPTIONS"
      Extra options passed to ssh/scp/rsync.
      Default: empty

  --rsync-options "OPTIONS"
      Extra options passed to rsync.
      Default: -az

  --extra-arg ARG
      Extra argument passed to the Python script.
      Can be repeated. Use one token per --extra-arg.

  -h, --help
      Show this help message.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

warn() {
  echo "WARNING: $*" >&2
}

require_value() {
  local opt="$1"
  local val="${2:-}"

  if [[ -z "${val}" || "${val}" == --* ]]; then
    die "${opt} requires a value"
  fi
}

trim() {
  local s="$1"

  s="${s//$'\r'/}"
  s="${s//$'\xef\xbb\xbf'/}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"

  printf '%s' "${s}"
}

sanitize_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

shell_join_array_for_remote() {
  local item

  for item in "$@"; do
    printf ' %q' "$item"
  done
}

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

NODES_FILE=""
SCRIPT_PATH=""
CONFIG_FILE=""

REMOTE_PROJECT_DIR="/root/metacs-fl"
REMOTE_VENV_ACTIVATE=""
PYTHON_BIN="python3"

RUN_ID=""

LOCAL_LOG_ROOT="server_only_logs"
REMOTE_OUTPUT_DIR=""
LOCAL_OUTPUT_ROOT="server_only_results"
CLEAN_LOCAL_OUTPUT="true"

SSH_OPTIONS=""
RSYNC_OPTIONS="-az"

EXTRA_ARGS=()

# ----------------------------------------------------------------------
# Parse args
# ----------------------------------------------------------------------

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --nodes-file)
      require_value "$1" "${2:-}"
      NODES_FILE="$2"
      shift 2
      ;;

    --script)
      require_value "$1" "${2:-}"
      SCRIPT_PATH="$2"
      shift 2
      ;;

    --config-file)
      require_value "$1" "${2:-}"
      CONFIG_FILE="$2"
      shift 2
      ;;

    --remote-project-dir)
      require_value "$1" "${2:-}"
      REMOTE_PROJECT_DIR="$2"
      shift 2
      ;;

    --remote-venv-activate)
      require_value "$1" "${2:-}"
      REMOTE_VENV_ACTIVATE="$2"
      shift 2
      ;;

    --python-bin)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;

    --run-id)
      require_value "$1" "${2:-}"
      RUN_ID="$2"
      shift 2
      ;;

    --local-log-root)
      require_value "$1" "${2:-}"
      LOCAL_LOG_ROOT="$2"
      shift 2
      ;;

    --remote-output-dir)
      require_value "$1" "${2:-}"
      REMOTE_OUTPUT_DIR="$2"
      shift 2
      ;;

    --local-output-root)
      require_value "$1" "${2:-}"
      LOCAL_OUTPUT_ROOT="$2"
      shift 2
      ;;

    --clean-local-output)
      require_value "$1" "${2:-}"
      CLEAN_LOCAL_OUTPUT="$2"
      shift 2
      ;;

    --ssh-options)
      require_value "$1" "${2:-}"
      SSH_OPTIONS="$2"
      shift 2
      ;;

    --rsync-options)
      require_value "$1" "${2:-}"
      RSYNC_OPTIONS="$2"
      shift 2
      ;;

    --extra-arg)
      require_value "$1" "${2:-}"
      EXTRA_ARGS+=("$2")
      shift 2
      ;;

    -h|--help)
      usage
      exit 0
      ;;

    *)
      echo "ERROR: unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

# ----------------------------------------------------------------------
# Validate
# ----------------------------------------------------------------------

[[ -n "${NODES_FILE}" ]] || die "--nodes-file is required"
[[ -f "${NODES_FILE}" ]] || die "nodes file not found: ${NODES_FILE}"
[[ -n "${SCRIPT_PATH}" ]] || die "--script is required"

if [[ "${CLEAN_LOCAL_OUTPUT}" != "true" && "${CLEAN_LOCAL_OUTPUT}" != "false" ]]; then
  die "--clean-local-output must be true or false"
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "${REMOTE_VENV_ACTIVATE}" ]]; then
  REMOTE_VENV_ACTIVATE="${REMOTE_PROJECT_DIR}/.venv/bin/activate"
fi

# ----------------------------------------------------------------------
# Read node file and find server target
# ----------------------------------------------------------------------

MODE=""
SERVER_SSH_TARGET=""
SERVER_RUNTIME_HOST=""

LINE_NO=0

while IFS= read -r RAW_LINE || [[ -n "${RAW_LINE}" ]]; do
  LINE_NO=$((LINE_NO + 1))

  LINE="${RAW_LINE%%#*}"
  LINE="$(trim "${LINE}")"

  [[ -z "${LINE}" ]] && continue

  read -r FIELD1 FIELD2 FIELD3 EXTRA <<< "${LINE}"

  FIELD1="$(trim "${FIELD1:-}")"
  FIELD2="$(trim "${FIELD2:-}")"
  FIELD3="$(trim "${FIELD3:-}")"
  EXTRA="$(trim "${EXTRA:-}")"

  if [[ "${FIELD1}" == "mode" ]]; then
    [[ -n "${FIELD2}" ]] || die "${NODES_FILE}:${LINE_NO}: invalid mode line"
    MODE="${FIELD2}"
    continue
  fi

  if [[ "${FIELD1}" == "server" ]]; then
    if [[ -n "${SERVER_SSH_TARGET}" ]]; then
      die "expected exactly one server entry, found multiple"
    fi

    SERVER_SSH_TARGET="${FIELD2}"
    SERVER_RUNTIME_HOST="${FIELD3}"
  fi
done < "${NODES_FILE}"

[[ -n "${MODE}" ]] || die "nodes file must contain a mode line"
[[ -n "${SERVER_SSH_TARGET}" ]] || die "nodes file must contain one server entry"

case "${MODE}" in
  local|remote|g5k)
    ;;
  *)
    die "unsupported mode: ${MODE}"
    ;;
esac

if [[ "${MODE}" == "local" && "${SERVER_SSH_TARGET}" != "-" ]]; then
  die "local mode expects '-' as server ssh target"
fi

if [[ "${MODE}" != "local" && "${SERVER_SSH_TARGET}" == "-" ]]; then
  die "${MODE} mode requires a real server ssh target"
fi

# ----------------------------------------------------------------------
# Prepare command
# ----------------------------------------------------------------------

PY_ARGS=()

if [[ -n "${CONFIG_FILE}" ]]; then
  PY_ARGS+=(--config-file "${CONFIG_FILE}")
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  PY_ARGS+=("${EXTRA_ARGS[@]}")
fi

LOCAL_LOG_DIR="${LOCAL_LOG_ROOT}/${RUN_ID}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_ROOT}/${RUN_ID}"

mkdir -p "${LOCAL_LOG_DIR}"

STDOUT_LOG="${LOCAL_LOG_DIR}/server_only.out"
STDERR_LOG="${LOCAL_LOG_DIR}/server_only.err"
MANIFEST="${LOCAL_LOG_DIR}/server_only_manifest.txt"

{
  echo "run_id=${RUN_ID}"
  echo "mode=${MODE}"
  echo "nodes_file=${NODES_FILE}"
  echo "server_ssh_target=${SERVER_SSH_TARGET}"
  echo "server_runtime_host=${SERVER_RUNTIME_HOST}"
  echo "remote_project_dir=${REMOTE_PROJECT_DIR}"
  echo "remote_venv_activate=${REMOTE_VENV_ACTIVATE}"
  echo "python_bin=${PYTHON_BIN}"
  echo "script=${SCRIPT_PATH}"
  echo "config_file=${CONFIG_FILE}"
  echo "remote_output_dir=${REMOTE_OUTPUT_DIR}"
  echo "local_output_dir=${LOCAL_OUTPUT_DIR}"
  echo "extra_args=$(printf '%q ' "${EXTRA_ARGS[@]}")"
} > "${MANIFEST}"

echo "[SERVER_ONLY] Run ID: ${RUN_ID}"
echo "[SERVER_ONLY] Mode: ${MODE}"
echo "[SERVER_ONLY] Server target: ${SERVER_SSH_TARGET}"
echo "[SERVER_ONLY] Script: ${SCRIPT_PATH}"
echo "[SERVER_ONLY] Config: ${CONFIG_FILE:-<none>}"
echo "[SERVER_ONLY] Logs: ${LOCAL_LOG_DIR}"

# ----------------------------------------------------------------------
# Execute
# ----------------------------------------------------------------------

if [[ "${MODE}" == "local" ]]; then
  [[ -d "${REMOTE_PROJECT_DIR}" ]] || die "project directory not found: ${REMOTE_PROJECT_DIR}"
  [[ -f "${REMOTE_VENV_ACTIVATE}" ]] || die "venv activate file not found: ${REMOTE_VENV_ACTIVATE}"

  (
    set -euo pipefail

    cd "${REMOTE_PROJECT_DIR}"
    source "${REMOTE_VENV_ACTIVATE}"

    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${SCRIPT_PATH}" "${PY_ARGS[@]}"
  ) > "${STDOUT_LOG}" 2> "${STDERR_LOG}"

else
  REMOTE_PY_ARGS="$(shell_join_array_for_remote "${PY_ARGS[@]}")"

  ssh ${SSH_OPTIONS} "${SERVER_SSH_TARGET}" "
    set -euo pipefail

    cd '${REMOTE_PROJECT_DIR}'
    source '${REMOTE_VENV_ACTIVATE}'

    PYTHONUNBUFFERED=1 '${PYTHON_BIN}' -u '${SCRIPT_PATH}' ${REMOTE_PY_ARGS}
  " > "${STDOUT_LOG}" 2> "${STDERR_LOG}"
fi

# ----------------------------------------------------------------------
# Collect output directory, if requested
# ----------------------------------------------------------------------

if [[ -n "${REMOTE_OUTPUT_DIR}" ]]; then
  if [[ "${CLEAN_LOCAL_OUTPUT}" == "true" && -e "${LOCAL_OUTPUT_DIR}" ]]; then
    rm -rf "${LOCAL_OUTPUT_DIR}"
  fi

  mkdir -p "${LOCAL_OUTPUT_DIR}"

  if [[ "${MODE}" == "local" ]]; then
    if [[ "${REMOTE_OUTPUT_DIR}" = /* ]]; then
      SRC="${REMOTE_OUTPUT_DIR}"
    else
      SRC="${REMOTE_PROJECT_DIR}/${REMOTE_OUTPUT_DIR}"
    fi

    if [[ -d "${SRC}" ]]; then
      rsync ${RSYNC_OPTIONS} "${SRC}/" "${LOCAL_OUTPUT_DIR}/"
    else
      warn "local output directory not found: ${SRC}"
    fi
  else
    if [[ "${REMOTE_OUTPUT_DIR}" = /* ]]; then
      REMOTE_SRC="${REMOTE_OUTPUT_DIR}"
    else
      REMOTE_SRC="${REMOTE_PROJECT_DIR}/${REMOTE_OUTPUT_DIR}"
    fi

    if ssh ${SSH_OPTIONS} "${SERVER_SSH_TARGET}" "test -d '${REMOTE_SRC}'"; then
      rsync ${RSYNC_OPTIONS} "${SERVER_SSH_TARGET}:${REMOTE_SRC}/" "${LOCAL_OUTPUT_DIR}/"
    else
      warn "remote output directory not found: ${SERVER_SSH_TARGET}:${REMOTE_SRC}"
    fi
  fi
fi

echo "[SERVER_ONLY] Completed successfully."
echo "[SERVER_ONLY] stdout: ${STDOUT_LOG}"
echo "[SERVER_ONLY] stderr: ${STDERR_LOG}"

if [[ -n "${REMOTE_OUTPUT_DIR}" ]]; then
  echo "[SERVER_ONLY] collected output: ${LOCAL_OUTPUT_DIR}"
fi

exit 0
