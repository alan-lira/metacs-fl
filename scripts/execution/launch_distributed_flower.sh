#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  launch_distributed_flower.sh --nodes-file FILE [options]

Required:
  --nodes-file FILE
      Node description file.

Options:
  --remote-project-dir DIR
      Project directory as seen by the machine running the FL process.
      For local mode: local project path.
      For remote/g5k mode: project path on each remote node.
      Default: /root/metacs-fl

  --remote-venv-activate FILE
      Virtualenv activation script as seen by the machine running the FL process.
      Default: <remote-project-dir>/.venv/bin/activate

  --remote-flower-executor-cfg FILE
      Flower executor config file already present on the machine running the FL process.
      This is used when --custom-flower-executor-cfg is not provided.
      Default: <remote-project-dir>/metacs_fl/flower_executor/config/flower_executor.cfg

  --remote-config-file FILE
      Backward-compatible alias for --remote-flower-executor-cfg.

  --remote-flower-server-cfg FILE
      Flower server config file already present on the machine running the FL process.
      Optional. Used only for patching the runtime executor config.

  --remote-flower-client-cfg FILE
      Flower client config file already present on the machine running the FL process.
      Optional. Used only for patching the runtime executor config.

  --custom-flower-executor-cfg FILE
      Custom Flower executor config file to use for this run.
      The file is read from the launcher machine.
      In local mode, it is copied to a local runtime config directory.
      In remote/g5k mode, it is copied to the remote runtime config directory.

  --custom-flower-server-cfg FILE
      Custom Flower server config file to use for this run.
      The file is read from the launcher machine.
      In remote/g5k mode, it is copied to the remote runtime config directory.

  --custom-flower-client-cfg FILE
      Custom Flower client config file to use for this run.
      The file is read from the launcher machine.
      In remote/g5k mode, it is copied to the remote runtime config directory.

  --remote-runtime-config-dir DIR
      Directory where per-run config files are copied on remote nodes.
      Default: <remote-project-dir>/.distributed_runtime_configs/<run-id>

  --patch-executor-config true|false
      Whether to patch the runtime flower_executor.cfg so it points to the chosen
      flower_server.cfg and flower_client.cfg paths.
      Default: true

  --patch-grpc-config true|false
      Whether to patch runtime flower_server.cfg and flower_client.cfg gRPC settings.
      The client config server_ip_address is set to the server runtime host.
      The server config listen_ip_address is set to 0.0.0.0 in remote/g5k mode.
      Default: true

  --server-port PORT
      Fallback server port used when a runtime flower_server.cfg is unavailable
      or does not contain [gRPC Settings] listen_port.
      Default: 8080

  --remote-hostfile-dir DIR
      Directory where the runtime hostfile is copied on remote nodes.
      In local mode, this is not used for copying.
      Default: <remote-project-dir>

  --action ACTION
      main.py action.
      Default: execute_fl_with_flower

  --repetitions N
      Number of repetitions.
      Default: 1

  --python-bin BIN
      Python executable to use on the machine running the FL process.
      Default: python3

  --local-log-root DIR
      Root directory for local logs.
      Default: distributed_launch_logs

  --local-gather-root DIR
      Root directory for gathered results.
      Default: gathered_results

  --gather-outputs true|false
      Value passed to main.py --gather-outputs.
      Default: false

  --max-parallel-remote-ops N
      Maximum number of remote validation/copy/collection operations in parallel.
      Default: 8

  --ssh-options "OPTIONS"
      Extra options passed to ssh and scp.
      Example: --ssh-options "-o StrictHostKeyChecking=no"
      Default: empty

  --rsync-options "OPTIONS"
      Extra options passed to rsync.
      Default: -az

  --run-id ID
      Optional explicit run id.
      Default: current timestamp

  -h, --help
      Show this help message.

Node file format:
  mode local
  server - 127.0.0.1
  client - 127.0.0.1

  mode remote
  server user@server.example.com server.example.com
  client user@client1.example.com client1.example.com

  mode g5k
  server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
  client root@paradoxe-2.rennes.g5k paradoxe-2.rennes.grid5000.fr

Examples from the project root:
  bash scripts/execution/launch_distributed_flower.sh \
    --nodes-file scripts/nodes.local.txt \
    --remote-project-dir "$PWD"

  bash scripts/execution/launch_distributed_flower.sh \
    --nodes-file scripts/nodes.g5k.txt \
    --remote-project-dir /root/metacs-fl \
    --repetitions 3

  bash scripts/execution/launch_distributed_flower.sh \
    --nodes-file scripts/nodes.g5k.txt \
    --remote-project-dir /root/metacs-fl \
    --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
    --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
    --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
    --max-parallel-remote-ops 8
EOF
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

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

  # Remove Windows carriage returns.
  s="${s//$'\r'/}"

  # Remove UTF-8 BOM if present.
  s="${s//$'\xef\xbb\xbf'/}"

  # Trim leading whitespace.
  s="${s#"${s%%[![:space:]]*}"}"

  # Trim trailing whitespace.
  s="${s%"${s##*[![:space:]]}"}"

  printf '%s' "${s}"
}

sanitize_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

array_contains() {
  local needle="$1"
  shift

  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done

  return 1
}

wait_for_available_slot() {
  local max_jobs="$1"

  while true; do
    local running
    running="$(jobs -pr | wc -l)"

    if [[ "${running}" -lt "${max_jobs}" ]]; then
      break
    fi

    sleep 1
  done
}

run_parallel_stage() {
  local stage_name="$1"
  local max_jobs="$2"
  shift 2

  local targets=("$@")

  local pids=()
  declare -A pid_to_target=()
  declare -A pid_to_log=()

  local failed=0

  echo
  echo "[LOCAL] Starting parallel stage: ${stage_name}"
  echo "[LOCAL] Max parallel operations: ${max_jobs}"

  for target in "${targets[@]}"; do
    wait_for_available_slot "${max_jobs}"

    local safe_target
    safe_target="$(sanitize_name "${target}")"

    local stage_log="${LOCAL_LOG_DIR}/${stage_name}_${safe_target}.log"

    echo "[LOCAL] ${stage_name}: ${target}"
    echo "[LOCAL] Log: ${stage_log}"

    (
      set -euo pipefail

      case "${stage_name}" in
        verify_remote)
          verify_remote_one "${target}"
          ;;
        copy_runtime_files)
          copy_runtime_files_one "${target}"
          ;;
        collect_results)
          collect_results_one "${target}"
          ;;
        *)
          echo "Unknown parallel stage: ${stage_name}" >&2
          exit 1
          ;;
      esac
    ) > "${stage_log}" 2>&1 &

    local pid="$!"
    pids+=("${pid}")
    pid_to_target["${pid}"]="${target}"
    pid_to_log["${pid}"]="${stage_log}"
  done

  for pid in "${pids[@]}"; do
    local target="${pid_to_target[${pid}]}"
    local stage_log="${pid_to_log[${pid}]}"

    if wait "${pid}"; then
      echo "[LOCAL] ${stage_name} succeeded on ${target}."
    else
      echo "[LOCAL] ERROR: ${stage_name} failed on ${target}."
      echo "[LOCAL] Check log: ${stage_log}"
      failed=1
    fi
  done

  if [[ "${failed}" -ne 0 ]]; then
    return 1
  fi

  return 0
}

# Patch a flower_executor.cfg copy so that it points to the selected
# flower_server.cfg and flower_client.cfg files when possible.
patch_executor_config_file() {
  local EXECUTOR_CFG="$1"
  local SERVER_CFG_PATH="$2"
  local CLIENT_CFG_PATH="$3"

  python3 - "${EXECUTOR_CFG}" "${SERVER_CFG_PATH}" "${CLIENT_CFG_PATH}" <<'PY'
from __future__ import annotations

import sys
from configparser import ConfigParser
from pathlib import Path

executor_cfg = Path(sys.argv[1])
server_cfg_path = sys.argv[2].strip()
client_cfg_path = sys.argv[3].strip()

parser = ConfigParser(interpolation=None)
parser.optionxform = str
parser.read(executor_cfg)

server_option_names = {
    "flower_server_cfg",
    "flower_server_config",
    "flower_server_config_file",
    "server_cfg",
    "server_config",
    "server_config_file",
    "flower_server_settings_file",
    "base_server_config_file",
}

client_option_names = {
    "flower_client_cfg",
    "flower_client_config",
    "flower_client_config_file",
    "client_cfg",
    "client_config",
    "client_config_file",
    "flower_client_settings_file",
    "base_client_config_file",
}

changed = False

for section in parser.sections():
    for option, value in list(parser.items(section)):
        normalized_option = option.strip().lower()
        normalized_value = value.strip().replace("\\", "/")
        value_name = Path(normalized_value).name.lower()

        if server_cfg_path and (
            normalized_option in server_option_names
            or value_name == "flower_server.cfg"
        ):
            parser.set(section, option, server_cfg_path)
            changed = True

        if client_cfg_path and (
            normalized_option in client_option_names
            or value_name == "flower_client.cfg"
        ):
            parser.set(section, option, client_cfg_path)
            changed = True

with executor_cfg.open("w", encoding="utf-8") as f:
    parser.write(f)

print("patched=true" if changed else "patched=false")
PY
}

patch_grpc_config_files() {
  local SERVER_CFG="$1"
  local CLIENT_CFG="$2"
  local SERVER_RUNTIME_HOST="$3"
  local SERVER_PORT="$4"
  local MODE_VALUE="$5"

  python3 - "${SERVER_CFG}" "${CLIENT_CFG}" "${SERVER_RUNTIME_HOST}" "${SERVER_PORT}" "${MODE_VALUE}" <<'PY'
from __future__ import annotations

import sys
from configparser import ConfigParser
from pathlib import Path

server_cfg_arg = sys.argv[1].strip()
client_cfg_arg = sys.argv[2].strip()
server_runtime_host = sys.argv[3].strip()
server_port = sys.argv[4].strip()
mode = sys.argv[5].strip()

server_cfg = Path(server_cfg_arg) if server_cfg_arg else None
client_cfg = Path(client_cfg_arg) if client_cfg_arg else None

def read_cfg(path: Path) -> ConfigParser:
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(path)
    return parser

def write_cfg(parser: ConfigParser, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        parser.write(f)

def ensure_section(parser: ConfigParser, section: str) -> None:
    if not parser.has_section(section):
        parser.add_section(section)

patched_server = False
patched_client = False

if server_cfg and server_cfg.exists():
    parser = read_cfg(server_cfg)
    ensure_section(parser, "gRPC Settings")

    if mode in {"remote", "g5k"}:
        parser.set("gRPC Settings", "listen_ip_address", "0.0.0.0")
    elif not parser.has_option("gRPC Settings", "listen_ip_address"):
        parser.set("gRPC Settings", "listen_ip_address", "127.0.0.1")

    if server_port:
        parser.set("gRPC Settings", "listen_port", server_port)

    write_cfg(parser, server_cfg)
    patched_server = True

if client_cfg and client_cfg.exists():
    parser = read_cfg(client_cfg)
    ensure_section(parser, "gRPC Settings")

    parser.set("gRPC Settings", "server_ip_address", server_runtime_host)

    if server_port:
        parser.set("gRPC Settings", "server_port", server_port)

    write_cfg(parser, client_cfg)
    patched_client = True

print(
    "grpc_patched=true "
    f"server_cfg={patched_server} "
    f"client_cfg={patched_client} "
    f"server_runtime_host={server_runtime_host} "
    f"server_port={server_port}"
)
PY
}

read_server_port_from_config() {
  local SERVER_CFG="$1"
  local DEFAULT_PORT="$2"

  python3 - "${SERVER_CFG}" "${DEFAULT_PORT}" <<'PY'
from configparser import ConfigParser
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
default_port = sys.argv[2]

parser = ConfigParser(interpolation=None)
parser.read(cfg)

if parser.has_option("gRPC Settings", "listen_port"):
    print(parser.get("gRPC Settings", "listen_port").strip())
else:
    print(default_port)
PY
}

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

NODES_FILE=""

REMOTE_PROJECT_DIR="/root/metacs-fl"
REMOTE_VENV_ACTIVATE=""

REMOTE_FLOWER_EXECUTOR_CFG=""
REMOTE_FLOWER_SERVER_CFG=""
REMOTE_FLOWER_CLIENT_CFG=""

CUSTOM_FLOWER_EXECUTOR_CFG=""
CUSTOM_FLOWER_SERVER_CFG=""
CUSTOM_FLOWER_CLIENT_CFG=""

REMOTE_RUNTIME_CONFIG_DIR=""
PATCH_EXECUTOR_CONFIG="true"
PATCH_GRPC_CONFIG="true"
SERVER_PORT="8080"

REMOTE_HOSTFILE_DIR=""

ACTION="execute_fl_with_flower"
REPETITIONS="1"
PYTHON_BIN="python3"

LOCAL_LOG_ROOT="distributed_launch_logs"
LOCAL_GATHER_ROOT="gathered_results"

GATHER_OUTPUTS="false"

MAX_PARALLEL_REMOTE_OPS="8"
SSH_OPTIONS=""
RSYNC_OPTIONS="-az"

RUN_ID=""

# ----------------------------------------------------------------------
# Parse arguments
# ----------------------------------------------------------------------

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --nodes-file)
      require_value "$1" "${2:-}"
      NODES_FILE="$2"
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

    --remote-flower-executor-cfg|--remote-config-file)
      require_value "$1" "${2:-}"
      REMOTE_FLOWER_EXECUTOR_CFG="$2"
      shift 2
      ;;

    --remote-flower-server-cfg)
      require_value "$1" "${2:-}"
      REMOTE_FLOWER_SERVER_CFG="$2"
      shift 2
      ;;

    --remote-flower-client-cfg)
      require_value "$1" "${2:-}"
      REMOTE_FLOWER_CLIENT_CFG="$2"
      shift 2
      ;;

    --custom-flower-executor-cfg)
      require_value "$1" "${2:-}"
      CUSTOM_FLOWER_EXECUTOR_CFG="$2"
      shift 2
      ;;

    --custom-flower-server-cfg)
      require_value "$1" "${2:-}"
      CUSTOM_FLOWER_SERVER_CFG="$2"
      shift 2
      ;;

    --custom-flower-client-cfg)
      require_value "$1" "${2:-}"
      CUSTOM_FLOWER_CLIENT_CFG="$2"
      shift 2
      ;;

    --remote-runtime-config-dir)
      require_value "$1" "${2:-}"
      REMOTE_RUNTIME_CONFIG_DIR="$2"
      shift 2
      ;;

    --patch-executor-config)
      require_value "$1" "${2:-}"
      PATCH_EXECUTOR_CONFIG="$2"
      shift 2
      ;;

    --patch-grpc-config)
      require_value "$1" "${2:-}"
      PATCH_GRPC_CONFIG="$2"
      shift 2
      ;;

    --server-port)
      require_value "$1" "${2:-}"
      SERVER_PORT="$2"
      shift 2
      ;;

    --remote-hostfile-dir)
      require_value "$1" "${2:-}"
      REMOTE_HOSTFILE_DIR="$2"
      shift 2
      ;;

    --action)
      require_value "$1" "${2:-}"
      ACTION="$2"
      shift 2
      ;;

    --repetitions)
      require_value "$1" "${2:-}"
      REPETITIONS="$2"
      shift 2
      ;;

    --python-bin)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;

    --local-log-root)
      require_value "$1" "${2:-}"
      LOCAL_LOG_ROOT="$2"
      shift 2
      ;;

    --local-gather-root)
      require_value "$1" "${2:-}"
      LOCAL_GATHER_ROOT="$2"
      shift 2
      ;;

    --gather-outputs)
      require_value "$1" "${2:-}"
      GATHER_OUTPUTS="$2"
      shift 2
      ;;

    --max-parallel-remote-ops)
      require_value "$1" "${2:-}"
      MAX_PARALLEL_REMOTE_OPS="$2"
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

    --run-id)
      require_value "$1" "${2:-}"
      RUN_ID="$2"
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
# Validate basic parameters
# ----------------------------------------------------------------------

[[ -n "${NODES_FILE}" ]] || {
  echo "ERROR: --nodes-file is required" >&2
  echo >&2
  usage >&2
  exit 1
}

[[ -f "${NODES_FILE}" ]] || die "nodes file not found: ${NODES_FILE}"
[[ -n "${REMOTE_PROJECT_DIR}" ]] || die "--remote-project-dir cannot be empty"

if [[ -z "${REMOTE_VENV_ACTIVATE}" ]]; then
  REMOTE_VENV_ACTIVATE="${REMOTE_PROJECT_DIR}/.venv/bin/activate"
fi

if [[ -z "${REMOTE_FLOWER_EXECUTOR_CFG}" ]]; then
  REMOTE_FLOWER_EXECUTOR_CFG="${REMOTE_PROJECT_DIR}/metacs_fl/flower_executor/config/flower_executor.cfg"
fi

if [[ -z "${REMOTE_HOSTFILE_DIR}" ]]; then
  REMOTE_HOSTFILE_DIR="${REMOTE_PROJECT_DIR}"
fi

if ! [[ "${REPETITIONS}" =~ ^[0-9]+$ ]]; then
  die "--repetitions must be a positive integer"
fi

if [[ "${REPETITIONS}" -lt 1 ]]; then
  die "--repetitions must be >= 1"
fi

if [[ "${GATHER_OUTPUTS}" != "true" && "${GATHER_OUTPUTS}" != "false" ]]; then
  die "--gather-outputs must be true or false"
fi

if [[ "${PATCH_EXECUTOR_CONFIG}" != "true" && "${PATCH_EXECUTOR_CONFIG}" != "false" ]]; then
  die "--patch-executor-config must be true or false"
fi

if [[ "${PATCH_GRPC_CONFIG}" != "true" && "${PATCH_GRPC_CONFIG}" != "false" ]]; then
  die "--patch-grpc-config must be true or false"
fi

if ! [[ "${MAX_PARALLEL_REMOTE_OPS}" =~ ^[0-9]+$ ]]; then
  die "--max-parallel-remote-ops must be a positive integer"
fi

if [[ "${MAX_PARALLEL_REMOTE_OPS}" -lt 1 ]]; then
  die "--max-parallel-remote-ops must be >= 1"
fi

for CUSTOM_CFG in \
  "${CUSTOM_FLOWER_EXECUTOR_CFG}" \
  "${CUSTOM_FLOWER_SERVER_CFG}" \
  "${CUSTOM_FLOWER_CLIENT_CFG}"
do
  if [[ -n "${CUSTOM_CFG}" && ! -f "${CUSTOM_CFG}" ]]; then
    die "custom config file not found: ${CUSTOM_CFG}"
  fi
done

# ----------------------------------------------------------------------
# Read node file
# ----------------------------------------------------------------------

MODE=""

ROLES=()
SSH_TARGETS=()
RUNTIME_HOSTS=()

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
    [[ -n "${FIELD2}" ]] || die "${NODES_FILE}:${LINE_NO}: mode line requires one value: local, remote, or g5k"
    [[ -z "${FIELD3}" && -z "${EXTRA}" ]] || die "${NODES_FILE}:${LINE_NO}: mode line has too many fields: ${LINE}"

    if [[ -n "${MODE}" ]]; then
      die "${NODES_FILE}:${LINE_NO}: duplicate mode line. Previous mode was '${MODE}'"
    fi

    MODE="${FIELD2}"
    continue
  fi

  ROLE="${FIELD1}"
  SSH_TARGET="${FIELD2}"
  RUNTIME_HOST="${FIELD3}"

  if [[ -n "${EXTRA}" ]]; then
    die "${NODES_FILE}:${LINE_NO}: invalid line with too many fields: ${LINE}"
  fi

  if [[ "${ROLE}" != "server" && "${ROLE}" != "client" ]]; then
    die "${NODES_FILE}:${LINE_NO}: invalid role '${ROLE}'. Expected 'server' or 'client'"
  fi

  if [[ -z "${SSH_TARGET}" || -z "${RUNTIME_HOST}" ]]; then
    die "${NODES_FILE}:${LINE_NO}: invalid node line. Expected: <server|client> <ssh_target|-> <runtime_host>"
  fi

  ROLES+=("${ROLE}")
  SSH_TARGETS+=("${SSH_TARGET}")
  RUNTIME_HOSTS+=("${RUNTIME_HOST}")
done < "${NODES_FILE}"

if [[ -z "${MODE}" ]]; then
  die "nodes file must contain exactly one mode line: mode local | mode remote | mode g5k"
fi

case "${MODE}" in
  local|remote|g5k)
    ;;
  *)
    printf "ERROR: unsupported mode: %q\n" "${MODE}" >&2
    echo "Valid modes: local, remote, g5k" >&2
    echo >&2
    echo "Hint: if the mode looks correct, check for hidden characters with:" >&2
    echo "  cat -A ${NODES_FILE}" >&2
    exit 1
    ;;
esac

if [[ "${#ROLES[@]}" -eq 0 ]]; then
  die "no nodes found in nodes file: ${NODES_FILE}"
fi

SERVER_COUNT=0
CLIENT_COUNT=0

for ROLE in "${ROLES[@]}"; do
  if [[ "${ROLE}" == "server" ]]; then
    SERVER_COUNT=$((SERVER_COUNT + 1))
  elif [[ "${ROLE}" == "client" ]]; then
    CLIENT_COUNT=$((CLIENT_COUNT + 1))
  fi
done

if [[ "${SERVER_COUNT}" -ne 1 ]]; then
  die "expected exactly one server entry, found ${SERVER_COUNT}"
fi

if [[ "${CLIENT_COUNT}" -lt 1 ]]; then
  die "expected at least one client entry"
fi

if [[ "${MODE}" == "local" ]]; then
  for SSH_TARGET in "${SSH_TARGETS[@]}"; do
    if [[ "${SSH_TARGET}" != "-" ]]; then
      die "local mode expects '-' as ssh_target, found: ${SSH_TARGET}"
    fi
  done
fi

if [[ "${MODE}" == "remote" || "${MODE}" == "g5k" ]]; then
  for SSH_TARGET in "${SSH_TARGETS[@]}"; do
    if [[ "${SSH_TARGET}" == "-" ]]; then
      die "${MODE} mode requires real ssh_target values"
    fi
  done
fi

# ----------------------------------------------------------------------
# Unique remote targets
# ----------------------------------------------------------------------

UNIQUE_SSH_TARGETS=()

if [[ "${MODE}" != "local" ]]; then
  for SSH_TARGET in "${SSH_TARGETS[@]}"; do
    if ! array_contains "${SSH_TARGET}" "${UNIQUE_SSH_TARGETS[@]}"; then
      UNIQUE_SSH_TARGETS+=("${SSH_TARGET}")
    fi
  done
fi

# ----------------------------------------------------------------------
# Run metadata
# ----------------------------------------------------------------------

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

LOCAL_LOG_DIR="${LOCAL_LOG_ROOT}/${RUN_ID}"
LOCAL_GATHER_DIR="${LOCAL_GATHER_ROOT}/${RUN_ID}"
LOCAL_RUNTIME_CONFIG_DIR="${LOCAL_LOG_DIR}/runtime_configs"

mkdir -p "${LOCAL_LOG_DIR}"
mkdir -p "${LOCAL_GATHER_DIR}"
mkdir -p "${LOCAL_RUNTIME_CONFIG_DIR}"

LOCAL_HOSTFILE="${LOCAL_LOG_DIR}/nodes_runtime_${RUN_ID}.txt"
REMOTE_HOSTFILE="${REMOTE_HOSTFILE_DIR}/nodes_runtime_${RUN_ID}.txt"

if [[ -z "${REMOTE_RUNTIME_CONFIG_DIR}" ]]; then
  REMOTE_RUNTIME_CONFIG_DIR="${REMOTE_PROJECT_DIR}/.distributed_runtime_configs/${RUN_ID}"
fi

# ----------------------------------------------------------------------
# Prepare runtime config files
# ----------------------------------------------------------------------

LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG="${LOCAL_RUNTIME_CONFIG_DIR}/flower_executor.cfg"
LOCAL_RUNTIME_FLOWER_SERVER_CFG=""
LOCAL_RUNTIME_FLOWER_CLIENT_CFG=""

RUNTIME_FLOWER_EXECUTOR_CFG=""
RUNTIME_FLOWER_SERVER_CFG=""
RUNTIME_FLOWER_CLIENT_CFG=""

if [[ -n "${CUSTOM_FLOWER_SERVER_CFG}" ]]; then
  LOCAL_RUNTIME_FLOWER_SERVER_CFG="${LOCAL_RUNTIME_CONFIG_DIR}/flower_server.cfg"
  cp "${CUSTOM_FLOWER_SERVER_CFG}" "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}"
fi

if [[ -n "${CUSTOM_FLOWER_CLIENT_CFG}" ]]; then
  LOCAL_RUNTIME_FLOWER_CLIENT_CFG="${LOCAL_RUNTIME_CONFIG_DIR}/flower_client.cfg"
  cp "${CUSTOM_FLOWER_CLIENT_CFG}" "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}"
fi

if [[ -n "${CUSTOM_FLOWER_EXECUTOR_CFG}" ]]; then
  cp "${CUSTOM_FLOWER_EXECUTOR_CFG}" "${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}"

  if [[ "${MODE}" == "local" ]]; then
    RUNTIME_FLOWER_EXECUTOR_CFG="${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}"

    if [[ -n "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" ]]; then
      RUNTIME_FLOWER_SERVER_CFG="${LOCAL_RUNTIME_FLOWER_SERVER_CFG}"
    elif [[ -n "${REMOTE_FLOWER_SERVER_CFG}" ]]; then
      RUNTIME_FLOWER_SERVER_CFG="${REMOTE_FLOWER_SERVER_CFG}"
    fi

    if [[ -n "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" ]]; then
      RUNTIME_FLOWER_CLIENT_CFG="${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}"
    elif [[ -n "${REMOTE_FLOWER_CLIENT_CFG}" ]]; then
      RUNTIME_FLOWER_CLIENT_CFG="${REMOTE_FLOWER_CLIENT_CFG}"
    fi
  else
    RUNTIME_FLOWER_EXECUTOR_CFG="${REMOTE_RUNTIME_CONFIG_DIR}/flower_executor.cfg"

    if [[ -n "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" ]]; then
      RUNTIME_FLOWER_SERVER_CFG="${REMOTE_RUNTIME_CONFIG_DIR}/flower_server.cfg"
    elif [[ -n "${REMOTE_FLOWER_SERVER_CFG}" ]]; then
      RUNTIME_FLOWER_SERVER_CFG="${REMOTE_FLOWER_SERVER_CFG}"
    fi

    if [[ -n "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" ]]; then
      RUNTIME_FLOWER_CLIENT_CFG="${REMOTE_RUNTIME_CONFIG_DIR}/flower_client.cfg"
    elif [[ -n "${REMOTE_FLOWER_CLIENT_CFG}" ]]; then
      RUNTIME_FLOWER_CLIENT_CFG="${REMOTE_FLOWER_CLIENT_CFG}"
    fi
  fi
else
  if [[ "${MODE}" == "local" ]]; then
    cp "${REMOTE_FLOWER_EXECUTOR_CFG}" "${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}"
    RUNTIME_FLOWER_EXECUTOR_CFG="${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}"

    if [[ -n "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" ]]; then
      RUNTIME_FLOWER_SERVER_CFG="${LOCAL_RUNTIME_FLOWER_SERVER_CFG}"
    elif [[ -n "${REMOTE_FLOWER_SERVER_CFG}" ]]; then
      RUNTIME_FLOWER_SERVER_CFG="${REMOTE_FLOWER_SERVER_CFG}"
    fi

    if [[ -n "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" ]]; then
      RUNTIME_FLOWER_CLIENT_CFG="${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}"
    elif [[ -n "${REMOTE_FLOWER_CLIENT_CFG}" ]]; then
      RUNTIME_FLOWER_CLIENT_CFG="${REMOTE_FLOWER_CLIENT_CFG}"
    fi
  else
    RUNTIME_FLOWER_EXECUTOR_CFG="${REMOTE_FLOWER_EXECUTOR_CFG}"

    if [[ -n "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" ]]; then
      RUNTIME_FLOWER_SERVER_CFG="${REMOTE_RUNTIME_CONFIG_DIR}/flower_server.cfg"
    elif [[ -n "${REMOTE_FLOWER_SERVER_CFG}" ]]; then
      RUNTIME_FLOWER_SERVER_CFG="${REMOTE_FLOWER_SERVER_CFG}"
    fi

    if [[ -n "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" ]]; then
      RUNTIME_FLOWER_CLIENT_CFG="${REMOTE_RUNTIME_CONFIG_DIR}/flower_client.cfg"
    elif [[ -n "${REMOTE_FLOWER_CLIENT_CFG}" ]]; then
      RUNTIME_FLOWER_CLIENT_CFG="${REMOTE_FLOWER_CLIENT_CFG}"
    fi
  fi
fi

# ----------------------------------------------------------------------
# Generate runtime hostfile for Flower/gRPC
# ----------------------------------------------------------------------

: > "${LOCAL_HOSTFILE}"

for IDX in "${!ROLES[@]}"; do
  echo "${ROLES[$IDX]} ${RUNTIME_HOSTS[$IDX]}" >> "${LOCAL_HOSTFILE}"
done

# ----------------------------------------------------------------------
# Derive server runtime address and patch gRPC configs
# ----------------------------------------------------------------------

SERVER_RUNTIME_HOST=""
SERVER_RUNTIME_PORT="${SERVER_PORT}"

for IDX in "${!ROLES[@]}"; do
  if [[ "${ROLES[$IDX]}" == "server" ]]; then
    SERVER_RUNTIME_HOST="${RUNTIME_HOSTS[$IDX]}"
    break
  fi
done

[[ -n "${SERVER_RUNTIME_HOST}" ]] || die "could not determine server runtime host from node file"

if [[ -n "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" && -f "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" ]]; then
  SERVER_RUNTIME_PORT="$(read_server_port_from_config "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" "${SERVER_PORT}")"
fi

if [[ "${PATCH_GRPC_CONFIG}" == "true" ]]; then
  if [[ -n "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" || -n "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" ]]; then
    GRPC_PATCH_RESULT="$(
      patch_grpc_config_files \
        "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" \
        "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" \
        "${SERVER_RUNTIME_HOST}" \
        "${SERVER_RUNTIME_PORT}" \
        "${MODE}"
    )"

    echo "[LOCAL] Runtime gRPC config patch result: ${GRPC_PATCH_RESULT}"
  fi
fi

if [[ "${PATCH_EXECUTOR_CONFIG}" == "true" ]]; then
  if [[ -f "${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}" ]]; then
    if [[ -n "${RUNTIME_FLOWER_SERVER_CFG}" || -n "${RUNTIME_FLOWER_CLIENT_CFG}" ]]; then
      PATCH_RESULT="$(patch_executor_config_file \
        "${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}" \
        "${RUNTIME_FLOWER_SERVER_CFG}" \
        "${RUNTIME_FLOWER_CLIENT_CFG}")"
      echo "[LOCAL] Runtime executor config patch result: ${PATCH_RESULT}"
    fi
  elif [[ -n "${CUSTOM_FLOWER_SERVER_CFG}" || -n "${CUSTOM_FLOWER_CLIENT_CFG}" ]]; then
    warn "custom server/client cfg provided, but no custom executor cfg is available to patch"
    warn "provide --custom-flower-executor-cfg if flower_executor.cfg must point to copied server/client cfg files"
  fi
fi

echo "[LOCAL] Launch mode: ${MODE}"
echo "[LOCAL] Nodes file: ${NODES_FILE}"
echo "[LOCAL] Run ID: ${RUN_ID}"
echo
echo "[LOCAL] Generated runtime hostfile:"
cat "${LOCAL_HOSTFILE}"
echo

echo "[LOCAL] Server runtime host: ${SERVER_RUNTIME_HOST}"
echo "[LOCAL] Server runtime port: ${SERVER_RUNTIME_PORT}"
echo "[LOCAL] Remote project directory: ${REMOTE_PROJECT_DIR}"
echo "[LOCAL] Remote venv activate: ${REMOTE_VENV_ACTIVATE}"
echo "[LOCAL] Runtime flower_executor.cfg: ${RUNTIME_FLOWER_EXECUTOR_CFG}"
echo "[LOCAL] Runtime flower_server.cfg: ${RUNTIME_FLOWER_SERVER_CFG:-<not overridden>}"
echo "[LOCAL] Runtime flower_client.cfg: ${RUNTIME_FLOWER_CLIENT_CFG:-<not overridden>}"
echo "[LOCAL] Action: ${ACTION}"
echo "[LOCAL] Repetitions: ${REPETITIONS}"
echo "[LOCAL] Python binary: ${PYTHON_BIN}"
echo "[LOCAL] gather-outputs: ${GATHER_OUTPUTS}"
echo "[LOCAL] Local log directory: ${LOCAL_LOG_DIR}"
echo "[LOCAL] Local gather directory: ${LOCAL_GATHER_DIR}"
echo "[LOCAL] Max parallel remote ops: ${MAX_PARALLEL_REMOTE_OPS}"

if [[ "${MODE}" != "local" ]]; then
  echo "[LOCAL] Remote hostfile path: ${REMOTE_HOSTFILE}"
  echo "[LOCAL] Remote runtime config directory: ${REMOTE_RUNTIME_CONFIG_DIR}"
fi

echo

# ----------------------------------------------------------------------
# Remote helper functions
# ----------------------------------------------------------------------

verify_remote_one() {
  local SSH_TARGET="$1"

  ssh ${SSH_OPTIONS} "${SSH_TARGET}" "
    set -euo pipefail

    test -d '${REMOTE_PROJECT_DIR}' || {
      echo 'ERROR: project directory not found: ${REMOTE_PROJECT_DIR}'
      exit 1
    }

    test -f '${REMOTE_VENV_ACTIVATE}' || {
      echo 'ERROR: venv activate file not found: ${REMOTE_VENV_ACTIVATE}'
      exit 1
    }

    command -v '${PYTHON_BIN}' >/dev/null 2>&1 || {
      echo 'ERROR: python binary not found on remote node: ${PYTHON_BIN}'
      exit 1
    }

    command -v rsync >/dev/null 2>&1 || {
      echo 'ERROR: rsync not found on remote node'
      exit 1
    }

    mkdir -p '${REMOTE_HOSTFILE_DIR}'
    mkdir -p '${REMOTE_RUNTIME_CONFIG_DIR}'

    if [[ ! -f '${RUNTIME_FLOWER_EXECUTOR_CFG}' && ! -f '${REMOTE_RUNTIME_CONFIG_DIR}/flower_executor.cfg' ]]; then
      if [[ '${RUNTIME_FLOWER_EXECUTOR_CFG}' != '${REMOTE_RUNTIME_CONFIG_DIR}/flower_executor.cfg' ]]; then
        test -f '${RUNTIME_FLOWER_EXECUTOR_CFG}' || {
          echo 'ERROR: runtime flower_executor.cfg not found on remote node: ${RUNTIME_FLOWER_EXECUTOR_CFG}'
          exit 1
        }
      fi
    fi
  "
}

copy_runtime_files_one() {
  local SSH_TARGET="$1"

  scp ${SSH_OPTIONS} "${LOCAL_HOSTFILE}" "${SSH_TARGET}:${REMOTE_HOSTFILE}"

  if [[ -f "${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}" ]]; then
    scp ${SSH_OPTIONS} "${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}" \
      "${SSH_TARGET}:${REMOTE_RUNTIME_CONFIG_DIR}/flower_executor.cfg"
  fi

  if [[ -n "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" && -f "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" ]]; then
    scp ${SSH_OPTIONS} "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" \
      "${SSH_TARGET}:${REMOTE_RUNTIME_CONFIG_DIR}/flower_server.cfg"
  fi

  if [[ -n "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" && -f "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" ]]; then
    scp ${SSH_OPTIONS} "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" \
      "${SSH_TARGET}:${REMOTE_RUNTIME_CONFIG_DIR}/flower_client.cfg"
  fi
}

discover_output_folders_local() {
  "${PYTHON_BIN}" - <<PY
from configparser import ConfigParser
from pathlib import Path

config_file = Path("${RUNTIME_FLOWER_EXECUTOR_CFG}")
repetitions = int("${REPETITIONS}")

parser = ConfigParser(interpolation=None)
parser.read(config_file)

sections = [
    s for s in parser.sections()
    if s.startswith("Execution_") and s.endswith("_N Settings")
]

folders = []

for section in sections:
    if not parser.has_option(section, "execution_output_folder"):
        continue

    template = parser.get(section, "execution_output_folder").strip()

    for repetition_idx in range(1, repetitions + 1):
        folders.append(template.replace("N", str(repetition_idx)))

for folder in folders:
    print(folder)
PY
}

discover_output_folders_remote() {
  local SSH_TARGET="$1"

  ssh ${SSH_OPTIONS} "${SSH_TARGET}" "
    set -euo pipefail

    cd '${REMOTE_PROJECT_DIR}'

    '${PYTHON_BIN}' - <<'PY'
from configparser import ConfigParser
from pathlib import Path

config_file = Path('${RUNTIME_FLOWER_EXECUTOR_CFG}')
repetitions = int('${REPETITIONS}')

parser = ConfigParser(interpolation=None)
parser.read(config_file)

sections = [
    s for s in parser.sections()
    if s.startswith('Execution_') and s.endswith('_N Settings')
]

folders = []

for section in sections:
    if not parser.has_option(section, 'execution_output_folder'):
        continue

    template = parser.get(section, 'execution_output_folder').strip()

    for repetition_idx in range(1, repetitions + 1):
        folders.append(template.replace('N', str(repetition_idx)))

for folder in folders:
    print(folder)
PY
  "
}

collect_results_one() {
  local SSH_TARGET="$1"

  local NODE_ID
  NODE_ID="$(sanitize_name "${SSH_TARGET}")"

  local NODE_GATHER_DIR="${LOCAL_GATHER_DIR}/node_${NODE_ID}"
  mkdir -p "${NODE_GATHER_DIR}"

  mapfile -t OUTPUT_FOLDERS < <(discover_output_folders_remote "${SSH_TARGET}")

  if [[ "${#OUTPUT_FOLDERS[@]}" -eq 0 ]]; then
    warn "no execution_output_folder entries found on ${SSH_TARGET}"
    return 0
  fi

  printf "[LOCAL] %s output folder(s) declared on %s:\n" "${#OUTPUT_FOLDERS[@]}" "${SSH_TARGET}"
  printf "  %s\n" "${OUTPUT_FOLDERS[@]}"

  for REL_OUTPUT_FOLDER in "${OUTPUT_FOLDERS[@]}"; do
    local REMOTE_ABS_OUTPUT_FOLDER="${REMOTE_PROJECT_DIR}/${REL_OUTPUT_FOLDER}"
    local LOCAL_DEST_FOLDER="${NODE_GATHER_DIR}/${REL_OUTPUT_FOLDER}"

    echo "[LOCAL] Checking ${SSH_TARGET}:${REMOTE_ABS_OUTPUT_FOLDER}"

    if ssh ${SSH_OPTIONS} "${SSH_TARGET}" "test -d '${REMOTE_ABS_OUTPUT_FOLDER}'"; then
      mkdir -p "${LOCAL_DEST_FOLDER}"

      echo "[LOCAL] Pulling ${SSH_TARGET}:${REMOTE_ABS_OUTPUT_FOLDER}/"
      echo "[LOCAL]      -> ${LOCAL_DEST_FOLDER}/"

      rsync ${RSYNC_OPTIONS} \
        "${SSH_TARGET}:${REMOTE_ABS_OUTPUT_FOLDER}/" \
        "${LOCAL_DEST_FOLDER}/"
    else
      warn "output folder does not exist on ${SSH_TARGET}: ${REMOTE_ABS_OUTPUT_FOLDER}"
    fi
  done
}

# ----------------------------------------------------------------------
# Verify paths and distribute configs
# ----------------------------------------------------------------------

if [[ "${MODE}" == "local" ]]; then
  echo "[LOCAL] Verifying local paths..."

  [[ -d "${REMOTE_PROJECT_DIR}" ]] || die "project directory not found: ${REMOTE_PROJECT_DIR}"
  [[ -f "${REMOTE_VENV_ACTIVATE}" ]] || die "venv activate file not found: ${REMOTE_VENV_ACTIVATE}"
  [[ -f "${RUNTIME_FLOWER_EXECUTOR_CFG}" ]] || die "runtime flower_executor.cfg not found: ${RUNTIME_FLOWER_EXECUTOR_CFG}"

  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python binary not found locally: ${PYTHON_BIN}"
  command -v rsync >/dev/null 2>&1 || die "rsync not found locally"
else
  run_parallel_stage "verify_remote" "${MAX_PARALLEL_REMOTE_OPS}" "${UNIQUE_SSH_TARGETS[@]}" || {
    die "remote verification failed"
  }

  run_parallel_stage "copy_runtime_files" "${MAX_PARALLEL_REMOTE_OPS}" "${UNIQUE_SSH_TARGETS[@]}" || {
    die "runtime file copy failed"
  }
fi

echo

# ----------------------------------------------------------------------
# Launch distributed execution
# ----------------------------------------------------------------------

PIDS=()
PROCESS_NAMES=()

echo "[LOCAL] Launching distributed execution..."

if [[ "${MODE}" == "local" ]]; then
  PROCESS_NAME="local_controller"
  SAFE_PROCESS_NAME="$(sanitize_name "${PROCESS_NAME}")"

  echo "[LOCAL] Local mode detected."
  echo "[LOCAL] Launching a single local controller process."
  echo "[LOCAL] The FlowerExecutor will decide locally whether to start the server and clients."

  (
    set -euo pipefail

    cd "${REMOTE_PROJECT_DIR}"
    source "${REMOTE_VENV_ACTIVATE}"

    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u main.py "${ACTION}" \
      --config-file "${RUNTIME_FLOWER_EXECUTOR_CFG}" \
      --repetitions "${REPETITIONS}" \
      --hostfile "${LOCAL_HOSTFILE}" \
      --gather-outputs "${GATHER_OUTPUTS}"
  ) > "${LOCAL_LOG_DIR}/${SAFE_PROCESS_NAME}.out" \
    2> "${LOCAL_LOG_DIR}/${SAFE_PROCESS_NAME}.err" &

  PIDS+=("$!")
  PROCESS_NAMES+=("${PROCESS_NAME}")
else
  for IDX in "${!ROLES[@]}"; do
    ROLE="${ROLES[$IDX]}"
    SSH_TARGET="${SSH_TARGETS[$IDX]}"
    RUNTIME_HOST="${RUNTIME_HOSTS[$IDX]}"

    PROCESS_NAME="${IDX}_${ROLE}_${RUNTIME_HOST}"
    SAFE_PROCESS_NAME="$(sanitize_name "${PROCESS_NAME}")"

    echo "[LOCAL] Launching ${ROLE} on ${SSH_TARGET}"

    ssh ${SSH_OPTIONS} "${SSH_TARGET}" "
      set -euo pipefail

      cd '${REMOTE_PROJECT_DIR}'
      source '${REMOTE_VENV_ACTIVATE}'

      PYTHONUNBUFFERED=1 '${PYTHON_BIN}' -u main.py '${ACTION}' \
        --config-file '${RUNTIME_FLOWER_EXECUTOR_CFG}' \
        --repetitions '${REPETITIONS}' \
        --hostfile '${REMOTE_HOSTFILE}' \
        --gather-outputs '${GATHER_OUTPUTS}'
    " > "${LOCAL_LOG_DIR}/${SAFE_PROCESS_NAME}.out" \
      2> "${LOCAL_LOG_DIR}/${SAFE_PROCESS_NAME}.err" &

    PIDS+=("$!")
    PROCESS_NAMES+=("${PROCESS_NAME}")
  done
fi

echo
echo "[LOCAL] Waiting for all processes..."
echo "[LOCAL] Follow stdout with:"
echo "tail -f ${LOCAL_LOG_DIR}/*.out"
echo "[LOCAL] Follow stderr with:"
echo "tail -f ${LOCAL_LOG_DIR}/*.err"
echo

FAILED=0

for IDX in "${!PIDS[@]}"; do
  PID="${PIDS[$IDX]}"
  PROCESS_NAME="${PROCESS_NAMES[$IDX]}"

  if wait "${PID}"; then
    echo "[LOCAL] ${PROCESS_NAME} finished successfully."
  else
    STATUS="$?"
    echo "[LOCAL] ERROR: ${PROCESS_NAME} failed with exit code ${STATUS}."
    FAILED=1
  fi
done

echo

if [[ "${FAILED}" -ne 0 ]]; then
  echo "[LOCAL] At least one process failed."
  echo "[LOCAL] Logs are in ${LOCAL_LOG_DIR}/"
  echo "[LOCAL] Pulling available results anyway..."
else
  echo "[LOCAL] All processes finished successfully."
fi

# ----------------------------------------------------------------------
# Pull/copy result folders
# ----------------------------------------------------------------------

echo
echo "[LOCAL] Discovering and collecting result folders..."

if [[ "${MODE}" == "local" ]]; then
  NODE_ID="local"
  NODE_GATHER_DIR="${LOCAL_GATHER_DIR}/node_${NODE_ID}"

  mkdir -p "${NODE_GATHER_DIR}"

  echo "[LOCAL] Discovering local output folders"

  mapfile -t OUTPUT_FOLDERS < <(discover_output_folders_local)

  if [[ "${#OUTPUT_FOLDERS[@]}" -eq 0 ]]; then
    warn "no execution_output_folder entries found"
  else
    printf "[LOCAL] %s output folder(s) declared locally:\n" "${#OUTPUT_FOLDERS[@]}"
    printf "  %s\n" "${OUTPUT_FOLDERS[@]}"

    for REL_OUTPUT_FOLDER in "${OUTPUT_FOLDERS[@]}"; do
      ABS_OUTPUT_FOLDER="${REMOTE_PROJECT_DIR}/${REL_OUTPUT_FOLDER}"
      LOCAL_DEST_FOLDER="${NODE_GATHER_DIR}/${REL_OUTPUT_FOLDER}"

      echo "[LOCAL] Checking ${ABS_OUTPUT_FOLDER}"

      if [[ -d "${ABS_OUTPUT_FOLDER}" ]]; then
        mkdir -p "${LOCAL_DEST_FOLDER}"

        echo "[LOCAL] Copying ${ABS_OUTPUT_FOLDER}/"
        echo "[LOCAL]      -> ${LOCAL_DEST_FOLDER}/"

        rsync ${RSYNC_OPTIONS} "${ABS_OUTPUT_FOLDER}/" "${LOCAL_DEST_FOLDER}/"
      else
        warn "output folder does not exist locally: ${ABS_OUTPUT_FOLDER}"
      fi
    done
  fi
else
  run_parallel_stage "collect_results" "${MAX_PARALLEL_REMOTE_OPS}" "${UNIQUE_SSH_TARGETS[@]}" || {
    warn "result collection failed on at least one node"
  }
fi

# ----------------------------------------------------------------------
# Save run metadata locally
# ----------------------------------------------------------------------

RUN_MANIFEST="${LOCAL_GATHER_DIR}/distributed_run_manifest.txt"

{
  echo "run_id=${RUN_ID}"
  echo "mode=${MODE}"
  echo "nodes_file=${NODES_FILE}"
  echo "server_runtime_host=${SERVER_RUNTIME_HOST}"
  echo "server_runtime_port=${SERVER_RUNTIME_PORT}"
  echo "remote_project_dir=${REMOTE_PROJECT_DIR}"
  echo "remote_venv_activate=${REMOTE_VENV_ACTIVATE}"
  echo "remote_flower_executor_cfg=${REMOTE_FLOWER_EXECUTOR_CFG}"
  echo "remote_flower_server_cfg=${REMOTE_FLOWER_SERVER_CFG}"
  echo "remote_flower_client_cfg=${REMOTE_FLOWER_CLIENT_CFG}"
  echo "custom_flower_executor_cfg=${CUSTOM_FLOWER_EXECUTOR_CFG}"
  echo "custom_flower_server_cfg=${CUSTOM_FLOWER_SERVER_CFG}"
  echo "custom_flower_client_cfg=${CUSTOM_FLOWER_CLIENT_CFG}"
  echo "runtime_flower_executor_cfg=${RUNTIME_FLOWER_EXECUTOR_CFG}"
  echo "runtime_flower_server_cfg=${RUNTIME_FLOWER_SERVER_CFG}"
  echo "runtime_flower_client_cfg=${RUNTIME_FLOWER_CLIENT_CFG}"
  echo "remote_runtime_config_dir=${REMOTE_RUNTIME_CONFIG_DIR}"
  echo "remote_hostfile_dir=${REMOTE_HOSTFILE_DIR}"
  echo "action=${ACTION}"
  echo "repetitions=${REPETITIONS}"
  echo "python_bin=${PYTHON_BIN}"
  echo "gather_outputs=${GATHER_OUTPUTS}"
  echo "patch_executor_config=${PATCH_EXECUTOR_CONFIG}"
  echo "patch_grpc_config=${PATCH_GRPC_CONFIG}"
  echo "max_parallel_remote_ops=${MAX_PARALLEL_REMOTE_OPS}"
  echo "ssh_options=${SSH_OPTIONS}"
  echo "rsync_options=${RSYNC_OPTIONS}"
  echo "local_hostfile=${LOCAL_HOSTFILE}"

  if [[ "${MODE}" != "local" ]]; then
    echo "remote_hostfile=${REMOTE_HOSTFILE}"
  fi

  echo
  echo "[nodes]"
  for IDX in "${!ROLES[@]}"; do
    echo "${ROLES[$IDX]} ${SSH_TARGETS[$IDX]} ${RUNTIME_HOSTS[$IDX]}"
  done

  echo
  echo "[runtime_hostfile]"
  cat "${LOCAL_HOSTFILE}"
} > "${RUN_MANIFEST}"

cp "${NODES_FILE}" "${LOCAL_GATHER_DIR}/$(basename "${NODES_FILE}")"
cp "${LOCAL_HOSTFILE}" "${LOCAL_GATHER_DIR}/$(basename "${LOCAL_HOSTFILE}")"

if [[ -f "${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}" ]]; then
  cp "${LOCAL_RUNTIME_FLOWER_EXECUTOR_CFG}" "${LOCAL_GATHER_DIR}/flower_executor.runtime.cfg"
fi

if [[ -n "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" && -f "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" ]]; then
  cp "${LOCAL_RUNTIME_FLOWER_SERVER_CFG}" "${LOCAL_GATHER_DIR}/flower_server.runtime.cfg"
fi

if [[ -n "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" && -f "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" ]]; then
  cp "${LOCAL_RUNTIME_FLOWER_CLIENT_CFG}" "${LOCAL_GATHER_DIR}/flower_client.runtime.cfg"
fi

echo
echo "[LOCAL] Results gathered in: ${LOCAL_GATHER_DIR}"
echo "[LOCAL] Run manifest: ${RUN_MANIFEST}"
echo "[LOCAL] Logs are in: ${LOCAL_LOG_DIR}"

if [[ "${FAILED}" -ne 0 ]]; then
  exit 1
fi

exit 0
