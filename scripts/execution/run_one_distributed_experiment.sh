#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# MetaCS-FL one-experiment pipeline
#
# Sequence:
#   1. Optional setup/install on target nodes
#   2. Distributed Flower execution
#   3. Merge gathered distributed outputs
# ----------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  run_one_distributed_experiment.sh --nodes-file FILE [options]

Required:
  --nodes-file FILE
      Node description file shared by setup and launch scripts.

Main options:
  --run-id ID
      Explicit run id used for launch and merge.
      Default: current timestamp.

  --install true|false
      Whether to run the setup/install script before launching.
      Default: false.

  --remote-project-dir DIR
      Project directory on the target machines.
      Default: /root/metacs-fl

  --remote-venv-activate FILE
      Virtualenv activation script on target machines.
      Default: <remote-project-dir>/.venv/bin/activate

Script paths:
  --setup-script FILE
      Path to setup script.
      Default: scripts/setup/setup_remote_metacsfl_node.sh

  --launch-script FILE
      Path to distributed launch script.
      Default: scripts/execution/launch_distributed_flower.sh

  --merge-script FILE
      Path to merge script.
      Default: scripts/post_execution/merge_distributed_outputs.py

Setup/install options:
  --repo-url URL
      Repository URL passed to setup script.
      Default: https://github.com/alan-lira/metacs-fl.git

  --branch BRANCH
      Git branch passed to setup script.
      Default: main

  --repo-auth none|token|ssh
      Repository authentication mode passed to setup script.
      Default: none

  --prompt-github-token true|false
      Whether setup should prompt for a GitHub token when --repo-auth token.
      Default: true

  --github-token-env NAME
      Environment variable used by setup script for the GitHub token.
      Default: GITHUB_TOKEN

  --force-reclone true|false
      Passed to setup script.
      Default: true

  --install-powerjoular true|false
      Passed to setup script.
      Default: true

  --append-venv-to-bashrc true|false
      Passed to setup script.
      Default: true

  --install-metacsfl-package true|false
      Passed to setup script.
      Default: true

  --max-parallel-installs N
      Passed to setup script.
      Default: 4

  --output-dir DIR
      Base directory for local pipeline artifacts.
      If provided, default local roots become:
        DIR/setup_logs
        DIR/distributed_launch_logs
        DIR/gathered_results
        DIR/merged_results
      Explicit root-specific arguments override this default.

  --setup-log-root DIR
      Passed to setup script.
      Default: setup_logs

Launch options:
  --custom-flower-executor-cfg FILE
      Custom flower_executor.cfg passed to launch script.

  --custom-flower-server-cfg FILE
      Custom flower_server.cfg passed to launch script.

  --custom-flower-client-cfg FILE
      Custom flower_client.cfg passed to launch script.

  --remote-flower-executor-cfg FILE
      Remote flower_executor.cfg passed to launch script.

  --remote-flower-server-cfg FILE
      Remote flower_server.cfg passed to launch script.

  --remote-flower-client-cfg FILE
      Remote flower_client.cfg passed to launch script.

  --patch-executor-config true|false
      Passed to launch script.
      Default: true

  --patch-grpc-config true|false
      Passed to launch script.
      Default: true

  --server-port PORT
      Passed to launch script.
      Default: 8080

  --repetitions N
      Passed to launch script.
      Default: 1

  --action ACTION
      main.py action passed to launch script.
      Default: execute_fl_with_flower

  --gather-outputs true|false
      Passed to launch script.
      Default: false

  --max-parallel-remote-ops N
      Passed to launch script.
      Default: 8

  --local-log-root DIR
      Passed to launch script.
      Default: distributed_launch_logs

  --local-gather-root DIR
      Passed to launch script.
      Default: gathered_results

Merge options:
  --merge-output-root DIR
      Root directory where merged outputs are written.
      Final output path will be <merge-output-root>/<run-id>.
      Default: merged_results

  --node-prefix PREFIX
      Node directory prefix passed to merge script.
      Default: node_

  --add-source-node true|false
      Passed to merge script.
      Default: false

  --deduplicate-rows true|false
      Passed to merge script.
      Default: false

  --clean-merge-output true|false
      Whether merge script deletes previous output before writing.
      Default: true

Common:
  --python-bin BIN
      Python executable passed to setup and launch scripts.
      Default: python3

  --ssh-options "OPTIONS"
      SSH options passed to setup and launch scripts.
      Default: empty

  --rsync-options "OPTIONS"
      Rsync options passed to launch script.
      Default: -az

Extra passthrough:
  --setup-arg ARG
      Extra single argument passed to setup script.
      Can be repeated. Use twice for option/value pairs:
        --setup-arg --some-option --setup-arg value

  --launch-arg ARG
      Extra single argument passed to launch script.
      Can be repeated.

  --merge-arg ARG
      Extra single argument passed to merge script.
      Can be repeated.

  -h, --help
      Show this help message.

Examples:

  Run only launch + merge:
    bash scripts/run_one_distributed_experiment.sh \
      --nodes-file scripts/nodes.g5k.txt \
      --remote-project-dir /root/metacs-fl \
      --run-id cifar10_iid_test_001 \
      --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
      --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
      --custom-flower-client-cfg experiments/my_run/flower_client.cfg

  Run install + launch + merge:
    bash scripts/run_one_distributed_experiment.sh \
      --install true \
      --nodes-file scripts/nodes.g5k.txt \
      --remote-project-dir /root/metacs-fl \
      --repo-auth token \
      --prompt-github-token true \
      --branch main \
      --run-id cifar10_iid_test_001 \
      --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
      --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
      --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
      --max-parallel-installs 8 \
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

bool_check() {
  local name="$1"
  local value="$2"

  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    die "${name} must be true or false"
  fi
}

positive_int_check() {
  local name="$1"
  local value="$2"

  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    die "${name} must be a positive integer"
  fi

  if [[ "${value}" -lt 1 ]]; then
    die "${name} must be >= 1"
  fi
}

print_step() {
  local msg="$1"

  echo
  echo "======================================================================"
  echo "${msg}"
  echo "======================================================================"
}

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

NODES_FILE=""

RUN_ID=""
INSTALL="false"

REMOTE_PROJECT_DIR="/root/metacs-fl"
REMOTE_VENV_ACTIVATE=""

SETUP_SCRIPT="scripts/setup/setup_remote_metacsfl_node.sh"
LAUNCH_SCRIPT="scripts/execution/launch_distributed_flower.sh"
MERGE_SCRIPT="scripts/post_execution/merge_distributed_outputs.py"

# Setup defaults.
REPO_URL="https://github.com/alan-lira/metacs-fl.git"
BRANCH="main"
REPO_AUTH="none"
PROMPT_GITHUB_TOKEN="true"
GITHUB_TOKEN_ENV="GITHUB_TOKEN"
FORCE_RECLONE="true"
INSTALL_POWERJOULAR="true"
APPEND_VENV_TO_BASHRC="true"
INSTALL_METACSFL_PACKAGE="true"
MAX_PARALLEL_INSTALLS="4"
SETUP_LOG_ROOT="setup_logs"

# Launch defaults.
CUSTOM_FLOWER_EXECUTOR_CFG=""
CUSTOM_FLOWER_SERVER_CFG=""
CUSTOM_FLOWER_CLIENT_CFG=""

REMOTE_FLOWER_EXECUTOR_CFG=""
REMOTE_FLOWER_SERVER_CFG=""
REMOTE_FLOWER_CLIENT_CFG=""

PATCH_EXECUTOR_CONFIG="true"
PATCH_GRPC_CONFIG="true"
SERVER_PORT="8080"

REPETITIONS="1"
ACTION="execute_fl_with_flower"
GATHER_OUTPUTS="false"
MAX_PARALLEL_REMOTE_OPS="8"

LOCAL_LOG_ROOT="distributed_launch_logs"
LOCAL_GATHER_ROOT="gathered_results"

# Merge defaults.
MERGE_OUTPUT_ROOT="merged_results"
NODE_PREFIX="node_"
ADD_SOURCE_NODE="false"
DEDUPLICATE_ROWS="false"
CLEAN_MERGE_OUTPUT="true"
OUTPUT_DIR=""
SETUP_LOG_ROOT_EXPLICIT="false"
LOCAL_LOG_ROOT_EXPLICIT="false"
LOCAL_GATHER_ROOT_EXPLICIT="false"
MERGE_OUTPUT_ROOT_EXPLICIT="false"

# Common.
PYTHON_BIN="python3"
SSH_OPTIONS=""
RSYNC_OPTIONS="-az"

SETUP_EXTRA_ARGS=()
LAUNCH_EXTRA_ARGS=()
MERGE_EXTRA_ARGS=()

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

    --run-id)
      require_value "$1" "${2:-}"
      RUN_ID="$2"
      shift 2
      ;;

    --install)
      require_value "$1" "${2:-}"
      INSTALL="$2"
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

    --setup-script)
      require_value "$1" "${2:-}"
      SETUP_SCRIPT="$2"
      shift 2
      ;;

    --launch-script)
      require_value "$1" "${2:-}"
      LAUNCH_SCRIPT="$2"
      shift 2
      ;;

    --merge-script)
      require_value "$1" "${2:-}"
      MERGE_SCRIPT="$2"
      shift 2
      ;;

    --repo-url)
      require_value "$1" "${2:-}"
      REPO_URL="$2"
      shift 2
      ;;

    --branch)
      require_value "$1" "${2:-}"
      BRANCH="$2"
      shift 2
      ;;

    --repo-auth)
      require_value "$1" "${2:-}"
      REPO_AUTH="$2"
      shift 2
      ;;

    --prompt-github-token)
      require_value "$1" "${2:-}"
      PROMPT_GITHUB_TOKEN="$2"
      shift 2
      ;;

    --github-token-env)
      require_value "$1" "${2:-}"
      GITHUB_TOKEN_ENV="$2"
      shift 2
      ;;

    --force-reclone)
      require_value "$1" "${2:-}"
      FORCE_RECLONE="$2"
      shift 2
      ;;

    --install-powerjoular)
      require_value "$1" "${2:-}"
      INSTALL_POWERJOULAR="$2"
      shift 2
      ;;

    --append-venv-to-bashrc)
      require_value "$1" "${2:-}"
      APPEND_VENV_TO_BASHRC="$2"
      shift 2
      ;;

    --install-metacsfl-package)
      require_value "$1" "${2:-}"
      INSTALL_METACSFL_PACKAGE="$2"
      shift 2
      ;;

    --max-parallel-installs)
      require_value "$1" "${2:-}"
      MAX_PARALLEL_INSTALLS="$2"
      shift 2
      ;;

    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="$2"
      shift 2
      ;;

    --setup-log-root)
      require_value "$1" "${2:-}"
      SETUP_LOG_ROOT="$2"
      SETUP_LOG_ROOT_EXPLICIT="true"
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

    --repetitions)
      require_value "$1" "${2:-}"
      REPETITIONS="$2"
      shift 2
      ;;

    --action)
      require_value "$1" "${2:-}"
      ACTION="$2"
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

    --local-log-root)
      require_value "$1" "${2:-}"
      LOCAL_LOG_ROOT="$2"
      LOCAL_LOG_ROOT_EXPLICIT="true"
      shift 2
      ;;

    --local-gather-root)
      require_value "$1" "${2:-}"
      LOCAL_GATHER_ROOT="$2"
      LOCAL_GATHER_ROOT_EXPLICIT="true"
      shift 2
      ;;

    --merge-output-root)
      require_value "$1" "${2:-}"
      MERGE_OUTPUT_ROOT="$2"
      MERGE_OUTPUT_ROOT_EXPLICIT="true"
      shift 2
      ;;

    --node-prefix)
      require_value "$1" "${2:-}"
      NODE_PREFIX="$2"
      shift 2
      ;;

    --add-source-node)
      require_value "$1" "${2:-}"
      ADD_SOURCE_NODE="$2"
      shift 2
      ;;

    --deduplicate-rows)
      require_value "$1" "${2:-}"
      DEDUPLICATE_ROWS="$2"
      shift 2
      ;;

    --clean-merge-output)
      require_value "$1" "${2:-}"
      CLEAN_MERGE_OUTPUT="$2"
      shift 2
      ;;

    --python-bin)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
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

    --setup-arg)
      require_value "$1" "${2:-}"
      SETUP_EXTRA_ARGS+=("$2")
      shift 2
      ;;

    --launch-arg)
      require_value "$1" "${2:-}"
      LAUNCH_EXTRA_ARGS+=("$2")
      shift 2
      ;;

    --merge-arg)
      require_value "$1" "${2:-}"
      MERGE_EXTRA_ARGS+=("$2")
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
[[ -f "${LAUNCH_SCRIPT}" ]] || die "launch script not found: ${LAUNCH_SCRIPT}"
[[ -f "${MERGE_SCRIPT}" ]] || die "merge script not found: ${MERGE_SCRIPT}"

if [[ "${INSTALL}" == "true" ]]; then
  [[ -f "${SETUP_SCRIPT}" ]] || die "setup script not found: ${SETUP_SCRIPT}"
fi

bool_check "--install" "${INSTALL}"
bool_check "--prompt-github-token" "${PROMPT_GITHUB_TOKEN}"
bool_check "--force-reclone" "${FORCE_RECLONE}"
bool_check "--install-powerjoular" "${INSTALL_POWERJOULAR}"
bool_check "--append-venv-to-bashrc" "${APPEND_VENV_TO_BASHRC}"
bool_check "--install-metacsfl-package" "${INSTALL_METACSFL_PACKAGE}"
bool_check "--patch-executor-config" "${PATCH_EXECUTOR_CONFIG}"
bool_check "--patch-grpc-config" "${PATCH_GRPC_CONFIG}"
bool_check "--gather-outputs" "${GATHER_OUTPUTS}"
bool_check "--add-source-node" "${ADD_SOURCE_NODE}"
bool_check "--deduplicate-rows" "${DEDUPLICATE_ROWS}"
bool_check "--clean-merge-output" "${CLEAN_MERGE_OUTPUT}"

positive_int_check "--max-parallel-installs" "${MAX_PARALLEL_INSTALLS}"
positive_int_check "--max-parallel-remote-ops" "${MAX_PARALLEL_REMOTE_OPS}"
positive_int_check "--repetitions" "${REPETITIONS}"

case "${REPO_AUTH}" in
  none|token|ssh)
    ;;
  *)
    die "--repo-auth must be one of: none, token, ssh"
    ;;
esac

for CFG in \
  "${CUSTOM_FLOWER_EXECUTOR_CFG}" \
  "${CUSTOM_FLOWER_SERVER_CFG}" \
  "${CUSTOM_FLOWER_CLIENT_CFG}"
do
  if [[ -n "${CFG}" && ! -f "${CFG}" ]]; then
    die "custom config file not found: ${CFG}"
  fi
done

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "${REMOTE_VENV_ACTIVATE}" ]]; then
  REMOTE_VENV_ACTIVATE="${REMOTE_PROJECT_DIR}/.venv/bin/activate"
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR%/}"

  if [[ "${SETUP_LOG_ROOT_EXPLICIT}" != "true" ]]; then
    SETUP_LOG_ROOT="${OUTPUT_DIR}/setup_logs"
  fi

  if [[ "${LOCAL_LOG_ROOT_EXPLICIT}" != "true" ]]; then
    LOCAL_LOG_ROOT="${OUTPUT_DIR}/distributed_launch_logs"
  fi

  if [[ "${LOCAL_GATHER_ROOT_EXPLICIT}" != "true" ]]; then
    LOCAL_GATHER_ROOT="${OUTPUT_DIR}/gathered_results"
  fi

  if [[ "${MERGE_OUTPUT_ROOT_EXPLICIT}" != "true" ]]; then
    MERGE_OUTPUT_ROOT="${OUTPUT_DIR}/merged_results"
  fi
fi

GATHERED_ROOT="${LOCAL_GATHER_ROOT}/${RUN_ID}"
MERGED_OUTPUT_DIR="${MERGE_OUTPUT_ROOT}/${RUN_ID}"

mkdir -p "${LOCAL_LOG_ROOT}"
mkdir -p "${LOCAL_GATHER_ROOT}"
mkdir -p "${MERGE_OUTPUT_ROOT}"

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

print_step "[PIPELINE] Configuration"

echo "[PIPELINE] run_id: ${RUN_ID}"
echo "[PIPELINE] install: ${INSTALL}"
echo "[PIPELINE] nodes_file: ${NODES_FILE}"
echo "[PIPELINE] remote_project_dir: ${REMOTE_PROJECT_DIR}"
echo "[PIPELINE] remote_venv_activate: ${REMOTE_VENV_ACTIVATE}"
echo "[PIPELINE] gathered_root: ${GATHERED_ROOT}"
echo "[PIPELINE] merged_output_dir: ${MERGED_OUTPUT_DIR}"
echo "[PIPELINE] setup_script: ${SETUP_SCRIPT}"
echo "[PIPELINE] launch_script: ${LAUNCH_SCRIPT}"
echo "[PIPELINE] merge_script: ${MERGE_SCRIPT}"

# ----------------------------------------------------------------------
# Step 1: Optional setup/install
# ----------------------------------------------------------------------

if [[ "${INSTALL}" == "true" ]]; then
  print_step "[PIPELINE] Step 1/3: setup/install"

  SETUP_CMD=(
    bash "${SETUP_SCRIPT}"
    --nodes-file "${NODES_FILE}"
    --remote-project-dir "${REMOTE_PROJECT_DIR}"
    --repo-url "${REPO_URL}"
    --branch "${BRANCH}"
    --repo-auth "${REPO_AUTH}"
    --prompt-github-token "${PROMPT_GITHUB_TOKEN}"
    --github-token-env "${GITHUB_TOKEN_ENV}"
    --force-reclone "${FORCE_RECLONE}"
    --install-powerjoular "${INSTALL_POWERJOULAR}"
    --append-venv-to-bashrc "${APPEND_VENV_TO_BASHRC}"
    --install-metacsfl-package "${INSTALL_METACSFL_PACKAGE}"
    --python-bin "${PYTHON_BIN}"
    --max-parallel-installs "${MAX_PARALLEL_INSTALLS}"
    --setup-log-root "${SETUP_LOG_ROOT}"
    --run-id "${RUN_ID}_setup"
  )

  if [[ -n "${SSH_OPTIONS}" ]]; then
    SETUP_CMD+=(--ssh-options "${SSH_OPTIONS}")
  fi

  if [[ "${#SETUP_EXTRA_ARGS[@]}" -gt 0 ]]; then
    SETUP_CMD+=("${SETUP_EXTRA_ARGS[@]}")
  fi

  echo "[PIPELINE] Running setup command:"
  printf '  %q' "${SETUP_CMD[@]}"
  echo

  "${SETUP_CMD[@]}"
else
  print_step "[PIPELINE] Step 1/3: setup/install skipped"
  echo "[PIPELINE] Use --install true to run setup before launch."
fi

# ----------------------------------------------------------------------
# Step 2: Launch distributed execution
# ----------------------------------------------------------------------

print_step "[PIPELINE] Step 2/3: distributed launch"

LAUNCH_CMD=(
  bash "${LAUNCH_SCRIPT}"
  --nodes-file "${NODES_FILE}"
  --remote-project-dir "${REMOTE_PROJECT_DIR}"
  --remote-venv-activate "${REMOTE_VENV_ACTIVATE}"
  --patch-executor-config "${PATCH_EXECUTOR_CONFIG}"
  --patch-grpc-config "${PATCH_GRPC_CONFIG}"
  --server-port "${SERVER_PORT}"
  --action "${ACTION}"
  --repetitions "${REPETITIONS}"
  --python-bin "${PYTHON_BIN}"
  --local-log-root "${LOCAL_LOG_ROOT}"
  --local-gather-root "${LOCAL_GATHER_ROOT}"
  --gather-outputs "${GATHER_OUTPUTS}"
  --max-parallel-remote-ops "${MAX_PARALLEL_REMOTE_OPS}"
  --rsync-options "${RSYNC_OPTIONS}"
  --run-id "${RUN_ID}"
)

if [[ -n "${SSH_OPTIONS}" ]]; then
  LAUNCH_CMD+=(--ssh-options "${SSH_OPTIONS}")
fi

if [[ -n "${CUSTOM_FLOWER_EXECUTOR_CFG}" ]]; then
  LAUNCH_CMD+=(--custom-flower-executor-cfg "${CUSTOM_FLOWER_EXECUTOR_CFG}")
fi

if [[ -n "${CUSTOM_FLOWER_SERVER_CFG}" ]]; then
  LAUNCH_CMD+=(--custom-flower-server-cfg "${CUSTOM_FLOWER_SERVER_CFG}")
fi

if [[ -n "${CUSTOM_FLOWER_CLIENT_CFG}" ]]; then
  LAUNCH_CMD+=(--custom-flower-client-cfg "${CUSTOM_FLOWER_CLIENT_CFG}")
fi

if [[ -n "${REMOTE_FLOWER_EXECUTOR_CFG}" ]]; then
  LAUNCH_CMD+=(--remote-flower-executor-cfg "${REMOTE_FLOWER_EXECUTOR_CFG}")
fi

if [[ -n "${REMOTE_FLOWER_SERVER_CFG}" ]]; then
  LAUNCH_CMD+=(--remote-flower-server-cfg "${REMOTE_FLOWER_SERVER_CFG}")
fi

if [[ -n "${REMOTE_FLOWER_CLIENT_CFG}" ]]; then
  LAUNCH_CMD+=(--remote-flower-client-cfg "${REMOTE_FLOWER_CLIENT_CFG}")
fi

if [[ "${#LAUNCH_EXTRA_ARGS[@]}" -gt 0 ]]; then
  LAUNCH_CMD+=("${LAUNCH_EXTRA_ARGS[@]}")
fi

echo "[PIPELINE] Running launch command:"
printf '  %q' "${LAUNCH_CMD[@]}"
echo

"${LAUNCH_CMD[@]}"

if [[ ! -d "${GATHERED_ROOT}" ]]; then
  die "expected gathered root was not created: ${GATHERED_ROOT}"
fi

# ----------------------------------------------------------------------
# Step 3: Merge distributed outputs
# ----------------------------------------------------------------------

print_step "[PIPELINE] Step 3/3: merge gathered outputs"

MERGE_CMD=(
  "${PYTHON_BIN}" "${MERGE_SCRIPT}"
  "${GATHERED_ROOT}"
  "${MERGED_OUTPUT_DIR}"
  --node-prefix "${NODE_PREFIX}"
  --add-source-node "${ADD_SOURCE_NODE}"
  --deduplicate-rows "${DEDUPLICATE_ROWS}"
  --clean-output "${CLEAN_MERGE_OUTPUT}"
)

if [[ "${#MERGE_EXTRA_ARGS[@]}" -gt 0 ]]; then
  MERGE_CMD+=("${MERGE_EXTRA_ARGS[@]}")
fi

echo "[PIPELINE] Running merge command:"
printf '  %q' "${MERGE_CMD[@]}"
echo

"${MERGE_CMD[@]}"

# ----------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------

print_step "[PIPELINE] Completed successfully"

echo "[PIPELINE] Run ID: ${RUN_ID}"
echo "[PIPELINE] Gathered results:"
echo "  ${GATHERED_ROOT}"
echo "[PIPELINE] Merged results:"
echo "  ${MERGED_OUTPUT_DIR}"
echo "[PIPELINE] Launch logs:"
echo "  ${LOCAL_LOG_ROOT}/${RUN_ID}"

if [[ "${INSTALL}" == "true" ]]; then
  echo "[PIPELINE] Setup logs:"
  echo "  ${SETUP_LOG_ROOT}/${RUN_ID}_setup"
fi

if [[ -f "${MERGED_OUTPUT_DIR}/merge_report.json" ]]; then
  echo "[PIPELINE] Merge report:"
  echo "  ${MERGED_OUTPUT_DIR}/merge_report.json"
fi

exit 0
