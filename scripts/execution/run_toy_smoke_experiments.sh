#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Run MetaCS-FL toy smoke experiments.
#
# This is a preflight wrapper around run_many_distributed_experiments.sh.
# It validates that the distributed and server-only execution backends work
# before launching larger experiment campaigns.
# ----------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  run_toy_smoke_experiments.sh --nodes-file FILE [options]

Required:
  --nodes-file FILE
      Node description file.

Main options:
  --campaign-id ID
      Campaign id for the smoke run.
      Default: toy_smoke_<timestamp>

  --toy-folder DIR
      Local toy smoke experiments folder.
      Default: toy_smoke_experiments

  --manifest FILE
      Smoke campaign manifest.
      Default: <toy-folder>/campaign_manifest.csv

  --remote-project-dir DIR
      Project directory on target nodes.
      Default: /root/metacs-fl

  --remote-venv-activate FILE
      Virtualenv activation script on target nodes.
      Default: <remote-project-dir>/.venv/bin/activate

  --install true|false
      Whether run_many_distributed_experiments.sh should run setup for distributed experiments.
      Default: false

  --sync-toy-folder true|false
      Whether to sync the toy smoke folder to remote nodes before running.
      Default: true

  --max-parallel-remote-ops N
      Parallelism used by the distributed launcher.
      Default: 8

  --max-parallel-syncs N
      Parallelism used when syncing the toy folder to remote nodes.
      Default: 8

  --repetitions N
      Number of repetitions passed to distributed smoke experiments.
      Default: 1

  --dry-run true|false
      Only validate and print what would run.
      Default: false

Script paths:
  --run-many-script FILE
      Path to run_many_distributed_experiments.sh.
      Default: scripts/execution/run_many_distributed_experiments.sh

Common:
  --python-bin BIN
      Python executable.
      Default: python3

  --ssh-options "OPTIONS"
      Extra options passed to ssh.
      Default: empty

  --rsync-options "OPTIONS"
      Extra options passed to rsync.
      Default: -az --delete

Examples:

  Local smoke test:
    bash scripts/execution/run_toy_smoke_experiments.sh \
      --nodes-file scripts/nodes.local.txt \
      --remote-project-dir "$PWD" \
      --remote-venv-activate "$PWD/.venv/bin/activate"

  Grid'5000 smoke test:
    bash scripts/execution/run_toy_smoke_experiments.sh \
      --nodes-file scripts/nodes.g5k.txt \
      --remote-project-dir /root/metacs-fl \
      --remote-venv-activate /root/metacs-fl/.venv/bin/activate

  Grid'5000 dry-run:
    bash scripts/execution/run_toy_smoke_experiments.sh \
      --nodes-file scripts/nodes.g5k.txt \
      --remote-project-dir /root/metacs-fl \
      --dry-run true
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

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

NODES_FILE=""
CAMPAIGN_ID=""

TOY_FOLDER="toy_smoke_experiments"
MANIFEST=""

REMOTE_PROJECT_DIR="/root/metacs-fl"
REMOTE_VENV_ACTIVATE=""

INSTALL="false"
SYNC_TOY_FOLDER="true"
DRY_RUN="false"

MAX_PARALLEL_REMOTE_OPS="8"
MAX_PARALLEL_SYNCS="8"
REPETITIONS="1"

RUN_MANY_SCRIPT="scripts/execution/run_many_distributed_experiments.sh"

PYTHON_BIN="python3"
SSH_OPTIONS=""
RSYNC_OPTIONS="-az --delete"

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

    --campaign-id)
      require_value "$1" "${2:-}"
      CAMPAIGN_ID="$2"
      shift 2
      ;;

    --toy-folder)
      require_value "$1" "${2:-}"
      TOY_FOLDER="$2"
      shift 2
      ;;

    --manifest)
      require_value "$1" "${2:-}"
      MANIFEST="$2"
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

    --install)
      require_value "$1" "${2:-}"
      INSTALL="$2"
      shift 2
      ;;

    --sync-toy-folder)
      require_value "$1" "${2:-}"
      SYNC_TOY_FOLDER="$2"
      shift 2
      ;;

    --max-parallel-remote-ops)
      require_value "$1" "${2:-}"
      MAX_PARALLEL_REMOTE_OPS="$2"
      shift 2
      ;;

    --max-parallel-syncs)
      require_value "$1" "${2:-}"
      MAX_PARALLEL_SYNCS="$2"
      shift 2
      ;;

    --repetitions)
      require_value "$1" "${2:-}"
      REPETITIONS="$2"
      shift 2
      ;;

    --dry-run)
      require_value "$1" "${2:-}"
      DRY_RUN="$2"
      shift 2
      ;;

    --run-many-script)
      require_value "$1" "${2:-}"
      RUN_MANY_SCRIPT="$2"
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
[[ -d "${TOY_FOLDER}" ]] || die "toy folder not found: ${TOY_FOLDER}"
[[ -f "${RUN_MANY_SCRIPT}" ]] || die "run-many script not found: ${RUN_MANY_SCRIPT}"

if [[ -z "${MANIFEST}" ]]; then
  MANIFEST="${TOY_FOLDER}/campaign_manifest.csv"
fi

[[ -f "${MANIFEST}" ]] || die "manifest not found: ${MANIFEST}"

bool_check "--install" "${INSTALL}"
bool_check "--sync-toy-folder" "${SYNC_TOY_FOLDER}"
bool_check "--dry-run" "${DRY_RUN}"

positive_int_check "--max-parallel-remote-ops" "${MAX_PARALLEL_REMOTE_OPS}"
positive_int_check "--max-parallel-syncs" "${MAX_PARALLEL_SYNCS}"
positive_int_check "--repetitions" "${REPETITIONS}"

if [[ -z "${CAMPAIGN_ID}" ]]; then
  CAMPAIGN_ID="toy_smoke_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "${REMOTE_VENV_ACTIVATE}" ]]; then
  REMOTE_VENV_ACTIVATE="${REMOTE_PROJECT_DIR}/.venv/bin/activate"
fi

# ----------------------------------------------------------------------
# Parse node file
# ----------------------------------------------------------------------

MODE=""
UNIQUE_TARGETS=()

while IFS= read -r RAW_LINE || [[ -n "${RAW_LINE}" ]]; do
  LINE="${RAW_LINE%%#*}"
  LINE="$(trim "${LINE}")"

  [[ -z "${LINE}" ]] && continue

  read -r FIELD1 FIELD2 FIELD3 EXTRA <<< "${LINE}"

  FIELD1="$(trim "${FIELD1:-}")"
  FIELD2="$(trim "${FIELD2:-}")"

  if [[ "${FIELD1}" == "mode" ]]; then
    MODE="${FIELD2}"
    continue
  fi

  if [[ "${FIELD1}" == "server" || "${FIELD1}" == "client" ]]; then
    if [[ "${FIELD2}" != "-" ]]; then
      found="false"
      for existing in "${UNIQUE_TARGETS[@]}"; do
        if [[ "${existing}" == "${FIELD2}" ]]; then
          found="true"
          break
        fi
      done

      if [[ "${found}" == "false" ]]; then
        UNIQUE_TARGETS+=("${FIELD2}")
      fi
    fi
  fi
done < "${NODES_FILE}"

[[ -n "${MODE}" ]] || die "nodes file must contain a mode line"

case "${MODE}" in
  local|remote|g5k)
    ;;
  *)
    die "unsupported mode in nodes file: ${MODE}"
    ;;
esac

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

echo "======================================================================"
echo "[TOY_SMOKE] Configuration"
echo "======================================================================"
echo "[TOY_SMOKE] campaign_id: ${CAMPAIGN_ID}"
echo "[TOY_SMOKE] nodes_file: ${NODES_FILE}"
echo "[TOY_SMOKE] mode: ${MODE}"
echo "[TOY_SMOKE] toy_folder: ${TOY_FOLDER}"
echo "[TOY_SMOKE] manifest: ${MANIFEST}"
echo "[TOY_SMOKE] remote_project_dir: ${REMOTE_PROJECT_DIR}"
echo "[TOY_SMOKE] remote_venv_activate: ${REMOTE_VENV_ACTIVATE}"
echo "[TOY_SMOKE] install: ${INSTALL}"
echo "[TOY_SMOKE] sync_toy_folder: ${SYNC_TOY_FOLDER}"
echo "[TOY_SMOKE] dry_run: ${DRY_RUN}"

if [[ "${MODE}" != "local" ]]; then
  echo "[TOY_SMOKE] remote targets:"
  for target in "${UNIQUE_TARGETS[@]}"; do
    echo "  - ${target}"
  done
fi

# ----------------------------------------------------------------------
# Sync toy folder to remote nodes
# ----------------------------------------------------------------------

if [[ "${MODE}" != "local" && "${SYNC_TOY_FOLDER}" == "true" ]]; then
  echo
  echo "======================================================================"
  echo "[TOY_SMOKE] Syncing toy folder to remote nodes"
  echo "======================================================================"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[TOY_SMOKE] Dry run: skipping remote sync."
  else
    PIDS=()
    declare -A PID_TO_TARGET=()
    FAILED_SYNC=0

    for target in "${UNIQUE_TARGETS[@]}"; do
      wait_for_available_slot "${MAX_PARALLEL_SYNCS}"

      (
        set -euo pipefail

        echo "[TOY_SMOKE] Syncing ${TOY_FOLDER}/ to ${target}:${REMOTE_PROJECT_DIR}/${TOY_FOLDER}/"

        ssh ${SSH_OPTIONS} "${target}" "mkdir -p '${REMOTE_PROJECT_DIR}/${TOY_FOLDER}'"

        rsync ${RSYNC_OPTIONS} \
          "${TOY_FOLDER}/" \
          "${target}:${REMOTE_PROJECT_DIR}/${TOY_FOLDER}/"

        echo "[TOY_SMOKE] Sync complete on ${target}"
      ) &

      pid="$!"
      PIDS+=("${pid}")
      PID_TO_TARGET["${pid}"]="${target}"
    done

    for pid in "${PIDS[@]}"; do
      target="${PID_TO_TARGET[$pid]}"

      if wait "${pid}"; then
        :
      else
        echo "[TOY_SMOKE] ERROR: sync failed on ${target}" >&2
        FAILED_SYNC=1
      fi
    done

    if [[ "${FAILED_SYNC}" -ne 0 ]]; then
      die "one or more toy folder sync operations failed"
    fi
  fi
elif [[ "${MODE}" == "local" ]]; then
  echo
  echo "[TOY_SMOKE] Local mode: no remote sync needed."
else
  echo
  echo "[TOY_SMOKE] Remote toy folder sync disabled."
fi

# ----------------------------------------------------------------------
# Run campaign
# ----------------------------------------------------------------------

echo
echo "======================================================================"
echo "[TOY_SMOKE] Running smoke campaign"
echo "======================================================================"

CMD=(
  bash "${RUN_MANY_SCRIPT}"
  --manifest "${MANIFEST}"
  --campaign-id "${CAMPAIGN_ID}"
  --nodes-file "${NODES_FILE}"
  --remote-project-dir "${REMOTE_PROJECT_DIR}"
  --remote-venv-activate "${REMOTE_VENV_ACTIVATE}"
  --install "${INSTALL}"
  --max-parallel-experiments 1
  --max-parallel-remote-ops "${MAX_PARALLEL_REMOTE_OPS}"
  --repetitions "${REPETITIONS}"
  --python-bin "${PYTHON_BIN}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run true)
fi

if [[ -n "${SSH_OPTIONS}" ]]; then
  CMD+=(--ssh-options "${SSH_OPTIONS}")
fi

# run_many uses rsync options for distributed collection/copy.
CMD+=(--rsync-options "${RSYNC_OPTIONS}")

printf '[TOY_SMOKE] CMD:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo
echo "======================================================================"
echo "[TOY_SMOKE] Completed"
echo "======================================================================"
echo "[TOY_SMOKE] Campaign:"
echo "  campaign_runs/${CAMPAIGN_ID}"
echo "[TOY_SMOKE] Logs:"
echo "  campaign_runs/${CAMPAIGN_ID}/logs"
echo "[TOY_SMOKE] Summary:"
echo "  campaign_runs/${CAMPAIGN_ID}/summaries/campaign_summary.csv"

exit 0
