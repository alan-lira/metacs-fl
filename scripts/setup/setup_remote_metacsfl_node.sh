#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# MetaCS-FL multi-node installation coordinator
# ----------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  setup_remote_metacsfl_node.sh --nodes-file FILE [options]

Required:
  --nodes-file FILE
      Node description file using the shared format:
        mode local|remote|g5k
        server <ssh_target|-> <runtime_host>
        client <ssh_target|-> <runtime_host>

Options:
  --repo-url URL
      MetaCS-FL Git repository URL.
      Default: https://github.com/alan-lira/metacs-fl.git

  --branch BRANCH
      Git branch to clone.
      Default: main

  --repo-auth none|token|ssh
      Repository authentication mode.
        none  = clone normally using --repo-url.
        token = use a GitHub token for HTTPS clone/pull.
        ssh   = use SSH clone URL, e.g. git@github.com:alan-lira/metacs-fl.git.
      Default: none

  --prompt-github-token true|false
      If true and --repo-auth token is used, prompt locally for a GitHub token.
      If false, the script expects GITHUB_TOKEN to already be set.
      Default: true

  --github-token-env NAME
      Environment variable name used to read the GitHub token when --repo-auth token.
      Default: GITHUB_TOKEN

  --remote-project-dir DIR
      Project directory as seen by the target machine.
      For local mode: local project path.
      For remote/g5k mode: project path on each remote node.
      Default: /root/metacs-fl

  --powerjoular-version VERSION
      PowerJoular version to install.
      Default: 1.1.0

  --force-reclone true|false
      If true, remove the project directory and clone again.
      If false, reuse an existing clone and git pull.
      Default: true

  --install-powerjoular true|false
      Whether to install PowerJoular.
      Default: true

  --append-venv-to-bashrc true|false
      Whether to append venv activation to ~/.bashrc on each target.
      Default: true

  --install-metacsfl-package true|false
      Whether to install the MetaCS-FL project itself into the venv.
      The script uses: pip3 install -e <project-dir>
      Default: true

  --python-bin BIN
      Python executable used to create the virtual environment.
      Default: python3

  --ssh-options "OPTIONS"
      Extra options passed to ssh.
      Example: --ssh-options "-o StrictHostKeyChecking=no"
      Default: empty

  --max-parallel-installs N
      Maximum number of target nodes installed in parallel.
      Default: 4

  --output-dir DIR
      Base directory for local setup artifacts.
      If provided, default setup log root becomes DIR/setup_logs.
      Explicit --setup-log-root overrides this default.

  --setup-log-root DIR
      Local directory where per-target setup logs are written.
      Default: setup_logs

  --run-id ID
      Optional explicit run id for setup logs.
      Default: current timestamp

  -h, --help
      Show this help message.

Examples from the project root:

  Local installation without recloning the current working copy:
    bash scripts/setup/setup_remote_metacsfl_node.sh \
      --nodes-file scripts/nodes.local.txt \
      --remote-project-dir "$PWD" \
      --force-reclone false

  Grid'5000 installation using public HTTPS repository:
    bash scripts/setup/setup_remote_metacsfl_node.sh \
      --nodes-file scripts/nodes.g5k.txt \
      --remote-project-dir /root/metacs-fl

  Grid'5000 installation using private GitHub repository and token prompt:
    bash scripts/setup/setup_remote_metacsfl_node.sh \
      --nodes-file scripts/nodes.g5k.txt \
      --remote-project-dir /root/metacs-fl \
      --repo-auth token \
      --prompt-github-token true

  Grid'5000 installation with 8 parallel installs:
    bash scripts/setup/setup_remote_metacsfl_node.sh \
      --nodes-file scripts/nodes.g5k.txt \
      --remote-project-dir /root/metacs-fl \
      --max-parallel-installs 8
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

  s="${s//$'\r'/}"
  s="${s//$'\xef\xbb\xbf'/}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"

  printf '%s' "${s}"
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

sanitize_name() {
  echo "$1" | sed 's/[^A-Za-z0-9_.-]/_/g'
}

shell_quote() {
  printf "%q" "$1"
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

POWERJOULAR_VERSION="1.1.0"

METACS_FL_REPO_URL="https://github.com/alan-lira/metacs-fl.git"
METACS_FL_BRANCH="main"

REPO_AUTH="none"
PROMPT_GITHUB_TOKEN="true"
GITHUB_TOKEN_ENV="GITHUB_TOKEN"
GITHUB_TOKEN_VALUE=""

REMOTE_PROJECT_DIR="/root/metacs-fl"

FORCE_RECLONE="true"
INSTALL_POWERJOULAR="true"
APPEND_VENV_TO_BASHRC="true"
INSTALL_METACSFL_PACKAGE="true"

PYTHON_BIN="python3"

SSH_OPTIONS=""

MAX_PARALLEL_INSTALLS="4"
SETUP_LOG_ROOT="setup_logs"
OUTPUT_DIR=""
SETUP_LOG_ROOT_EXPLICIT="false"
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

    --repo-url)
      require_value "$1" "${2:-}"
      METACS_FL_REPO_URL="$2"
      shift 2
      ;;

    --branch)
      require_value "$1" "${2:-}"
      METACS_FL_BRANCH="$2"
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

    --remote-project-dir)
      require_value "$1" "${2:-}"
      REMOTE_PROJECT_DIR="$2"
      shift 2
      ;;

    --powerjoular-version)
      require_value "$1" "${2:-}"
      POWERJOULAR_VERSION="$2"
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
# Validate arguments
# ----------------------------------------------------------------------

[[ -n "${NODES_FILE}" ]] || {
  echo "ERROR: --nodes-file is required" >&2
  echo >&2
  usage >&2
  exit 1
}

[[ -f "${NODES_FILE}" ]] || die "nodes file not found: ${NODES_FILE}"
[[ -n "${REMOTE_PROJECT_DIR}" ]] || die "--remote-project-dir cannot be empty"

case "${REPO_AUTH}" in
  none|token|ssh)
    ;;
  *)
    die "--repo-auth must be one of: none, token, ssh"
    ;;
esac

if [[ "${PROMPT_GITHUB_TOKEN}" != "true" && "${PROMPT_GITHUB_TOKEN}" != "false" ]]; then
  die "--prompt-github-token must be true or false"
fi

if [[ "${FORCE_RECLONE}" != "true" && "${FORCE_RECLONE}" != "false" ]]; then
  die "--force-reclone must be true or false"
fi

if [[ "${INSTALL_POWERJOULAR}" != "true" && "${INSTALL_POWERJOULAR}" != "false" ]]; then
  die "--install-powerjoular must be true or false"
fi

if [[ "${APPEND_VENV_TO_BASHRC}" != "true" && "${APPEND_VENV_TO_BASHRC}" != "false" ]]; then
  die "--append-venv-to-bashrc must be true or false"
fi

if [[ "${INSTALL_METACSFL_PACKAGE}" != "true" && "${INSTALL_METACSFL_PACKAGE}" != "false" ]]; then
  die "--install-metacsfl-package must be true or false"
fi

if ! [[ "${MAX_PARALLEL_INSTALLS}" =~ ^[0-9]+$ ]]; then
  die "--max-parallel-installs must be a positive integer"
fi

if [[ "${MAX_PARALLEL_INSTALLS}" -lt 1 ]]; then
  die "--max-parallel-installs must be >= 1"
fi

if [[ "${REPO_AUTH}" == "token" ]]; then
  if [[ "${METACS_FL_REPO_URL}" != https://* ]]; then
    die "--repo-auth token requires an HTTPS --repo-url"
  fi

  GITHUB_TOKEN_VALUE="${!GITHUB_TOKEN_ENV:-}"

  if [[ -z "${GITHUB_TOKEN_VALUE}" && "${PROMPT_GITHUB_TOKEN}" == "true" ]]; then
    read -rsp "GitHub token: " GITHUB_TOKEN_VALUE
    echo
  fi

  if [[ -z "${GITHUB_TOKEN_VALUE}" ]]; then
    die "--repo-auth token requires ${GITHUB_TOKEN_ENV} to be set or --prompt-github-token true"
  fi
fi

if [[ "${REPO_AUTH}" == "ssh" ]]; then
  if [[ "${METACS_FL_REPO_URL}" != git@* && "${METACS_FL_REPO_URL}" != ssh://* ]]; then
    warn "--repo-auth ssh selected, but --repo-url does not look like an SSH URL"
    warn "Current repo URL: ${METACS_FL_REPO_URL}"
  fi
fi

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
    echo "Hint: if the mode looks correct, check hidden characters with:" >&2
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
# Build unique installation target list
# ----------------------------------------------------------------------

INSTALL_TARGETS=()

if [[ "${MODE}" == "local" ]]; then
  INSTALL_TARGETS=("local")
else
  for SSH_TARGET in "${SSH_TARGETS[@]}"; do
    if ! array_contains "${SSH_TARGET}" "${INSTALL_TARGETS[@]}"; then
      INSTALL_TARGETS+=("${SSH_TARGET}")
    fi
  done
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR%/}"

  if [[ "${SETUP_LOG_ROOT_EXPLICIT}" != "true" ]]; then
    SETUP_LOG_ROOT="${OUTPUT_DIR}/setup_logs"
  fi
fi

SETUP_LOG_DIR="${SETUP_LOG_ROOT}/${RUN_ID}"
mkdir -p "${SETUP_LOG_DIR}"

echo "[LOCAL] Setup mode: ${MODE}"
echo "[LOCAL] Nodes file: ${NODES_FILE}"
echo "[LOCAL] Repository: ${METACS_FL_REPO_URL}"
echo "[LOCAL] Branch: ${METACS_FL_BRANCH}"
echo "[LOCAL] Repository auth: ${REPO_AUTH}"
echo "[LOCAL] Project directory on targets: ${REMOTE_PROJECT_DIR}"
echo "[LOCAL] Force reclone: ${FORCE_RECLONE}"
echo "[LOCAL] Install PowerJoular: ${INSTALL_POWERJOULAR}"
echo "[LOCAL] Append venv activation to .bashrc: ${APPEND_VENV_TO_BASHRC}"
echo "[LOCAL] Install MetaCS-FL package: ${INSTALL_METACSFL_PACKAGE}"
echo "[LOCAL] Max parallel installs: ${MAX_PARALLEL_INSTALLS}"
echo "[LOCAL] Setup log directory: ${SETUP_LOG_DIR}"
echo

echo "[LOCAL] Installation targets:"
printf "  %s\n" "${INSTALL_TARGETS[@]}"
echo

# ----------------------------------------------------------------------
# Installation payload sender
# ----------------------------------------------------------------------

send_install_payload() {
  local POWERJOULAR_VERSION_ARG="$1"
  local METACS_FL_REPO_URL_ARG="$2"
  local METACS_FL_BRANCH_ARG="$3"
  local PROJECT_DIR_ARG="$4"
  local FORCE_RECLONE_ARG="$5"
  local INSTALL_POWERJOULAR_ARG="$6"
  local APPEND_VENV_TO_BASHRC_ARG="$7"
  local PYTHON_BIN_ARG="$8"
  local REPO_AUTH_ARG="$9"
  local INSTALL_METACSFL_PACKAGE_ARG="${10}"

  local TOKEN_ASSIGNMENT=""
  if [[ "${REPO_AUTH_ARG}" == "token" ]]; then
    TOKEN_ASSIGNMENT="GITHUB_TOKEN_VALUE=$(shell_quote "${GITHUB_TOKEN_VALUE}")"
  else
    TOKEN_ASSIGNMENT="GITHUB_TOKEN_VALUE=''"
  fi

  cat <<REMOTE_SCRIPT
${TOKEN_ASSIGNMENT}
set -euo pipefail

POWERJOULAR_VERSION=$(shell_quote "${POWERJOULAR_VERSION_ARG}")
METACS_FL_REPO_URL=$(shell_quote "${METACS_FL_REPO_URL_ARG}")
METACS_FL_BRANCH=$(shell_quote "${METACS_FL_BRANCH_ARG}")
PROJECT_DIR=$(shell_quote "${PROJECT_DIR_ARG}")
FORCE_RECLONE=$(shell_quote "${FORCE_RECLONE_ARG}")
INSTALL_POWERJOULAR=$(shell_quote "${INSTALL_POWERJOULAR_ARG}")
APPEND_VENV_TO_BASHRC=$(shell_quote "${APPEND_VENV_TO_BASHRC_ARG}")
PYTHON_BIN=$(shell_quote "${PYTHON_BIN_ARG}")
REPO_AUTH=$(shell_quote "${REPO_AUTH_ARG}")
INSTALL_METACSFL_PACKAGE=$(shell_quote "${INSTALL_METACSFL_PACKAGE_ARG}")

VENV_DIR="\${PROJECT_DIR}/.venv"

if [[ "\$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

apt_install_noninteractive() {
  if [[ -n "\${SUDO}" ]]; then
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "\$@"
  else
    env DEBIAN_FRONTEND=noninteractive apt-get install -y "\$@"
  fi
}

apt_fix_noninteractive() {
  if [[ -n "\${SUDO}" ]]; then
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -f -y
  else
    env DEBIAN_FRONTEND=noninteractive apt-get install -f -y
  fi
}

cleanup_git_remote_url() {
  if [[ "\${REPO_AUTH}" == "token" && -d "\${PROJECT_DIR}/.git" ]]; then
    git -C "\${PROJECT_DIR}" remote set-url origin "\${METACS_FL_REPO_URL}" >/dev/null 2>&1 || true
  fi
}

trap cleanup_git_remote_url EXIT

make_authenticated_repo_url() {
  if [[ "\${REPO_AUTH}" == "token" ]]; then
    if [[ -z "\${GITHUB_TOKEN_VALUE}" ]]; then
      echo "ERROR: missing GitHub token for token authentication" >&2
      exit 1
    fi

    if [[ "\${METACS_FL_REPO_URL}" != https://* ]]; then
      echo "ERROR: token authentication requires an HTTPS repository URL" >&2
      exit 1
    fi

    printf 'https://x-access-token:%s@%s' "\${GITHUB_TOKEN_VALUE}" "\${METACS_FL_REPO_URL#https://}"
  else
    printf '%s' "\${METACS_FL_REPO_URL}"
  fi
}

echo
echo "======================================================================"
echo "MetaCS-FL node installation"
echo "Host: \$(hostname)"
echo "User: \$(whoami)"
echo "Project directory: \${PROJECT_DIR}"
echo "Repository: \${METACS_FL_REPO_URL}"
echo "Branch: \${METACS_FL_BRANCH}"
echo "Repository auth: \${REPO_AUTH}"
echo "Install MetaCS-FL package: \${INSTALL_METACSFL_PACKAGE}"
echo "======================================================================"
echo

# ----------------------------------------------------------------------
# System packages
# ----------------------------------------------------------------------

if command -v apt-get >/dev/null 2>&1; then
  \${SUDO} apt-get update

  apt_install_noninteractive \\
    git \\
    stress-ng \\
    cpufrequtils \\
    python3-venv \\
    python3-pip \\
    python3-tk \\
    wget \\
    rsync \\
    openssh-client \\
    build-essential

  apt_install_noninteractive \\
    linux-tools-common \\
    "linux-tools-\$(uname -r)" || true

  echo "iperf3 iperf3/daemon boolean true" | \${SUDO} debconf-set-selections || true

  apt_install_noninteractive iperf3 || true

  if command -v systemctl >/dev/null 2>&1; then
    \${SUDO} systemctl enable iperf3 || true
    \${SUDO} systemctl start iperf3 || true
  fi
else
  echo "WARNING: apt-get not found. Skipping system package installation."
fi

# ----------------------------------------------------------------------
# Print versions
# ----------------------------------------------------------------------

echo
echo "Python version:"
\${PYTHON_BIN} --version

echo
echo "pip version:"
if command -v pip3 >/dev/null 2>&1; then
  pip3 --version
else
  echo "pip3 not found"
fi

# ----------------------------------------------------------------------
# Install PowerJoular
# ----------------------------------------------------------------------

if [[ "\${INSTALL_POWERJOULAR}" == "true" ]]; then
  if command -v powerjoular >/dev/null 2>&1; then
    echo
    echo "PowerJoular already installed: \$(powerjoular -v || true)"
  else
    ARCH="\$(dpkg --print-architecture 2>/dev/null || echo unknown)"

    if [[ "\${ARCH}" != "amd64" ]]; then
      echo
      echo "WARNING: PowerJoular .deb installation is configured for amd64, but this node is '\${ARCH}'."
      echo "Skipping PowerJoular installation."
    else
      echo
      echo "Installing PowerJoular \${POWERJOULAR_VERSION}..."

      TMP_DEB="/tmp/powerjoular_\${POWERJOULAR_VERSION}_amd64.deb"

      wget -O "\${TMP_DEB}" \\
        "https://github.com/joular/powerjoular/releases/download/\${POWERJOULAR_VERSION}/powerjoular_\${POWERJOULAR_VERSION}_amd64.deb"

      \${SUDO} dpkg -i "\${TMP_DEB}" || {
        echo "PowerJoular dpkg install failed. Trying apt-get -f install..."
        apt_fix_noninteractive
        \${SUDO} dpkg -i "\${TMP_DEB}"
      }

      rm -f "\${TMP_DEB}"

      echo "PowerJoular version installed: \$(powerjoular -v || true)"
    fi
  fi
else
  echo
  echo "Skipping PowerJoular installation because INSTALL_POWERJOULAR=false."
fi

# ----------------------------------------------------------------------
# Clone or update MetaCS-FL repository
# ----------------------------------------------------------------------

echo
echo "Preparing MetaCS-FL repository..."

AUTH_REPO_URL="\$(make_authenticated_repo_url)"

if [[ -d "\${PROJECT_DIR}/.git" && "\${FORCE_RECLONE}" == "false" ]]; then
  echo "Existing repository found. Updating it..."

  cd "\${PROJECT_DIR}"

  if [[ "\${REPO_AUTH}" == "token" ]]; then
    git remote set-url origin "\${AUTH_REPO_URL}"
  else
    git remote set-url origin "\${METACS_FL_REPO_URL}" || true
  fi

  git fetch origin
  git checkout "\${METACS_FL_BRANCH}"
  git pull origin "\${METACS_FL_BRANCH}"

  git remote set-url origin "\${METACS_FL_REPO_URL}"
else
  if [[ -e "\${PROJECT_DIR}" ]]; then
    echo "Removing existing project directory: \${PROJECT_DIR}"
    rm -rf "\${PROJECT_DIR}"
  fi

  mkdir -p "\$(dirname "\${PROJECT_DIR}")"

  git clone "\${AUTH_REPO_URL}" -b "\${METACS_FL_BRANCH}" "\${PROJECT_DIR}"

  git -C "\${PROJECT_DIR}" remote set-url origin "\${METACS_FL_REPO_URL}"
fi

if [[ -f "\${PROJECT_DIR}/mlc/mlc" ]]; then
  chmod +x "\${PROJECT_DIR}/mlc/mlc" || \${SUDO} chmod +x "\${PROJECT_DIR}/mlc/mlc"
else
  echo "WARNING: \${PROJECT_DIR}/mlc/mlc not found."
fi

# ----------------------------------------------------------------------
# Create Python virtual environment inside the project
# ----------------------------------------------------------------------

echo
echo "Creating Python virtual environment..."

rm -rf "\${VENV_DIR}"

\${PYTHON_BIN} -m venv "\${VENV_DIR}"

# shellcheck disable=SC1090
source "\${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

if [[ -f "\${PROJECT_DIR}/requirements.txt" ]]; then
  pip3 install -r "\${PROJECT_DIR}/requirements.txt"
else
  echo "WARNING: requirements.txt not found at \${PROJECT_DIR}/requirements.txt"
fi

# ----------------------------------------------------------------------
# Install MetaCS-FL package
# ----------------------------------------------------------------------

if [[ "\${INSTALL_METACSFL_PACKAGE}" == "true" ]]; then
  echo
  echo "Installing MetaCS-FL package in editable mode..."

  if [[ -f "\${PROJECT_DIR}/pyproject.toml" || -f "\${PROJECT_DIR}/setup.py" ]]; then
    pip3 install -e "\${PROJECT_DIR}"
  else
    echo "WARNING: neither pyproject.toml nor setup.py found in \${PROJECT_DIR}."
    echo "WARNING: skipping pip3 install -e \${PROJECT_DIR}."
  fi
else
  echo
  echo "Skipping MetaCS-FL package installation because INSTALL_METACSFL_PACKAGE=false."
fi

# ----------------------------------------------------------------------
# Optional shell convenience
# ----------------------------------------------------------------------

if [[ "\${APPEND_VENV_TO_BASHRC}" == "true" ]]; then
  BASHRC_LINE="source \${VENV_DIR}/bin/activate"

  touch "\${HOME}/.bashrc"

  if ! grep -Fxq "\${BASHRC_LINE}" "\${HOME}/.bashrc"; then
    echo "\${BASHRC_LINE}" >> "\${HOME}/.bashrc"
    echo "Added venv activation to \${HOME}/.bashrc"
  else
    echo "Venv activation already present in \${HOME}/.bashrc"
  fi
else
  echo "Skipping .bashrc modification because APPEND_VENV_TO_BASHRC=false."
fi

# ----------------------------------------------------------------------
# Final checks
# ----------------------------------------------------------------------

echo
echo "Installation completed successfully."
echo "Project directory: \${PROJECT_DIR}"
echo "Virtual environment: \${VENV_DIR}"
echo

echo "To activate manually:"
echo "source \${VENV_DIR}/bin/activate"
echo

echo "Checking main files:"

if [[ -f "\${PROJECT_DIR}/main.py" ]]; then
  ls -lh "\${PROJECT_DIR}/main.py"
else
  echo "WARNING: \${PROJECT_DIR}/main.py not found."
fi

if [[ -f "\${PROJECT_DIR}/metacs_fl/flower_executor/config/flower_executor.cfg" ]]; then
  ls -lh "\${PROJECT_DIR}/metacs_fl/flower_executor/config/flower_executor.cfg"
else
  echo "WARNING: default flower_executor.cfg not found."
fi

echo
echo "Python in venv:"
which python3
python3 --version

echo
echo "Checking MetaCS-FL package import:"
python3 - <<'PY'
try:
    import metacs_fl
    print("metacs_fl import: OK")
except Exception as exc:
    print(f"metacs_fl import: FAILED ({exc})")
    raise
PY

echo
echo "Git remote origin after cleanup:"
git -C "\${PROJECT_DIR}" remote get-url origin || true

echo
echo "Done on host: \$(hostname)"
REMOTE_SCRIPT
}

run_install_payload_local() {
  send_install_payload \
    "${POWERJOULAR_VERSION}" \
    "${METACS_FL_REPO_URL}" \
    "${METACS_FL_BRANCH}" \
    "${REMOTE_PROJECT_DIR}" \
    "${FORCE_RECLONE}" \
    "${INSTALL_POWERJOULAR}" \
    "${APPEND_VENV_TO_BASHRC}" \
    "${PYTHON_BIN}" \
    "${REPO_AUTH}" \
    "${INSTALL_METACSFL_PACKAGE}" | bash
}

run_install_payload_remote() {
  local SSH_TARGET="$1"

  # shellcheck disable=SC2086
  send_install_payload \
    "${POWERJOULAR_VERSION}" \
    "${METACS_FL_REPO_URL}" \
    "${METACS_FL_BRANCH}" \
    "${REMOTE_PROJECT_DIR}" \
    "${FORCE_RECLONE}" \
    "${INSTALL_POWERJOULAR}" \
    "${APPEND_VENV_TO_BASHRC}" \
    "${PYTHON_BIN}" \
    "${REPO_AUTH}" \
    "${INSTALL_METACSFL_PACKAGE}" | ssh ${SSH_OPTIONS} "${SSH_TARGET}" "bash -s"
}

run_install_for_target() {
  local TARGET="$1"

  if [[ "${TARGET}" == "local" ]]; then
    run_install_payload_local
  else
    run_install_payload_remote "${TARGET}"
  fi
}

# ----------------------------------------------------------------------
# Execute installation in parallel
# ----------------------------------------------------------------------

declare -A PID_TO_TARGET=()
declare -A PID_TO_LOG=()

PIDS=()

FAILED=0

for TARGET in "${INSTALL_TARGETS[@]}"; do
  wait_for_available_slot "${MAX_PARALLEL_INSTALLS}"

  SAFE_TARGET="$(sanitize_name "${TARGET}")"
  TARGET_LOG="${SETUP_LOG_DIR}/${SAFE_TARGET}.log"

  echo "[LOCAL] Starting installation on target: ${TARGET}"
  echo "[LOCAL] Log: ${TARGET_LOG}"

  (
    set -euo pipefail

    echo "======================================================================"
    echo "[LOCAL] Starting installation on target: ${TARGET}"
    echo "======================================================================"

    run_install_for_target "${TARGET}"

    echo
    echo "======================================================================"
    echo "[LOCAL] Installation finished successfully on target: ${TARGET}"
    echo "======================================================================"
  ) > "${TARGET_LOG}" 2>&1 &

  PID="$!"
  PIDS+=("${PID}")
  PID_TO_TARGET["${PID}"]="${TARGET}"
  PID_TO_LOG["${PID}"]="${TARGET_LOG}"
done

echo
echo "[LOCAL] All install jobs have been launched."
echo "[LOCAL] Logs are in: ${SETUP_LOG_DIR}"
echo

for PID in "${PIDS[@]}"; do
  TARGET="${PID_TO_TARGET[${PID}]}"
  TARGET_LOG="${PID_TO_LOG[${PID}]}"

  if wait "${PID}"; then
    echo "[LOCAL] Installation succeeded on ${TARGET}."
  else
    echo "[LOCAL] ERROR: installation failed on ${TARGET}."
    echo "[LOCAL] Check log: ${TARGET_LOG}"
    FAILED=1
  fi
done

echo
echo "======================================================================"
if [[ "${FAILED}" -eq 0 ]]; then
  echo "[LOCAL] Installation completed successfully on all targets."
else
  echo "[LOCAL] Installation failed on at least one target."
  echo "[LOCAL] Logs are in: ${SETUP_LOG_DIR}"
fi
echo "======================================================================"

if [[ "${FAILED}" -ne 0 ]]; then
  exit 1
fi

exit 0
