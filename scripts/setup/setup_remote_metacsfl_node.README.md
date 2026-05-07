# MetaCS-FL Multi-Node Setup Script

This document explains how to use:

```txt
scripts/setup/setup_remote_metacsfl_node.sh
```

to install MetaCS-FL on one or more target machines.

The setup script supports:

- local installation;
- generic remote installation;
- Grid'5000 installation;
- private GitHub repositories using a token;
- SSH-based repository cloning;
- parallel installation across multiple target nodes;
- editable installation of the `metacs_fl` Python package;
- per-target setup logs.

The script uses the same node-description files used by the distributed execution launcher:

```txt
scripts/nodes.local.txt
scripts/nodes.remote.txt
scripts/nodes.g5k.txt
```

All examples in this document assume commands are executed from the project root folder.

---

## 1. Script location

The setup script is located at:

```txt
scripts/setup/setup_remote_metacsfl_node.sh
```

Example call:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl
```

---

## 2. Main idea

The setup script is a coordinator.

It reads a node file, extracts the installation targets, removes duplicate SSH targets, and installs MetaCS-FL on each unique target.

For example, if the node file contains:

```txt
mode g5k

server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2.rennes.g5k paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3.rennes.g5k paradoxe-3.rennes.grid5000.fr
```

then the setup script installs MetaCS-FL on:

```txt
root@paradoxe-1.rennes.g5k
root@paradoxe-2.rennes.g5k
root@paradoxe-3.rennes.g5k
```

If the same SSH target appears more than once in the node file, it is installed only once.

The third column, the runtime host, is not used for installation. It is kept in the node file because the same file is also used by the distributed execution launcher.

---

## 3. Node file format

Each node file uses the shared format:

```txt
mode <local|remote|g5k>

<role> <ssh_target> <runtime_host>
```

Where:

| Field | Description |
|---|---|
| `mode` | Installation/execution mode. Must be `local`, `remote`, or `g5k`. |
| `role` | Process role. Must be `server` or `client`. |
| `ssh_target` | Address used by the setup script to reach the machine. Use `-` in local mode. |
| `runtime_host` | Address used later by Flower/gRPC during distributed execution. |

The setup script validates that:

- exactly one `server` entry exists;
- at least one `client` entry exists;
- `local` mode uses `-` as the SSH target;
- `remote` and `g5k` modes use real SSH targets.

The script also handles blank lines, comments, UTF-8 BOMs, and Windows CRLF line endings.

---

## 4. Example node files

### 4.1 `scripts/nodes.local.txt`

Use this when installing on the current local machine.

```txt
mode local

server - 127.0.0.1
client - 127.0.0.1
```

Alternative local version with distinct loopback aliases:

```txt
mode local

server - 127.0.0.1
client - 127.0.0.2
```

---

### 4.2 `scripts/nodes.remote.txt`

Use this for generic remote machines.

Using hostnames:

```txt
mode remote

server ubuntu@server.example.com server.example.com
client ubuntu@client1.example.com client1.example.com
client ubuntu@client2.example.com client2.example.com
```

Using IP addresses:

```txt
mode remote

server ubuntu@192.168.1.10 192.168.1.10
client ubuntu@192.168.1.11 192.168.1.11
client ubuntu@192.168.1.12 192.168.1.12
```

---

### 4.3 `scripts/nodes.g5k.txt`

Use this for Grid'5000.

```txt
mode g5k

server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2.rennes.g5k paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3.rennes.g5k paradoxe-3.rennes.grid5000.fr
```

If your SSH configuration defines short aliases, you can use them as SSH targets:

```txt
mode g5k

server root@paradoxe-1 paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2 paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3 paradoxe-3.rennes.grid5000.fr
```

---

## 5. What the setup script installs

On each target machine, the script:

1. installs system packages with `apt-get`;
2. installs or starts `iperf3`;
3. optionally installs PowerJoular;
4. clones or updates the MetaCS-FL repository;
5. makes `mlc/mlc` executable if it exists;
6. creates a Python virtual environment under the project directory;
7. installs `requirements.txt`;
8. optionally installs the MetaCS-FL package in editable mode;
9. optionally appends the virtual environment activation line to `~/.bashrc`;
10. checks important files and verifies that `metacs_fl` can be imported.

The editable package installation uses:

```bash
pip3 install -e <project-dir>
```

This is enabled by default.

---

## 6. Parameters

| Parameter | Description | Default |
|---|---|---|
| `--nodes-file FILE` | Node description file. | Required |
| `--repo-url URL` | MetaCS-FL Git repository URL. | `https://github.com/alan-lira/metacs-fl.git` |
| `--branch BRANCH` | Git branch to clone or pull. | `main` |
| `--repo-auth none\|token\|ssh` | Repository authentication mode. | `none` |
| `--prompt-github-token true\|false` | Prompt locally for a GitHub token when using `--repo-auth token`. | `true` |
| `--github-token-env NAME` | Environment variable used to read the GitHub token. | `GITHUB_TOKEN` |
| `--remote-project-dir DIR` | Project directory on the target machine. | `/root/metacs-fl` |
| `--powerjoular-version VERSION` | PowerJoular version to install. | `1.1.0` |
| `--force-reclone true\|false` | Remove the project directory and clone again. | `true` |
| `--install-powerjoular true\|false` | Whether to install PowerJoular. | `true` |
| `--append-venv-to-bashrc true\|false` | Append venv activation to `~/.bashrc`. | `true` |
| `--install-metacsfl-package true\|false` | Install MetaCS-FL with `pip3 install -e`. | `true` |
| `--python-bin BIN` | Python executable used to create the venv. | `python3` |
| `--ssh-options "OPTIONS"` | Extra options passed to `ssh`. | Empty |
| `--max-parallel-installs N` | Maximum number of parallel target installations. | `4` |
| `--setup-log-root DIR` | Local directory where setup logs are written. | `setup_logs` |
| `--run-id ID` | Explicit run id for setup logs. | Timestamp |

---

## 7. Repository authentication modes

### 7.1 Public repository

For a public repository, use the default authentication mode:

```bash
--repo-auth none
```

Example:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl
```

---

### 7.2 Private repository using a GitHub token

For a private HTTPS repository, use:

```bash
--repo-auth token
```

The script can prompt locally for the token:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --repo-url https://github.com/alan-lira/metacs-fl.git \
  --branch main
```

The prompt looks like:

```txt
GitHub token:
```

The token is not placed in the command line.

The script temporarily injects the token into the clone or pull command, then resets the Git remote URL back to the normal repository URL so the token is not stored in `.git/config`.

Do not pass the token directly inside `--repo-url`.

Avoid this:

```bash
--repo-url https://ghp_xxxxxxxxx@github.com/alan-lira/metacs-fl.git
```

because it can leak through shell history, logs, or process listings.

---

### 7.3 Private repository using an environment variable

Instead of prompting, you can export a token first:

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxx"
```

Then run:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token false \
  --repo-url https://github.com/alan-lira/metacs-fl.git \
  --branch main
```

You can also use a different environment variable name:

```bash
export METACS_GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxx"
```

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token false \
  --github-token-env METACS_GITHUB_TOKEN
```

---

### 7.4 Private repository using SSH

For SSH authentication, use:

```bash
--repo-auth ssh
```

and pass an SSH repository URL:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth ssh \
  --repo-url git@github.com:alan-lira/metacs-fl.git \
  --branch main
```

This requires the target nodes to have access to GitHub through SSH, either by local keys on each node or by SSH agent forwarding.

---

## 8. Example calls

### 8.1 Local installation without recloning

Use this when testing setup locally and you do not want to delete your current working tree:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --force-reclone false \
  --max-parallel-installs 1
```

This reuses the current repository, but still removes and recreates:

```txt
$PWD/.venv
```

It then installs requirements and installs the package in editable mode.

---

### 8.2 Local clean installation

Use this if you want to remove and reclone into a local directory:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$HOME/metacs-fl" \
  --force-reclone true \
  --repo-url https://github.com/alan-lira/metacs-fl.git \
  --branch main \
  --max-parallel-installs 1
```

---

### 8.3 Generic remote installation

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.remote.txt \
  --remote-project-dir /home/ubuntu/metacs-fl \
  --repo-url https://github.com/alan-lira/metacs-fl.git \
  --branch main \
  --max-parallel-installs 4
```

---

### 8.4 Generic remote installation with SSH options

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.remote.txt \
  --remote-project-dir /home/ubuntu/metacs-fl \
  --ssh-options "-o StrictHostKeyChecking=no" \
  --max-parallel-installs 4
```

---

### 8.5 Grid'5000 installation

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-url https://github.com/alan-lira/metacs-fl.git \
  --branch main \
  --max-parallel-installs 8
```

---

### 8.6 Grid'5000 installation with private repository token

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --repo-url https://github.com/alan-lira/metacs-fl.git \
  --branch main \
  --max-parallel-installs 8
```

---

### 8.7 Grid'5000 installation without PowerJoular

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --install-powerjoular false \
  --max-parallel-installs 8
```

---

### 8.8 Grid'5000 installation without modifying `.bashrc`

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --append-venv-to-bashrc false \
  --max-parallel-installs 8
```

---

### 8.9 Grid'5000 installation without editable package installation

Use this if the repository has no `pyproject.toml` or `setup.py`, or if you only want to install dependencies:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --install-metacsfl-package false \
  --max-parallel-installs 8
```

---

### 8.10 Reuse existing repository and update it

Use this when the repository already exists on the nodes and you want to pull updates instead of deleting it:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --force-reclone false \
  --branch main \
  --max-parallel-installs 8
```

Important: `--force-reclone false` reuses or updates the existing repository, but the script still recreates:

```txt
<remote-project-dir>/.venv
```

on each target.

---

### 8.11 Explicit setup run id

Use this to make log paths predictable:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id g5k_setup_test_001 \
  --max-parallel-installs 8
```

Logs will be written under:

```txt
setup_logs/g5k_setup_test_001/
```

---

## 9. Parallel installation

The setup script installs on multiple targets in parallel.

The number of parallel installations is controlled by:

```bash
--max-parallel-installs N
```

Example:

```bash
--max-parallel-installs 8
```

If your node file has 20 unique remote nodes and `--max-parallel-installs 8`, the script launches at most 8 installations at a time.

Use a lower value if:

- the repository server throttles too many simultaneous clones;
- the cluster network is congested;
- `apt-get update` overloads the package mirror;
- the nodes have limited CPU or I/O capacity;
- multiple entries point to the same physical machine or shared image.

Use a higher value if:

- the nodes are independent;
- the network is stable;
- you want faster setup across many machines.

For local mode, use:

```bash
--max-parallel-installs 1
```

---

## 10. Setup logs

Each target gets its own local setup log.

By default, logs are written to:

```txt
setup_logs/<run_id>/
```

Example:

```txt
setup_logs/20260506_221530/root_paradoxe-1.rennes.g5k.log
setup_logs/20260506_221530/root_paradoxe-2.rennes.g5k.log
setup_logs/20260506_221530/root_paradoxe-3.rennes.g5k.log
```

The log filename is based on the sanitized SSH target. For example:

```txt
root@paradoxe-16.rennes.g5k
```

becomes:

```txt
root_paradoxe-16.rennes.g5k.log
```

You can follow logs while the setup is running:

```bash
tail -f setup_logs/<run_id>/*.log
```

If a target fails, the coordinator prints the log file to inspect:

```txt
[LOCAL] ERROR: installation failed on root@paradoxe-2.rennes.g5k.
[LOCAL] Check log: setup_logs/<run_id>/root_paradoxe-2.rennes.g5k.log
```

---

## 11. Installation details on each target

### 11.1 System packages

When `apt-get` is available, the script installs:

```txt
git
stress-ng
cpufrequtils
python3-venv
python3-pip
python3-tk
wget
rsync
openssh-client
build-essential
linux-tools-common
linux-tools-$(uname -r)
iperf3
```

The `linux-tools-$(uname -r)` installation is allowed to fail because it may not be available for every kernel.

The script uses noninteractive apt installation through:

```bash
env DEBIAN_FRONTEND=noninteractive apt-get install -y ...
```

This works both as `root` and as a non-root user with `sudo`.

---

### 11.2 PowerJoular

PowerJoular is installed by default:

```bash
--install-powerjoular true
```

The default version is:

```txt
1.1.0
```

To disable it:

```bash
--install-powerjoular false
```

PowerJoular installation is skipped automatically on non-`amd64` architectures.

---

### 11.3 Repository clone or update

If:

```bash
--force-reclone true
```

the script removes the existing project directory and clones the repository again.

If:

```bash
--force-reclone false
```

and the directory is already a Git repository, the script runs:

```bash
git fetch origin
git checkout <branch>
git pull origin <branch>
```

When token authentication is used, the script temporarily sets an authenticated remote URL for clone/pull and then restores the normal repository URL afterward.

---

### 11.4 Python virtual environment

The virtual environment is created under:

```txt
<remote-project-dir>/.venv
```

For example:

```txt
/root/metacs-fl/.venv
```

or:

```txt
/home/ubuntu/metacs-fl/.venv
```

The script recreates this virtual environment on each run.

This happens even when:

```bash
--force-reclone false
```

is used.

---

### 11.5 Python dependencies

After activating the virtual environment, the script upgrades:

```txt
pip
setuptools
wheel
```

Then it installs:

```bash
pip3 install -r requirements.txt
```

if `requirements.txt` exists.

---

### 11.6 MetaCS-FL package installation

By default, the script installs MetaCS-FL itself in editable mode:

```bash
pip3 install -e <project-dir>
```

This requires one of the following files to exist in the project directory:

```txt
pyproject.toml
setup.py
```

To disable editable package installation:

```bash
--install-metacsfl-package false
```

The script verifies the installation by running:

```python
import metacs_fl
```

If editable installation is disabled, this import check still runs. It may still pass if the project root is importable from the execution context, but for robust execution it is recommended to keep editable installation enabled when packaging metadata is available.

---

### 11.7 Shell convenience

By default, the script appends this line to the target user's `~/.bashrc`:

```bash
source <remote-project-dir>/.venv/bin/activate
```

To disable this behavior:

```bash
--append-venv-to-bashrc false
```

The script avoids appending duplicate activation lines.

---

## 12. What the script does

The coordinator performs the following steps:

1. Parses command-line arguments.
2. Reads the shared node file.
3. Validates the mode and node entries.
4. Builds a unique list of installation targets.
5. Creates a setup log directory.
6. Prompts for a GitHub token if needed.
7. Launches installation jobs in parallel.
8. Writes one log file per target.
9. Waits for all jobs to finish.
10. Reports success or failure per target.

Each target installation performs the following steps:

1. Installs system packages.
2. Starts/enables `iperf3` when possible.
3. Installs PowerJoular if enabled.
4. Clones or updates the repository.
5. Makes `mlc/mlc` executable if present.
6. Recreates the Python virtual environment.
7. Installs Python dependencies.
8. Installs MetaCS-FL in editable mode if enabled.
9. Optionally appends venv activation to `~/.bashrc`.
10. Checks `main.py`, default `flower_executor.cfg`, Python version, and `metacs_fl` import.
11. Prints the cleaned Git remote origin.

---

## 13. Recommended workflow

A typical workflow is:

1. Prepare the node file:

```txt
scripts/nodes.g5k.txt
```

2. Install MetaCS-FL on all nodes:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --branch main \
  --max-parallel-installs 8
```

3. Check setup logs:

```bash
ls setup_logs/
tail -f setup_logs/<run_id>/*.log
```

4. Launch the distributed execution:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl
```

---

## 14. Common issues

### 14.1 Git clone fails for private repository

Use token authentication:

```bash
--repo-auth token
--prompt-github-token true
```

or SSH authentication:

```bash
--repo-auth ssh
--repo-url git@github.com:alan-lira/metacs-fl.git
```

Do not include the token directly in `--repo-url`.

---

### 14.2 GitHub token is not accepted

Make sure the token has permission to read the repository.

For fine-grained GitHub tokens, the token must have access to the repository and permission to read contents.

---

### 14.3 SSH works manually but setup fails

Check the first hostname field in the node file.

For example, this line:

```txt
server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
```

requires this to work:

```bash
ssh root@paradoxe-1.rennes.g5k
```

If you use SSH aliases, write the alias in the first hostname field:

```txt
server root@paradoxe-1 paradoxe-1.rennes.grid5000.fr
```

You can also pass SSH options:

```bash
--ssh-options "-o StrictHostKeyChecking=no"
```

---

### 14.4 `apt-get` lock errors

If many installs run in parallel on the same physical machine or shared image, `apt-get` may fail because another process holds the lock.

Reduce the parallelism:

```bash
--max-parallel-installs 1
```

or rerun after the other package operation finishes.

---

### 14.5 PowerJoular installation fails

PowerJoular installation can fail if:

- the architecture is not `amd64`;
- the `.deb` package URL changed;
- dependencies cannot be resolved;
- the target environment restricts package installation.

To skip PowerJoular:

```bash
--install-powerjoular false
```

---

### 14.6 `metacs_fl import: FAILED`

If the final import check fails, verify that:

- `requirements.txt` installed successfully;
- `pyproject.toml` or `setup.py` exists;
- editable package installation was not disabled;
- the package directory is named `metacs_fl`.

You can manually inspect the target:

```bash
ssh <target>
source /root/metacs-fl/.venv/bin/activate
python3 -c "import metacs_fl; print('OK')"
```

---

### 14.7 `pip3 install -e` is skipped

The setup script only runs editable installation if one of these files exists:

```txt
pyproject.toml
setup.py
```

If neither exists, the script prints a warning and skips:

```bash
pip3 install -e <project-dir>
```

Add packaging metadata or disable the package installation step with:

```bash
--install-metacsfl-package false
```

---

### 14.8 Node file has hidden characters

The script accepts Windows CRLF line endings, but if something still looks strange, inspect the node file with:

```bash
cat -A scripts/nodes.g5k.txt
```

If you see `^M`, normalize the file with:

```bash
sed -i 's/\r$//' scripts/nodes.g5k.txt
```

---

### 14.9 Too many parallel installs

If the setup causes network or package mirror pressure, reduce:

```bash
--max-parallel-installs
```

For example:

```bash
--max-parallel-installs 2
```

For large Grid'5000 allocations, values between `4` and `10` are usually safer than launching every node at once.

---

### 14.10 Token appears in output

The setup script is designed not to print the token and to reset the Git remote URL after clone/pull.

Still, avoid running with shell tracing enabled:

```bash
set -x
```

when using token authentication.

If a failure happens during clone/pull, inspect logs carefully before sharing them.

---

### 14.11 Existing repository is reused but dependencies look stale

When using:

```bash
--force-reclone false
```

the repository is updated instead of deleted, but the virtual environment is still recreated.

If dependencies still look wrong, inspect the setup log and check that:

```bash
pip3 install -r requirements.txt
```

completed successfully.

---

## 15. Summary

Use the setup script to prepare all nodes before launching distributed experiments.

For local testing:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --force-reclone false \
  --max-parallel-installs 1
```

For Grid'5000 with a private repository:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --branch main \
  --max-parallel-installs 8
```

Then launch the distributed run with:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl
```
