# One-Experiment Distributed Pipeline

This document explains how to use:

```txt
scripts/execution/run_one_distributed_experiment.sh
```

to run one complete MetaCS-FL distributed experiment pipeline.

The pipeline executes these stages in sequence:

1. optionally install/setup MetaCS-FL on the target nodes;
2. launch the distributed Flower execution;
3. merge the gathered per-node outputs into one consolidated result folder.

It is a wrapper around the existing scripts:

```txt
scripts/setup/setup_remote_metacsfl_node.sh
scripts/execution/launch_distributed_flower.sh
scripts/post_execution/merge_distributed_outputs.py
```

The setup script supports node files, repository authentication, parallel installation, and editable package installation. The launch script supports custom Flower configs, runtime gRPC patching, per-node gathering, and parallel remote operations. The merge script consolidates the gathered `node_*` folders into one merged output directory. 

All examples below assume commands are executed from the project root folder.

---

## 1. Script location

Location:

```txt
scripts/execution/run_one_distributed_experiment.sh
```

The examples in this README use:

```txt
scripts/execution/run_one_distributed_experiment.sh
```

---

## 2. Main idea

The pipeline script avoids manually running three commands for every experiment.

Instead of doing this:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh ...
bash scripts/execution/launch_distributed_flower.sh ...
python3 scripts/post_execution/merge_distributed_outputs.py ...
```

you run one command:

```bash
bash scripts/execution/run_one_distributed_experiment.sh ...
```

The same `--run-id` is used to connect the launch and merge stages.

For example, with:

```bash
--run-id cifar10_iid_test_001
```

the launcher writes gathered outputs to:

```txt
gathered_results/cifar10_iid_test_001/
```

and the merge stage writes consolidated outputs to:

```txt
merged_results/cifar10_iid_test_001/
```

If setup is enabled, setup logs are written to:

```txt
setup_logs/cifar10_iid_test_001_setup/
```

The launch logs are written to:

```txt
distributed_launch_logs/cifar10_iid_test_001/
```

---

## 3. When to use this script

Use this script when you want to run a complete experiment workflow for one experiment configuration.

Typical cases:

- fresh Grid'5000 allocation where setup is needed before execution;
- repeated experiment where nodes are already installed and only launch + merge are needed;
- local test of one experiment configuration;
- generic remote execution with post-processing;
- controlled run with a fixed `--run-id` for reproducibility.

---

## 4. Expected node files

The pipeline uses the same node-description files as the setup and launch scripts:

```txt
scripts/nodes.local.txt
scripts/nodes.remote.txt
scripts/nodes.g5k.txt
```

Node file format:

```txt
mode <local|remote|g5k>

<role> <ssh_target> <runtime_host>
```

Example Grid'5000 node file:

```txt
mode g5k

server root@paradoxe-16.rennes.g5k paradoxe-16.rennes.grid5000.fr
client root@paradoxe-17.rennes.g5k paradoxe-17.rennes.grid5000.fr
client root@paradoxe-18.rennes.g5k paradoxe-18.rennes.grid5000.fr
```

If your executor identifies the current node using short hostnames, use short runtime hostnames:

```txt
mode g5k

server root@paradoxe-16.rennes.g5k paradoxe-16
client root@paradoxe-17.rennes.g5k paradoxe-17
client root@paradoxe-18.rennes.g5k paradoxe-18
```

---

## 5. Command-line usage

Basic form:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --nodes-file <nodes-file> \
  [options]
```

The only required argument is:

```bash
--nodes-file
```

However, in practice, you should almost always provide:

```bash
--remote-project-dir
--run-id
```

and, for custom experiments:

```bash
--custom-flower-executor-cfg
--custom-flower-server-cfg
--custom-flower-client-cfg
```

---

## 6. Parameters

### 6.1 Main parameters

| Parameter | Description | Default |
|---|---|---|
| `--nodes-file FILE` | Node description file shared by setup and launch scripts. | Required |
| `--run-id ID` | Run identifier used for launch and merge. | Timestamp |
| `--install true\|false` | Whether to run setup before launching. | `false` |
| `--remote-project-dir DIR` | Project directory on target machines. | `/root/metacs-fl` |
| `--remote-venv-activate FILE` | Virtual environment activation script on target machines. | `<remote-project-dir>/.venv/bin/activate` |
| `--output-dir DIR` | Base directory for local pipeline artifacts. If provided, default roots become `DIR/setup_logs`, `DIR/distributed_launch_logs`, `DIR/gathered_results`, and `DIR/merged_results`. Explicit root-specific options override this default. | Not set |

---

### 6.2 Script path parameters

| Parameter | Description | Default |
|---|---|---|
| `--setup-script FILE` | Path to setup script. | `scripts/setup/setup_remote_metacsfl_node.sh` |
| `--launch-script FILE` | Path to distributed launch script. | `scripts/execution/launch_distributed_flower.sh` |
| `--merge-script FILE` | Path to merge script. | `scripts/post_execution/merge_distributed_outputs.py` |

These are useful if you place the wrapper in a different directory or want to test modified versions of the setup, launch, or merge scripts.

---

### 6.3 Setup/install parameters

These options are used only when:

```bash
--install true
```

| Parameter | Description | Default |
|---|---|---|
| `--repo-url URL` | Repository URL passed to setup script. | `https://github.com/alan-lira/metacs-fl.git` |
| `--branch BRANCH` | Git branch passed to setup script. | `main` |
| `--repo-auth none\|token\|ssh` | Repository authentication mode. | `none` |
| `--prompt-github-token true\|false` | Whether setup prompts for a GitHub token. | `true` |
| `--github-token-env NAME` | Environment variable used for the GitHub token. | `GITHUB_TOKEN` |
| `--force-reclone true\|false` | Whether setup reclones the repository. | `true` |
| `--install-powerjoular true\|false` | Whether setup installs PowerJoular. | `true` |
| `--append-venv-to-bashrc true\|false` | Whether setup appends venv activation to `.bashrc`. | `true` |
| `--install-metacsfl-package true\|false` | Whether setup installs MetaCS-FL with `pip3 install -e`. | `true` |
| `--max-parallel-installs N` | Maximum number of nodes installed in parallel. | `4` |
| `--setup-log-root DIR` | Root directory for setup logs. | `setup_logs` |

When setup is enabled, the wrapper passes:

```bash
--run-id <pipeline-run-id>_setup
```

to the setup script. Therefore, setup logs are written to:

```txt
<setup-log-root>/<run-id>_setup/
```

---

### 6.4 Launch parameters

| Parameter | Description | Default |
|---|---|---|
| `--custom-flower-executor-cfg FILE` | Custom `flower_executor.cfg` passed to launch script. | Not set |
| `--custom-flower-server-cfg FILE` | Custom `flower_server.cfg` passed to launch script. | Not set |
| `--custom-flower-client-cfg FILE` | Custom `flower_client.cfg` passed to launch script. | Not set |
| `--remote-flower-executor-cfg FILE` | Remote `flower_executor.cfg` passed to launch script. | Launch-script default |
| `--remote-config-file FILE` | Alias for `--remote-flower-executor-cfg`. | Launch-script default |
| `--remote-flower-server-cfg FILE` | Remote `flower_server.cfg` passed to launch script. | Not set |
| `--remote-flower-client-cfg FILE` | Remote `flower_client.cfg` passed to launch script. | Not set |
| `--patch-executor-config true\|false` | Whether launcher patches runtime executor config paths. | `true` |
| `--patch-grpc-config true\|false` | Whether launcher patches runtime gRPC settings. | `true` |
| `--server-port PORT` | Fallback gRPC server port. | `8080` |
| `--repetitions N` | Number of repetitions passed to launch script. | `1` |
| `--action ACTION` | `main.py` action passed to launch script. | `execute_fl_with_flower` |
| `--gather-outputs true\|false` | Value passed to launch script. | `false` |
| `--max-parallel-remote-ops N` | Remote validation/copy/collection parallelism. | `8` |
| `--local-log-root DIR` | Root directory for launch logs. | `distributed_launch_logs` |
| `--local-gather-root DIR` | Root directory for gathered results. | `gathered_results` |

The wrapper passes the same `--run-id` to the launch script. Therefore, launch outputs are written under:

```txt
<local-log-root>/<run-id>/
<local-gather-root>/<run-id>/
```

---

### 6.5 Merge parameters

| Parameter | Description | Default |
|---|---|---|
| `--merge-output-root DIR` | Root directory for merged outputs. | `merged_results` |
| `--node-prefix PREFIX` | Node folder prefix passed to merge script. | `node_` |
| `--add-source-node true\|false` | Add `source_node` column to merged CSVs. | `false` |
| `--deduplicate-rows true\|false` | Remove exact duplicate merged CSV rows. | `false` |
| `--clean-merge-output true\|false` | Delete existing merged output before writing. | `true` |

The wrapper merges:

```txt
<local-gather-root>/<run-id>/
```

into:

```txt
<merge-output-root>/<run-id>/
```

By default:

```txt
gathered_results/<run-id>/
merged_results/<run-id>/
```

---

### 6.6 Common parameters

| Parameter | Description | Default |
|---|---|---|
| `--python-bin BIN` | Python executable passed to setup, launch, and merge. | `python3` |
| `--ssh-options "OPTIONS"` | SSH options passed to setup and launch. | Empty |
| `--rsync-options "OPTIONS"` | Rsync options passed to launch. | `-az` |

---

### 6.7 Extra passthrough parameters

The wrapper also supports extra passthrough arguments.

| Parameter | Description |
|---|---|
| `--setup-arg ARG` | Add one extra argument to the setup script. Can be repeated. |
| `--launch-arg ARG` | Add one extra argument to the launch script. Can be repeated. |
| `--merge-arg ARG` | Add one extra argument to the merge script. Can be repeated. |

Use one `--setup-arg`, `--launch-arg`, or `--merge-arg` per token.

Example:

```bash
--launch-arg --remote-hostfile-dir \
--launch-arg /tmp/metacs_runtime
```

---

## 7. Output directories

For a run id:

```txt
cifar10_iid_test_001
```

the pipeline creates or uses:

```txt
<local-log-root>/cifar10_iid_test_001/
<local-gather-root>/cifar10_iid_test_001/
<merge-output-root>/cifar10_iid_test_001/
```

By default, these are:

```txt
distributed_launch_logs/cifar10_iid_test_001/
gathered_results/cifar10_iid_test_001/
merged_results/cifar10_iid_test_001/
```

If setup is enabled, it also creates:

```txt
<setup-log-root>/cifar10_iid_test_001_setup/
```

If `--output-dir DIR` is provided and no root-specific overrides are used, these become:

```txt
DIR/setup_logs/cifar10_iid_test_001_setup/
DIR/distributed_launch_logs/cifar10_iid_test_001/
DIR/gathered_results/cifar10_iid_test_001/
DIR/merged_results/cifar10_iid_test_001/
```

The final summary prints all relevant output paths.

---

## 8. Stage behavior

### 8.1 Stage 1: setup/install

This stage runs only when:

```bash
--install true
```

It calls:

```txt
scripts/setup/setup_remote_metacsfl_node.sh
```

The setup stage installs MetaCS-FL on each unique target from the node file.

It can:

- clone or update the repository;
- authenticate with GitHub using token or SSH;
- install system dependencies;
- install PowerJoular;
- create the virtual environment;
- install requirements;
- install the MetaCS-FL package in editable mode;
- run installations in parallel.

If setup fails, the pipeline stops.

---

### 8.2 Stage 2: distributed launch

This stage always runs.

It calls:

```txt
scripts/execution/launch_distributed_flower.sh
```

The launch stage:

- generates a runtime hostfile from the node file;
- stages custom Flower configs if provided;
- patches runtime executor config paths;
- patches runtime gRPC settings;
- verifies remote paths;
- copies runtime files to remote nodes;
- launches one distributed process per node entry, except in local mode where it launches one local controller process;
- gathers per-node outputs into `gathered_results/<run-id>/`.

If launch fails, the launcher still attempts to pull available results, but the pipeline stops if the launch command exits with an error.

---

### 8.3 Stage 3: merge gathered outputs

This stage always runs after a successful launch.

It calls:

```txt
scripts/post_execution/merge_distributed_outputs.py
```

The merge stage reads:

```txt
gathered_results/<run-id>/
```

and writes:

```txt
merged_results/<run-id>/
```

The merge script:

- discovers `node_*` folders;
- groups files by relative path;
- appends CSV files with matching headers;
- sorts by round-like columns when present;
- copies metadata;
- handles conflicts safely under `_merge_conflicts/`;
- writes `merge_report.json`.

---

## 9. Example calls

### 9.1 Launch + merge only

Use this when the nodes are already installed:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id cifar10_iid_test_001 \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --max-parallel-remote-ops 8 \
  --repetitions 1 \
  --clean-merge-output true
```

This produces:

```txt
distributed_launch_logs/cifar10_iid_test_001/
gathered_results/cifar10_iid_test_001/
merged_results/cifar10_iid_test_001/
```

---

### 9.2 Install + launch + merge

Use this for a fresh allocation:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install true \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --branch main \
  --force-reclone true \
  --max-parallel-installs 8 \
  --run-id cifar10_iid_test_001 \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --max-parallel-remote-ops 8 \
  --repetitions 1 \
  --clean-merge-output true
```

This produces:

```txt
setup_logs/cifar10_iid_test_001_setup/
distributed_launch_logs/cifar10_iid_test_001/
gathered_results/cifar10_iid_test_001/
merged_results/cifar10_iid_test_001/
```

---

### 9.3 Local test

Use this to test the full flow locally without installing:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --run-id local_test_001 \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --max-parallel-remote-ops 1 \
  --repetitions 1
```

Outputs:

```txt
distributed_launch_logs/local_test_001/
gathered_results/local_test_001/
merged_results/local_test_001/
```

---

### 9.4 Local test with setup

Use this if you want the wrapper to recreate the local venv and install the package:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install true \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --force-reclone false \
  --install-powerjoular false \
  --max-parallel-installs 1 \
  --run-id local_setup_test_001 \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg
```

Important: even with `--force-reclone false`, the setup script recreates:

```txt
$PWD/.venv
```

---

### 9.5 Grid'5000 with private repository token

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install true \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --branch main \
  --force-reclone true \
  --max-parallel-installs 8 \
  --run-id g5k_private_test_001 \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --max-parallel-remote-ops 8
```

---

### 9.6 Grid'5000 using SSH repository authentication

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install true \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth ssh \
  --repo-url git@github.com:alan-lira/metacs-fl.git \
  --branch main \
  --force-reclone true \
  --max-parallel-installs 8 \
  --run-id g5k_ssh_test_001 \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg
```

---

### 9.7 Use existing remote repository and only update it

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install true \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --force-reclone false \
  --branch main \
  --max-parallel-installs 8 \
  --run-id g5k_update_test_001 \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg
```

The repository is reused/updated, but the setup script still recreates the virtual environment.

---

### 9.8 Add source node column during merge

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id cifar10_debug_source_node \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --add-source-node true \
  --clean-merge-output true
```

This adds a `source_node` column to merged CSV files.

---

### 9.9 Deduplicate merged rows

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id cifar10_dedup_test \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --deduplicate-rows true \
  --clean-merge-output true
```

---

### 9.10 Custom output roots

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id cifar10_custom_roots \
  --local-log-root logs/distributed \
  --local-gather-root results/gathered \
  --merge-output-root results/merged \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg
```

This produces:

```txt
logs/distributed/cifar10_custom_roots/
results/gathered/cifar10_custom_roots/
results/merged/cifar10_custom_roots/
```

---

### 9.11 Custom base output directory

Use `--output-dir` when you want all local pipeline artifacts under the same base directory:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id cifar10_output_dir_test \
  --output-dir /mnt/d/results \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg
```

This produces:

```txt
/mnt/d/results/distributed_launch_logs/cifar10_output_dir_test/
/mnt/d/results/gathered_results/cifar10_output_dir_test/
/mnt/d/results/merged_results/cifar10_output_dir_test/
```

If setup is enabled, setup logs are written to:

```txt
/mnt/d/results/setup_logs/cifar10_output_dir_test_setup/
```

---

## 10. Recommended workflow

### First run on a fresh Grid'5000 allocation

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install true \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --branch main \
  --force-reclone true \
  --max-parallel-installs 8 \
  --run-id my_experiment_001 \
  --custom-flower-executor-cfg experiments/my_experiment/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_experiment/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_experiment/flower_client.cfg \
  --max-parallel-remote-ops 8 \
  --clean-merge-output true
```

### Subsequent run on already prepared nodes

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id my_experiment_002 \
  --custom-flower-executor-cfg experiments/my_experiment/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_experiment/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_experiment/flower_client.cfg \
  --max-parallel-remote-ops 8 \
  --clean-merge-output true
```

---

## 11. Logs and reports

For a run id:

```txt
my_experiment_001
```

setup logs, if setup is enabled:

```txt
setup_logs/my_experiment_001_setup/
```

launch logs:

```txt
distributed_launch_logs/my_experiment_001/
```

gathered results:

```txt
gathered_results/my_experiment_001/
```

merged results:

```txt
merged_results/my_experiment_001/
```

merge report:

```txt
merged_results/my_experiment_001/merge_report.json
```

To monitor the launch:

```bash
tail -f distributed_launch_logs/my_experiment_001/*.out
tail -f distributed_launch_logs/my_experiment_001/*.err
tail -f distributed_launch_logs/my_experiment_001/*.log
```

To monitor setup:

```bash
tail -f setup_logs/my_experiment_001_setup/*.log
```

---

## 12. Common issues

### 12.1 Setup was skipped but remote project is missing

If you run:

```bash
--install false
```

the wrapper assumes the target nodes are already prepared.

If the launch fails because the remote project or virtual environment is missing, rerun with:

```bash
--install true
```

or manually run:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh ...
```

---

### 12.2 Wrong run id

The merge stage expects launch results under:

```txt
gathered_results/<run-id>/
```

If you manually modify launch output roots or run ids, make sure the wrapper uses the same values.

---

### 12.3 Merge fails because no `node_*` folders exist

The merge script expects gathered node folders such as:

```txt
node_local/
node_root_paradoxe-16.rennes.g5k/
```

If your node prefix is different, use:

```bash
--node-prefix <prefix>
```

---

### 12.4 Custom config files not found

The wrapper validates custom config files before running.

Check these paths:

```bash
--custom-flower-executor-cfg
--custom-flower-server-cfg
--custom-flower-client-cfg
```

They are paths on the launcher machine, not paths on remote nodes.

---

### 12.5 Remote config files not found

Remote config files are paths on the target nodes.

Check these options:

```bash
--remote-flower-executor-cfg
--remote-flower-server-cfg
--remote-flower-client-cfg
```

For Grid'5000 with default installation, the executor config is usually:

```txt
/root/metacs-fl/metacs_fl/flower_executor/config/flower_executor.cfg
```

---

### 12.6 GitHub token prompt appears once

When using:

```bash
--install true
--repo-auth token
--prompt-github-token true
```

the setup script prompts once locally for the token and then uses it during remote setup.

Do not put the token directly in `--repo-url`.

---

### 12.7 Setup logs use `<run-id>_setup`

The wrapper intentionally passes:

```bash
--run-id <run-id>_setup
```

to the setup script to avoid mixing setup logs with launch logs.

Example:

```txt
setup_logs/cifar10_iid_test_001_setup/
distributed_launch_logs/cifar10_iid_test_001/
```

---

### 12.8 Pipeline stops after launch failure

If the launch script exits with an error, the wrapper stops before the merge stage.

The launch script itself may still have gathered partial results. You can manually merge them with:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/<run-id> \
  merged_results/<run-id> \
  --clean-output true
```

---

### 12.9 Need an option not exposed by the wrapper

Use passthrough arguments.

Example for launch:

```bash
--launch-arg --remote-hostfile-dir \
--launch-arg /tmp/metacs_runtime
```

Example for merge:

```bash
--merge-arg --some-future-option \
--merge-arg value
```

---

## 13. Summary

For a fresh Grid'5000 allocation:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install true \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --branch main \
  --force-reclone true \
  --max-parallel-installs 8 \
  --run-id my_experiment_001 \
  --custom-flower-executor-cfg experiments/my_experiment/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_experiment/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_experiment/flower_client.cfg \
  --max-parallel-remote-ops 8 \
  --clean-merge-output true
```

For already prepared nodes:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id my_experiment_002 \
  --custom-flower-executor-cfg experiments/my_experiment/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_experiment/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_experiment/flower_client.cfg \
  --max-parallel-remote-ops 8 \
  --clean-merge-output true
```

The final merged results will be available under:

```txt
merged_results/<run-id>/
```
