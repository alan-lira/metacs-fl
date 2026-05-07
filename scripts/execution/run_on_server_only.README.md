# Server-Only Experiment Runner

This document explains how to use:

```txt
scripts/execution/run_on_server_only.sh
```

to run one server-only MetaCS-FL experiment on the server node defined in a shared node file.

All examples assume commands are executed from the project root.

---

## 1. Purpose

Some experiments do not need distributed Flower clients. For example:

```txt
static_client_availability/scalability_experiments
static_client_availability/sensitivity_experiments
```

These experiments can be run only on the server node. This script:

1. reads the shared node file;
2. extracts the single `server` entry;
3. runs a Python script on that server node;
4. optionally passes a config file using `--config-file`;
5. optionally collects an output directory back to the launcher machine.

This script is also used by:

```txt
scripts/execution/run_many_distributed_experiments.sh
```

for manifest rows with:

```txt
backend=server_only_python
```

---

## 2. Node file behavior

The script uses the same node files as setup and distributed execution:

```txt
scripts/nodes.local.txt
scripts/nodes.remote.txt
scripts/nodes.g5k.txt
```

Format:

```txt
mode <local|remote|g5k>

server <ssh_target|-> <runtime_host>
client <ssh_target|-> <runtime_host>
```

Only the `server` entry is used for execution.

In local mode:

```txt
mode local

server - 127.0.0.1
client - 127.0.0.1
```

In Grid'5000 mode:

```txt
mode g5k

server root@paradoxe-16.rennes.g5k paradoxe-16.rennes.grid5000.fr
client root@paradoxe-17.rennes.g5k paradoxe-17.rennes.grid5000.fr
```

The script SSHes to the `ssh_target` of the server entry.

---

## 3. Command-line usage

Basic form:

```bash
bash scripts/execution/run_on_server_only.sh \
  --nodes-file <nodes-file> \
  --script <python-script>
```

With a config file:

```bash
bash scripts/execution/run_on_server_only.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --script experiments/static_client_availability/scalability_experiments/scalability_experiments.py \
  --config-file experiments/static_client_availability/scalability_experiments/scalability_experiments_metacsfl_015.cfg \
  --run-id scalability_metacsfl_015
```

---

## 4. Parameters

| Parameter | Description | Default |
|---|---|---|
| `--nodes-file FILE` | Shared node-description file. | Required |
| `--script FILE` | Python script to run on the server node. | Required |
| `--config-file FILE` | Optional config file passed as `--config-file FILE`. | Not set |
| `--remote-project-dir DIR` | Project directory on the server node. | `/root/metacs-fl` |
| `--remote-venv-activate FILE` | Virtual environment activation script. | `<remote-project-dir>/.venv/bin/activate` |
| `--python-bin BIN` | Python executable. | `python3` |
| `--run-id ID` | Run identifier. | Timestamp |
| `--local-log-root DIR` | Local log root. | `server_only_logs` |
| `--remote-output-dir DIR` | Optional output directory to collect after execution. | Not set |
| `--local-output-root DIR` | Local output root for collected output. | `server_only_results` |
| `--clean-local-output true\|false` | Remove local collected output before copying. | `true` |
| `--ssh-options "OPTIONS"` | Extra SSH/rsync options. | Empty |
| `--rsync-options "OPTIONS"` | Extra rsync options. | `-az` |
| `--extra-arg ARG` | Extra argument passed to the Python script. Can be repeated. | Not set |

---

## 5. Output files

For a run id:

```txt
scalability_metacsfl_015
```

the script writes logs to:

```txt
server_only_logs/scalability_metacsfl_015/
├── server_only.out
├── server_only.err
└── server_only_manifest.txt
```

If `--remote-output-dir` is provided, the script collects that directory into:

```txt
server_only_results/scalability_metacsfl_015/
```

---

## 6. Examples

### 6.1 Local server-only run

```bash
bash scripts/execution/run_on_server_only.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --script experiments/static_client_availability/scalability_experiments/scalability_experiments.py \
  --config-file experiments/static_client_availability/scalability_experiments/scalability_experiments_metacsfl_015.cfg \
  --run-id local_scalability_metacsfl_015
```

---

### 6.2 Grid'5000 server-only run

```bash
bash scripts/execution/run_on_server_only.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --script experiments/static_client_availability/scalability_experiments/scalability_experiments.py \
  --config-file experiments/static_client_availability/scalability_experiments/scalability_experiments_metacsfl_015.cfg \
  --remote-output-dir results/static_client_availability/scalability_results/metacsfl_015 \
  --run-id scalability_metacsfl_015
```

---

### 6.3 Passing extra script arguments

Use one `--extra-arg` per token:

```bash
bash scripts/execution/run_on_server_only.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --script experiments/my_script.py \
  --extra-arg --foo \
  --extra-arg bar \
  --extra-arg --verbose
```

This runs:

```bash
python3 experiments/my_script.py --foo bar --verbose
```

---

## 7. Use inside campaigns

`run_many_distributed_experiments.sh` calls this script for manifest rows like:

```csv
backend=server_only_python
```

The manifest fields used are:

```txt
server_script
server_config
remote_output_dir
```

---

## 8. Common issues

### Server node not found

The node file must contain exactly one `server` entry.

### Config path not found remotely

The script passes the config path to the remote Python command as written. If running remotely, the path must exist on the remote machine relative to `--remote-project-dir`, unless it is absolute.

### Output directory not collected

If `--remote-output-dir` is set but the directory does not exist after the run, the script prints a warning and still exits successfully if the Python command succeeded.

Check:

```txt
server_only_logs/<run-id>/server_only.out
server_only_logs/<run-id>/server_only.err
```
