# Distributed Flower Launcher

This document explains how to use:

```txt
scripts/execution/launch_distributed_flower.sh
```

to launch a distributed Flower-based execution of MetaCS-FL in different environments:

- local execution;
- generic remote execution;
- Grid'5000 execution.

The launcher supports all these cases using the same script and a node-description file.

The launcher is located at:

```txt
scripts/execution/launch_distributed_flower.sh
```

The node-description files are shared by setup and execution scripts and are located at:

```txt
scripts/nodes.local.txt
scripts/nodes.remote.txt
scripts/nodes.g5k.txt
```

All examples in this document assume commands are executed from the project root folder.

---

## 1. Main idea

The launcher separates two different concepts:

1. **SSH target**
   - The address used by the launcher to reach a machine and start a process.

2. **Runtime host**
   - The address written to the Flower/gRPC runtime hostfile.
   - This is the address used by the distributed processes to communicate with each other.

This distinction is important because the address used to SSH into a machine is not always the same address that should be used by Flower/gRPC.

For example, in Grid'5000, from your local machine you may SSH using:

```txt
root@paradoxe-1.rennes.g5k
```

but inside the Grid'5000 network, Flower/gRPC may need to use:

```txt
paradoxe-1.rennes.grid5000.fr
```

Therefore, the launcher uses a node file with both addresses.

---

## 2. Node file format

Each node file uses the following format:

```txt
mode <local|remote|g5k>

<role> <ssh_target> <runtime_host>
```

Where:

| Field | Description |
|---|---|
| `mode` | Execution mode. Must be `local`, `remote`, or `g5k`. |
| `role` | Process role. Must be `server` or `client`. |
| `ssh_target` | Address used by the launcher to reach the machine. Use `-` in local mode. |
| `runtime_host` | Address written to the Flower/gRPC runtime hostfile. |

The node file itself is **not** passed directly to Flower.

Instead, the launcher generates a runtime hostfile containing only:

```txt
server <runtime_host>
client <runtime_host>
client <runtime_host>
```

Example generated runtime hostfile:

```txt
server paradoxe-16.rennes.grid5000.fr
client paradoxe-17.rennes.grid5000.fr
client paradoxe-18.rennes.grid5000.fr
```

The generated runtime hostfile is passed to:

```bash
main.py execute_fl_with_flower --hostfile <generated-runtime-hostfile>
```

---

## 3. Supported modes

### 3.1 Local mode

Use `local` mode when all processes run on the same local machine.

In local mode:

- no SSH is used;
- the launcher starts a single local controller process;
- the `FlowerExecutor` decides locally whether to start the server and clients;
- `ssh_target` must be `-`;
- the runtime host is usually `127.0.0.1`.

Example:

```txt
mode local

server - 127.0.0.1
client - 127.0.0.1
```

This generates the following Flower runtime hostfile:

```txt
server 127.0.0.1
client 127.0.0.1
```

Important: even though the node file has both `server` and `client` entries, the launcher starts only **one** local controller process in `local` mode. This avoids launching two independent local executions that could write to the same output/model files at the same time.

If your executor has trouble distinguishing the local server and local client because both use `127.0.0.1`, you can use different loopback aliases:

```txt
mode local

server - 127.0.0.1
client - 127.0.0.2
```

Linux treats the whole `127.0.0.0/8` range as loopback, so `127.0.0.2` is still local.

---

### 3.2 Generic remote mode

Use `remote` mode when launching processes on remote machines outside Grid'5000.

In generic remote mode:

- the launcher connects to each machine using SSH;
- verifies remote paths;
- copies the generated runtime hostfile to each unique remote machine;
- optionally copies custom runtime config files to each unique remote machine;
- launches one process per node entry;
- collects output folders after execution.

In many generic remote setups, the SSH target and runtime host are almost the same.

Example with hostnames:

```txt
mode remote

server ubuntu@server.example.com server.example.com
client ubuntu@client1.example.com client1.example.com
client ubuntu@client2.example.com client2.example.com
```

Example with IP addresses:

```txt
mode remote

server ubuntu@192.168.1.10 192.168.1.10
client ubuntu@192.168.1.11 192.168.1.11
client ubuntu@192.168.1.12 192.168.1.12
```

The difference is that `ssh_target` usually includes the SSH user, while `runtime_host` usually does not.

---

### 3.3 Grid'5000 mode

Use `g5k` mode when launching processes on Grid'5000.

In Grid'5000 mode:

- the launcher uses the first address to SSH into the node;
- Flower/gRPC uses the second address for runtime communication;
- custom config files, when provided, are copied to every unique remote node;
- remote validation, runtime file copy, and result collection can run in parallel.

Example:

```txt
mode g5k

server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2.rennes.g5k paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3.rennes.g5k paradoxe-3.rennes.grid5000.fr
```

If your local SSH configuration defines short aliases, you can use them as SSH targets:

```txt
mode g5k

server root@paradoxe-1 paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2 paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3 paradoxe-3.rennes.grid5000.fr
```

If your executor identifies the current node using short hostnames, such as `hostname` returning `paradoxe-1`, use short runtime hosts:

```txt
mode g5k

server root@paradoxe-1.rennes.g5k paradoxe-1
client root@paradoxe-2.rennes.g5k paradoxe-2
client root@paradoxe-3.rennes.g5k paradoxe-3
```

This can avoid mismatches between the runtime hostfile and the local hostname detected by the executor.

---

## 4. Example node files

### 4.1 `scripts/nodes.local.txt`

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

Using full Grid'5000 runtime hostnames:

```txt
mode g5k

server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2.rennes.g5k paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3.rennes.g5k paradoxe-3.rennes.grid5000.fr
```

Using local SSH aliases:

```txt
mode g5k

server root@paradoxe-1 paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2 paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3 paradoxe-3.rennes.grid5000.fr
```

Using short runtime hostnames:

```txt
mode g5k

server root@paradoxe-1.rennes.g5k paradoxe-1
client root@paradoxe-2.rennes.g5k paradoxe-2
client root@paradoxe-3.rennes.g5k paradoxe-3
```

---

## 5. Script usage

From the project root folder, the launcher is called as:

```bash
bash scripts/execution/launch_distributed_flower.sh --nodes-file <nodes-file> [options]
```

The only required parameter is:

```bash
--nodes-file
```

All other important variables can be passed as command-line parameters.

Example:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt
```

---

## 6. Parameters

| Parameter | Description | Default |
|---|---|---|
| `--nodes-file FILE` | Node description file. | Required |
| `--remote-project-dir DIR` | Project directory as seen by the machine running the FL process. | `/root/metacs-fl` |
| `--remote-venv-activate FILE` | Virtual environment activation script as seen by the machine running the FL process. | `<remote-project-dir>/.venv/bin/activate` |
| `--remote-flower-executor-cfg FILE` | Flower executor config file already present on the execution machine. Used when no custom executor config is provided. | `<remote-project-dir>/metacs_fl/flower_executor/config/flower_executor.cfg` |
| `--remote-config-file FILE` | Backward-compatible alias for `--remote-flower-executor-cfg`. | Same as above |
| `--remote-flower-server-cfg FILE` | Flower server config file already present on the execution machine. Used only for patching the runtime executor config. | Not set |
| `--remote-flower-client-cfg FILE` | Flower client config file already present on the execution machine. Used only for patching the runtime executor config. | Not set |
| `--custom-flower-executor-cfg FILE` | Custom Flower executor config file read from the launcher machine for this run. | Not set |
| `--custom-flower-server-cfg FILE` | Custom Flower server config file read from the launcher machine for this run. | Not set |
| `--custom-flower-client-cfg FILE` | Custom Flower client config file read from the launcher machine for this run. | Not set |
| `--remote-runtime-config-dir DIR` | Directory where per-run config files are copied on remote nodes. | `<remote-project-dir>/.distributed_runtime_configs/<run-id>` |
| `--patch-executor-config true\|false` | Whether to patch the runtime `flower_executor.cfg` so it points to the chosen `flower_server.cfg` and `flower_client.cfg` paths. | `true` |
| `--patch-grpc-config true\|false` | Whether to patch runtime `flower_server.cfg` and `flower_client.cfg` gRPC settings. | `true` |
| `--server-port PORT` | Fallback server port used when no runtime server config is available or no `listen_port` is found. | `8080` |
| `--remote-hostfile-dir DIR` | Directory where the runtime hostfile is copied on remote nodes. | `<remote-project-dir>` |
| `--action ACTION` | Action passed to `main.py`. | `execute_fl_with_flower` |
| `--repetitions N` | Number of repetitions. | `1` |
| `--execution-blocks BLOCK` | Optional execution block selector passed to `main.py`, for example `Execution_1_N` or `Execution_2_N`. When omitted, all execution blocks defined in the executor config may run. | Not set |
| `--python-bin BIN` | Python executable to use. | `python3` |
| `--output-dir DIR` | Base directory for local launcher artifacts. If provided, the default log and gather roots become `DIR/distributed_launch_logs` and `DIR/gathered_results`. Explicit `--local-log-root` or `--local-gather-root` values override this default. | Not set |
| `--local-log-root DIR` | Root directory for local logs. | `distributed_launch_logs` |
| `--local-gather-root DIR` | Root directory for gathered results. | `gathered_results` |
| `--gather-outputs true\|false` | Value passed to `main.py --gather-outputs`. | `false` |
| `--max-parallel-remote-ops N` | Maximum number of remote validation, copy, and result-collection operations in parallel. | `8` |
| `--ssh-options "OPTIONS"` | Extra options passed to `ssh` and `scp`. | Empty |
| `--rsync-options "OPTIONS"` | Extra options passed to `rsync`. | `-az` |
| `--run-id ID` | Explicit run identifier. | Timestamp |

`--execution-blocks` is useful when one executor configuration defines multiple named execution blocks and only one block should run in the current launch. The value is forwarded unchanged to:

```txt
main.py --execution-blocks <block>
```

Do not pass the option when every block in the executor configuration should run.

---

## 7. Config file behavior

The launcher supports two ways of selecting configuration files.

### 7.1 Remote config files

Remote config files are files that already exist on the execution machine.

For example:

```bash
--remote-flower-executor-cfg /root/metacs-fl/metacs_fl/flower_executor/config/flower_executor.cfg
```

This is the default behavior.

The backward-compatible option below is still supported:

```bash
--remote-config-file /root/metacs-fl/metacs_fl/flower_executor/config/flower_executor.cfg
```

It is equivalent to:

```bash
--remote-flower-executor-cfg /root/metacs-fl/metacs_fl/flower_executor/config/flower_executor.cfg
```

---

### 7.2 Custom config files

Custom config files are files provided from the launcher machine for a specific run.

Use:

```bash
--custom-flower-executor-cfg <path>
--custom-flower-server-cfg <path>
--custom-flower-client-cfg <path>
```

In local mode, these files are copied to a local runtime config directory under:

```txt
<local-log-root>/<run_id>/runtime_configs/
```

In remote and Grid'5000 modes, these files are copied to each unique remote node under:

```txt
<remote-project-dir>/.distributed_runtime_configs/<run_id>/
```

For example:

```txt
/root/metacs-fl/.distributed_runtime_configs/20260506_221530/
```

The runtime config files are then passed to the execution through the runtime `flower_executor.cfg`.

---

### 7.3 Runtime executor config patching

When custom server or client config files are provided, the launcher can patch the runtime copy of `flower_executor.cfg` so that it points to the selected `flower_server.cfg` and `flower_client.cfg`.

This behavior is enabled by default:

```bash
--patch-executor-config true
```

To disable it:

```bash
--patch-executor-config false
```

The patching is applied only to the runtime copy of `flower_executor.cfg`. Your original config file is not modified.

The patcher updates common option names such as:

```txt
flower_server_cfg
flower_server_config
flower_server_config_file
server_cfg
server_config
server_config_file
flower_server_settings_file
base_server_config_file
```

and:

```txt
flower_client_cfg
flower_client_config
flower_client_config_file
client_cfg
client_config
client_config_file
flower_client_settings_file
base_client_config_file
```

It also updates entries whose value ends with:

```txt
flower_server.cfg
flower_client.cfg
```

The option names:

```txt
base_server_config_file
base_client_config_file
```

are the MetaCS-FL executor config names used by the current project configs. They are supported by the launcher patcher.

If the patch result is:

```txt
patched=false
```

then the launcher did not find a matching option in `flower_executor.cfg`. In that case, check the option names used by your executor config file, or manually set the correct paths inside your custom `flower_executor.cfg`.

A successful patch usually prints:

```txt
Runtime executor config patch result: patched=true
```

and the runtime `flower_executor.cfg` should point to files under:

```txt
<remote-project-dir>/.distributed_runtime_configs/<run-id>/
```

Example:

```ini
base_server_config_file = /root/metacs-fl/.distributed_runtime_configs/<run-id>/flower_server.cfg
base_client_config_file = /root/metacs-fl/.distributed_runtime_configs/<run-id>/flower_client.cfg
```

---

### 7.4 Runtime gRPC config patching

The launcher can also patch the runtime copies of `flower_server.cfg` and `flower_client.cfg`.

This behavior is enabled by default:

```bash
--patch-grpc-config true
```

To disable it:

```bash
--patch-grpc-config false
```

When enabled, the launcher derives the server runtime host from the `server` entry in the node file and patches the runtime client config:

```ini
[gRPC Settings]
server_ip_address = <server-runtime-host>
server_port = <server-runtime-port>
```

For example, if the generated runtime hostfile contains:

```txt
server paradoxe-16.rennes.grid5000.fr
client paradoxe-17.rennes.grid5000.fr
client paradoxe-18.rennes.grid5000.fr
```

then the runtime `flower_client.cfg` is patched to use:

```ini
[gRPC Settings]
server_ip_address = paradoxe-16.rennes.grid5000.fr
server_port = 8080
```

For remote and Grid'5000 mode, the runtime server config is patched to listen on all interfaces:

```ini
[gRPC Settings]
listen_ip_address = 0.0.0.0
listen_port = 8080
```

The server port is resolved as follows:

1. if a runtime `flower_server.cfg` exists and contains `[gRPC Settings] listen_port`, that value is used;
2. otherwise, the fallback value from `--server-port` is used;
3. the default fallback is `8080`.

The patching is applied only to runtime copies of the config files. Your original config files are not modified.

---

## 8. Parallel remote operations

For remote and Grid'5000 modes, the launcher can run some remote operations in parallel.

The parallelism limit is controlled by:

```bash
--max-parallel-remote-ops N
```

Default:

```bash
--max-parallel-remote-ops 8
```

The following stages are parallelized:

1. remote path validation;
2. runtime hostfile and config copy;
3. result collection.

The actual FL execution is also launched concurrently, one background SSH process per node entry.

Use a lower value if SSH, `scp`, `rsync`, or the remote filesystem becomes overloaded:

```bash
--max-parallel-remote-ops 2
```

Use a higher value for larger allocations:

```bash
--max-parallel-remote-ops 16
```

Each parallel stage writes per-target logs under:

```txt
<local-log-root>/<run_id>/
```

For example:

```txt
<local-log-root>/<run_id>/verify_remote_root_paradoxe-1.rennes.g5k.log
<local-log-root>/<run_id>/copy_runtime_files_root_paradoxe-1.rennes.g5k.log
<local-log-root>/<run_id>/collect_results_root_paradoxe-1.rennes.g5k.log
```

---

## 9. Example calls

### 9.1 Local execution with default configs

Use this when running both server and client locally.

From the project root folder:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --remote-flower-executor-cfg "$PWD/metacs_fl/flower_executor/config/flower_executor.cfg" \
  --repetitions 1
```

The backward-compatible version also works:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --remote-config-file "$PWD/metacs_fl/flower_executor/config/flower_executor.cfg" \
  --repetitions 1
```

If you are already inside the project root and your virtual environment follows the default layout:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD"
```

---

### 9.2 Local execution with custom configs

Use this when you want to run locally with custom config files:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --repetitions 1
```

In local mode, the launcher starts only one local controller process. The `FlowerExecutor` then starts the local server and clients.

---

### 9.3 Generic remote execution with default configs

Example using `/home/ubuntu/metacs-fl` on all remote machines:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.remote.txt \
  --remote-project-dir /home/ubuntu/metacs-fl \
  --remote-venv-activate /home/ubuntu/metacs-fl/.venv/bin/activate \
  --remote-flower-executor-cfg /home/ubuntu/metacs-fl/metacs_fl/flower_executor/config/flower_executor.cfg \
  --repetitions 1
```

If the remote project is located at `/root/metacs-fl`, the defaults already match:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.remote.txt
```

---

### 9.4 Generic remote execution with custom configs

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.remote.txt \
  --remote-project-dir /home/ubuntu/metacs-fl \
  --remote-venv-activate /home/ubuntu/metacs-fl/.venv/bin/activate \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --repetitions 1
```

The custom config files are copied to each unique remote node under:

```txt
/home/ubuntu/metacs-fl/.distributed_runtime_configs/<run_id>/
```

The runtime `flower_client.cfg` is patched so clients connect to the server runtime host from the node file.

---

### 9.5 Grid'5000 execution with default configs

Example with explicit paths:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --remote-flower-executor-cfg /root/metacs-fl/metacs_fl/flower_executor/config/flower_executor.cfg \
  --repetitions 1
```

Since the defaults already assume `/root/metacs-fl`, this is usually enough:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt
```

---

### 9.6 Grid'5000 execution with custom configs

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --repetitions 1
```

The custom config files are copied to each unique Grid'5000 node under:

```txt
/root/metacs-fl/.distributed_runtime_configs/<run_id>/
```

The runtime `flower_client.cfg` is patched so clients connect to the server runtime host from the node file. The runtime `flower_executor.cfg` is also patched to point to the copied runtime server/client configs when supported option names are found.

---

### 9.7 Grid'5000 execution with custom configs and parallel remote ops

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

---

### 9.8 Grid'5000 with multiple repetitions

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repetitions 3
```

---

### 9.9 Run one configured execution block

Use `--execution-blocks` when the selected executor config contains multiple blocks but this launch should execute only one of them:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --execution-blocks Execution_1_N \
  --repetitions 1
```

The block name must match a block understood by `main.py` and the selected executor configuration. Omit the option to run all configured blocks.

---

### 9.10 Custom run id

Use this when you want predictable output folder names.

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --run-id cifar10_iid_test_001
```

This creates logs and gathered results under:

```txt
distributed_launch_logs/cifar10_iid_test_001
gathered_results/cifar10_iid_test_001
```

It also copies custom runtime configs, if provided, to:

```txt
/root/metacs-fl/.distributed_runtime_configs/cifar10_iid_test_001/
```

for remote/Grid'5000 executions.

---

### 9.11 Custom output directory

Use this when you want all local launcher artifacts for a campaign under a specific base directory.

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --output-dir /mnt/d/results \
  --repetitions 1
```

This creates logs and gathered results under:

```txt
/mnt/d/results/distributed_launch_logs/<run_id>
/mnt/d/results/gathered_results/<run_id>
```

Explicit `--local-log-root` or `--local-gather-root` values override the corresponding root derived from `--output-dir`.

---

### 9.12 Custom Python binary

Use this if the remote virtual environment expects a specific Python executable.

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.remote.txt \
  --remote-project-dir /home/ubuntu/metacs-fl \
  --python-bin python3.11
```

---

### 9.13 Custom SSH options

Use this when you need to pass options to SSH and SCP:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.remote.txt \
  --remote-project-dir /home/ubuntu/metacs-fl \
  --ssh-options "-o StrictHostKeyChecking=no"
```

---

### 9.14 Custom rsync options

Use this when result collection needs additional rsync options:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.remote.txt \
  --remote-project-dir /home/ubuntu/metacs-fl \
  --rsync-options "-az --partial --info=progress2"
```

---

## 10. What the launcher does

The launcher performs the following steps:

1. Parses command-line arguments.
2. Reads the node file.
3. Validates the execution mode.
4. Checks that exactly one `server` entry exists.
5. Checks that at least one `client` entry exists.
6. Builds the list of unique SSH targets for remote/Grid'5000 modes.
7. Stages custom config files if provided.
8. Generates the Flower runtime hostfile.
9. Derives the server runtime host and port.
10. Patches runtime `flower_server.cfg` and `flower_client.cfg` gRPC settings if enabled.
11. Patches the runtime `flower_executor.cfg` if enabled.
12. Verifies local or remote paths.
13. Copies the runtime hostfile to remote nodes when needed.
14. Copies custom runtime config files to remote nodes when needed.
15. Starts the execution:
    - one local controller process in `local` mode;
    - one process per node entry in `remote` and `g5k` modes.
16. Waits for all processes to finish.
17. Parses the runtime executor config file to discover output folders.
18. Pulls or copies result folders into a local gather directory.
19. Saves a manifest describing the run.

---

## 11. Generated runtime hostfile

The node file is not passed directly to Flower.

For example, this node file:

```txt
mode g5k

server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2.rennes.g5k paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3.rennes.g5k paradoxe-3.rennes.grid5000.fr
```

generates this runtime hostfile:

```txt
server paradoxe-1.rennes.grid5000.fr
client paradoxe-2.rennes.grid5000.fr
client paradoxe-3.rennes.grid5000.fr
```

This generated hostfile is stored locally under:

```txt
<local-log-root>/<run_id>/nodes_runtime_<run_id>.txt
```

and copied remotely to:

```txt
<remote-hostfile-dir>/nodes_runtime_<run_id>.txt
```

By default, this remote path is:

```txt
<remote-project-dir>/nodes_runtime_<run_id>.txt
```

---

## 12. Output directories

For each execution, the launcher creates a run id.

By default, the run id is the current timestamp:

```txt
YYYYMMDD_HHMMSS
```

The launcher creates one log directory and one gathered-results directory:

```txt
<local-log-root>/<run_id>
<local-gather-root>/<run_id>
```

By default, the local roots are:

```txt
distributed_launch_logs
gathered_results
```

Therefore, with default roots, a run creates:

```txt
distributed_launch_logs/<run_id>
gathered_results/<run_id>
```

If `--output-dir DIR` is provided and no explicit `--local-log-root` or `--local-gather-root` is provided, the launcher uses:

```txt
DIR/distributed_launch_logs/<run_id>
DIR/gathered_results/<run_id>
```

For example:

```txt
/mnt/d/results/distributed_launch_logs/<run_id>
/mnt/d/results/gathered_results/<run_id>
```

The generated runtime hostfile is stored under:

```txt
<local-log-root>/<run_id>/nodes_runtime_<run_id>.txt
```

The local staged runtime config files are stored under:

```txt
<local-log-root>/<run_id>/runtime_configs/
```

For remote/Grid'5000 executions, custom runtime config files are copied to:

```txt
<remote-project-dir>/.distributed_runtime_configs/<run_id>/
```

unless `--remote-runtime-config-dir` is explicitly provided.

---

## 13. Logs

Each launched process writes two files:

```txt
<process>.out
<process>.err
```

In local mode, the launcher starts a single process:

```txt
<local-log-root>/<run_id>/local_controller.out
<local-log-root>/<run_id>/local_controller.err
```

In remote/Grid'5000 mode, each node entry gets its own log files, for example:

```txt
<local-log-root>/<run_id>/0_server_paradoxe-1.rennes.grid5000.fr.out
<local-log-root>/<run_id>/0_server_paradoxe-1.rennes.grid5000.fr.err
<local-log-root>/<run_id>/1_client_paradoxe-2.rennes.grid5000.fr.out
<local-log-root>/<run_id>/1_client_paradoxe-2.rennes.grid5000.fr.err
```

Parallel remote stages also write logs:

```txt
<local-log-root>/<run_id>/verify_remote_<target>.log
<local-log-root>/<run_id>/copy_runtime_files_<target>.log
<local-log-root>/<run_id>/collect_results_<target>.log
```

Monitor stdout with:

```bash
tail -f <local-log-root>/<run_id>/*.out
```

Monitor stderr with:

```bash
tail -f <local-log-root>/<run_id>/*.err
```

Monitor parallel stage logs with:

```bash
tail -f <local-log-root>/<run_id>/*.log
```

---

## 14. Gathered results

After execution, the launcher reads the runtime `flower_executor.cfg` and searches for sections like:

```ini
[Execution_..._N Settings]
execution_output_folder = ...
```

For each declared `execution_output_folder`, it resolves the repetition index and copies result folders into:

```txt
<local-gather-root>/<run_id>/node_<node_name>/
```

Example for Grid'5000:

```txt
gathered_results/20260506_221530/node_root_paradoxe-1.rennes.g5k/
gathered_results/20260506_221530/node_root_paradoxe-2.rennes.g5k/
gathered_results/20260506_221530/node_root_paradoxe-3.rennes.g5k/
```

In local mode, results are copied into:

```txt
<local-gather-root>/<run_id>/node_local/
```

The launcher also copies the runtime config files into the gather directory when available:

```txt
<local-gather-root>/<run_id>/flower_executor.runtime.cfg
<local-gather-root>/<run_id>/flower_server.runtime.cfg
<local-gather-root>/<run_id>/flower_client.runtime.cfg
```

---

## 15. Run manifest

Each run creates a manifest file:

```txt
<local-gather-root>/<run_id>/distributed_run_manifest.txt
```

The manifest records:

- run id;
- launch mode;
- node file used;
- server runtime host;
- server runtime port;
- remote project directory;
- remote virtual environment path;
- remote and custom config file paths;
- runtime config file paths;
- remote runtime config directory;
- remote hostfile directory;
- action;
- repetitions;
- Python binary;
- `gather_outputs`;
- whether executor config patching was enabled;
- whether gRPC config patching was enabled;
- maximum parallel remote operations;
- SSH options;
- rsync options;
- generated runtime hostfile;
- full node mapping.

This is useful for reproducibility.

---

## 16. Common issues

### 16.1 Config file not found

Example error:

```txt
ERROR: runtime flower_executor.cfg not found: ...
```

Check whether you meant to use a remote config file:

```bash
--remote-flower-executor-cfg /path/on/execution/machine/flower_executor.cfg
```

or a custom config file from the launcher machine:

```bash
--custom-flower-executor-cfg /path/on/launcher/machine/flower_executor.cfg
```

For local mode, this is commonly:

```bash
--remote-flower-executor-cfg "$PWD/metacs_fl/flower_executor/config/flower_executor.cfg"
```

For Grid'5000, this is commonly:

```bash
--remote-flower-executor-cfg /root/metacs-fl/metacs_fl/flower_executor/config/flower_executor.cfg
```

---

### 16.2 Virtual environment not found

Example error:

```txt
ERROR: venv activate file not found: /root/metacs-fl/.venv/bin/activate
```

Pass the correct virtual environment path:

```bash
--remote-venv-activate /path/to/venv/bin/activate
```

---

### 16.3 SSH works manually but the script fails

Check the `ssh_target` field in the node file.

For example, this line:

```txt
server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
```

requires this command to work:

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

### 16.4 Flower cannot connect between nodes

This usually means the runtime hostnames or gRPC config values are wrong.

Check the third column of the node file.

For Grid'5000, this may be:

```txt
paradoxe-1.rennes.grid5000.fr
```

or, if your executor expects short hostnames:

```txt
paradoxe-1
```

Also check the gathered runtime client config:

```txt
<local-gather-root>/<run_id>/flower_client.runtime.cfg
```

The client config should point to the server runtime host:

```ini
[gRPC Settings]
server_ip_address = <server-runtime-host>
server_port = <server-runtime-port>
```

If it does not, ensure gRPC patching is enabled:

```bash
--patch-grpc-config true
```

---

### 16.5 Server listens only on localhost

If the server config contains:

```ini
[gRPC Settings]
listen_ip_address = 127.0.0.1
```

remote clients cannot connect.

In remote/Grid'5000 mode, gRPC patching changes the runtime server config to:

```ini
[gRPC Settings]
listen_ip_address = 0.0.0.0
```

Ensure this behavior is enabled:

```bash
--patch-grpc-config true
```

---

### 16.6 Local mode role ambiguity

If your local node file is:

```txt
mode local

server - 127.0.0.1
client - 127.0.0.1
```

and your executor identifies roles by matching local IP/hostname against the hostfile, both local roles may match the same address.

Try:

```txt
mode local

server - 127.0.0.1
client - 127.0.0.2
```

or adapt the executor to distinguish local processes by role/index instead of host address only.

Note that the launcher itself starts only one process in local mode.

---

### 16.7 Custom server/client config was copied but not used

If the launcher prints:

```txt
patched=false
```

then it did not find a recognized server/client config option inside `flower_executor.cfg`.

Current supported server option names include:

```txt
flower_server_cfg
flower_server_config
flower_server_config_file
server_cfg
server_config
server_config_file
flower_server_settings_file
base_server_config_file
```

Current supported client option names include:

```txt
flower_client_cfg
flower_client_config
flower_client_config_file
client_cfg
client_config
client_config_file
flower_client_settings_file
base_client_config_file
```

If your config uses different option names, either:

1. update the launcher patcher to recognize them; or
2. manually set the correct `flower_server.cfg` and `flower_client.cfg` paths inside your custom executor config.

---

### 16.8 Runtime gRPC config patch did not happen

The gRPC patcher only patches runtime server/client config files that are staged locally.

This means it normally applies when you pass:

```bash
--custom-flower-server-cfg ...
--custom-flower-client-cfg ...
```

If you use only remote config files and no custom server/client config files, the launcher does not download and patch the remote files. In that case, make sure the remote files are already correct, or pass custom config files.

---

### 16.9 Node file has hidden characters

The script accepts Windows CRLF line endings, but if something still looks strange, inspect the node file with:

```bash
cat -A scripts/nodes.g5k.txt
```

A healthy file should look like normal text lines ending with `$`.

If you see `^M`, normalize the file with:

```bash
sed -i 's/\r$//' scripts/nodes.g5k.txt
```

---

### 16.10 Parallel remote operation fails on some nodes

Remote verification, runtime file copy, and result collection run in parallel.

If you see failures during these stages, inspect the corresponding stage log:

```txt
<local-log-root>/<run_id>/verify_remote_<target>.log
<local-log-root>/<run_id>/copy_runtime_files_<target>.log
<local-log-root>/<run_id>/collect_results_<target>.log
```

You can reduce parallelism with:

```bash
--max-parallel-remote-ops 2
```

or disable practical parallelism with:

```bash
--max-parallel-remote-ops 1
```

---

## 17. Recommended workflow

A typical workflow is:

1. Install MetaCS-FL on all nodes:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --max-parallel-installs 8
```

2. Launch the distributed Flower execution:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

3. Monitor logs:

```bash
tail -f <local-log-root>/<run_id>/*.out
tail -f <local-log-root>/<run_id>/*.err
tail -f <local-log-root>/<run_id>/*.log
```

4. Inspect gathered results:

```bash
ls <local-gather-root>/<run_id>/
```
