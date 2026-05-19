# MetaCS-FL

**MetaCS-FL** is a metaheuristic-based client selection framework for Federated Learning (FL) systems. It targets heterogeneous Cross-Device FL scenarios, where clients may differ in computational capacity, energy behavior, communication performance, availability, data quantity, and data distribution.

The framework models client selection not only as the choice of which clients participate in a federated round, but also as a task-allocation problem: selected clients may receive different local data workloads according to system, energy, fairness, and learning objectives.

MetaCS-FL supports:

- profiling-aware client selection;
- time- and energy-aware scheduling;
- multi-objective optimization;
- metaheuristic refinement (implemented approaches: Large Neighborhood Search);
- configurable client selection triggers;
- distributed execution with Flower;
- Grid'5000 and generic remote execution;
- post-processing, merging, and analysis scripts for reproducible experiments.

---

## Table of contents

- [Reproducing published experiments](#reproducing-published-experiments)
  - [FGCS 2026 experiments](#fgcs-2026-experiments)
- [Recommended execution order](#recommended-execution-order)
- [Requirements](#requirements)
  - [Python environment](#python-environment)
  - [System packages](#system-packages)
  - [Energy measurement notes](#energy-measurement-notes)
- [1. Configure the node file](#1-configure-the-node-file)
- [2. Install/setup remote nodes](#2-installsetup-remote-nodes)
- [3. Run toy smoke experiments](#3-run-toy-smoke-experiments)
- [4. Smoke test decision](#4-smoke-test-decision)
  - [4.1 If the smoke test fails: inspect logs and fix environment/configs](#41-if-the-smoke-test-fails-inspect-logs-and-fix-environmentconfigs)
    - [Check for out-of-memory (OOM) kills](#check-for-out-of-memory-oom-kills)
  - [4.2 If the smoke test passes: continue to the experiment campaign](#42-if-the-smoke-test-passes-continue-to-the-experiment-campaign)
- [5. Generate or run the experiment campaign](#5-generate-or-run-the-experiment-campaign)
  - [5.1 Experiment packs](#51-experiment-packs)
  - [5.2 Generate experiment packs](#52-generate-experiment-packs)
  - [5.3 Dry-run the FGCS 2026 pack](#53-dry-run-the-fgcs-2026-pack)
  - [5.4 Run many experiments as a campaign](#54-run-many-experiments-as-a-campaign)
  - [5.5 Resume the campaign](#55-resume-the-campaign)
- [6. Merge distributed outputs](#6-merge-distributed-outputs)
- [7. Run analysis scripts](#7-run-analysis-scripts)
  - [7.1 DP impact analysis](#71-dp-impact-analysis)
  - [7.2 Performance plots](#72-performance-plots)
  - [7.3 Scalability analysis](#73-scalability-analysis)
  - [7.4 Performance summary](#74-performance-summary)
- [8. Export plots, tables, and metrics](#8-export-plots-tables-and-metrics)
- [Optional execution modes](#optional-execution-modes)
  - [Run one distributed experiment](#run-one-distributed-experiment)
  - [Run server-only experiments](#run-server-only-experiments)
  - [Launch distributed Flower directly](#launch-distributed-flower-directly)
- [Quick command summary](#quick-command-summary)
  - [Setup](#setup)
  - [Smoke test](#smoke-test)
  - [Full campaign dry-run](#full-campaign-dry-run)
  - [Full campaign](#full-campaign)
  - [Resume campaign](#resume-campaign)
- [Notes for Grid'5000](#notes-for-grid5000)
- [Results](#results)
  - [FGCS 2026](#fgcs-2026)
- [Scientific Productions](#scientific-productions)
  - [1. MetaCS-FL: A Metaheuristic-Based Framework for Client Selection in Federated Learning Systems](#1-metacs-fl-a-metaheuristic-based-framework-for-client-selection-in-federated-learning-systems)
- [License](#license)

---

## Reproducing published experiments

This repository may evolve over time as the project receives improvements, fixes, and new features. The `main` branch is stable, but it will continue to evolve after each publication. Therefore, for exact reproducibility of published experiments, use the frozen release tag associated with the corresponding publication instead of relying on the default branch.

### FGCS 2026 experiments

To reproduce the experiments reported in the FGCS 2026 paper, use the frozen release tag `v0.2.0`:

```bash
git clone https://github.com/alan-lira/metacs-fl.git
cd metacs-fl
git checkout v0.2.0
```

When using the remote/Grid'5000 setup script, pass:

```bash
--branch v0.2.0
```

---

## Recommended execution order

1. [Configure the node file](#1-configure-the-node-file)
2. [Install/setup remote nodes](#2-installsetup-remote-nodes)
3. [Run toy smoke experiments](#3-run-toy-smoke-experiments)
4. [Smoke test decision](#4-smoke-test-decision)
   - [4.1 If the smoke test fails: inspect logs and fix environment/configs](#41-if-the-smoke-test-fails-inspect-logs-and-fix-environmentconfigs)
   - [4.2 If the smoke test passes: continue to the experiment campaign](#42-if-the-smoke-test-passes-continue-to-the-experiment-campaign)
5. [Generate or run the experiment campaign](#5-generate-or-run-the-experiment-campaign)
   - [5.1 Experiment packs](#51-experiment-packs)
   - [5.2 Generate experiment packs](#52-generate-experiment-packs)
   - [5.3 Dry-run the FGCS 2026 pack](#53-dry-run-the-fgcs-2026-pack)
   - [5.4 Run many experiments as a campaign](#54-run-many-experiments-as-a-campaign)
   - [5.5 Resume the campaign](#55-resume-the-campaign)
6. [Merge distributed outputs](#6-merge-distributed-outputs)
7. [Run analysis scripts](#7-run-analysis-scripts)
   - [7.1 DP impact analysis](#71-dp-impact-analysis)
   - [7.2 Performance plots](#72-performance-plots)
   - [7.3 Scalability analysis](#73-scalability-analysis)
   - [7.4 Performance summary](#74-performance-summary)
8. [Export plots, tables, and metrics](#8-export-plots-tables-and-metrics)

## Requirements

The exact Python and system dependencies depend on the experiment mode. Local simulation, distributed Flower execution, Grid'5000 deployment, and energy profiling require different subsets of dependencies.

### Python environment

The project is currently tested with a Python virtual environment and the dependencies listed in:

```txt
requirements.txt
```

Install Python dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip3 install -r requirements.txt
pip3 install -e .
```

The current Python dependency set includes:

```txt
contractions==0.1.73
cryptography==44.0.3
emoji==2.15.0
flwr-datasets==0.5.0
flwr[simulation]==1.21.0
iterators==0.0.2
keras==3.11.3
matplotlib==3.10.6
nltk==3.9.2
numpy==2.1.3
pandas==2.3.2
Pillow==11.3.0
psutil==7.1.0
pycryptodome==3.23.0
pyJoules==0.5.1
pynvml==13.0.1
pyRAPL==0.2.3.1
python-dateutil==2.9.0.post0
ray[default]==2.31.0
requests==2.32.5
scikit-learn==1.7.2
seaborn==0.13.2
setuptools==80.9.0
tensorflow==2.19.1
```

Energy/profiling-related Python packages include:

```txt
pyJoules
pyRAPL
pynvml
psutil
```

### System packages

Typical system dependencies include:

- `git`;
- `python3-venv`;
- `python3-pip`;
- `python3-tk`;
- `wget`;
- `rsync`;
- `openssh-client`;
- `iperf3`;
- `stress-ng`;
- `cpufrequtils`;
- Linux performance tools, when using system-level profiling;
- PowerJoular, when using PowerJoular-based energy measurements.

For remote/Grid'5000 execution, the setup script installs the main system dependencies automatically:

```txt
scripts/setup/setup_remote_metacsfl_node.sh
```

See:

```txt
scripts/setup/setup_remote_metacsfl_node.README.md
```

### Energy measurement notes

MetaCS-FL can use multiple energy/profiling mechanisms depending on the experiment configuration, including PowerJoular, pyJoules, PyRAPL, and NVML-based tools.

RAPL-based tools such as `pyRAPL` require Intel RAPL support and permission to read powercap files. On Linux, check whether RAPL is available with:

```bash
ls /sys/class/powercap/
```

If RAPL entries are not visible, the relevant kernel module may need to be loaded:

```bash
sudo modprobe intel_rapl_common
```

Depending on the machine and security policy, the executing user may also need read permissions for the RAPL powercap interface, or the profiling component may need to run with elevated privileges.

For NVIDIA GPU energy/profiling through NVML, make sure NVIDIA drivers and NVML support are available on the target machine.

[Back to table of contents](#table-of-contents)

---

## 1. Configure the node file

Distributed execution uses node files stored under:

```txt
scripts/nodes.local.txt
scripts/nodes.remote.txt
scripts/nodes.g5k.txt
```

Each node file has the format:

```txt
mode <local|remote|g5k>

<role> <ssh_target> <runtime_host>
```

Example Grid'5000 node file:

```txt
mode g5k

server root@paradoxe-1.rennes.g5k paradoxe-1.rennes.grid5000.fr
client root@paradoxe-2.rennes.g5k paradoxe-2.rennes.grid5000.fr
client root@paradoxe-3.rennes.g5k paradoxe-3.rennes.grid5000.fr
```

The first hostname is used for SSH. The second hostname is used by Flower/gRPC at runtime.

Before continuing, verify:

- the selected mode: `local`, `remote`, or `g5k`;
- SSH connectivity to every node;
- the runtime hostname expected by Flower/gRPC;
- the remote project directory;
- the virtual environment activation path;
- whether the experiment requires distributed clients or server-only execution.

[Back to recommended execution order](#recommended-execution-order)

---

## 2. Install/setup remote nodes

Use the setup script to install system dependencies, clone the repository, create the virtual environment, install requirements, install the MetaCS-FL package, and optionally install PowerJoular.

Script:

```txt
scripts/setup/setup_remote_metacsfl_node.sh
```

README:

```txt
scripts/setup/setup_remote_metacsfl_node.README.md
```

Example Grid'5000 setup:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --branch v0.2.0 \
  --force-reclone true \
  --install-powerjoular true \
  --install-metacsfl-package true \
  --max-parallel-installs 8
```

If the repository is public or already available remotely:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth none \
  --branch v0.2.0 \
  --max-parallel-installs 8
```

For remote/Grid'5000 execution, always run the toy smoke test after setup and before running a full campaign.

[Back to recommended execution order](#recommended-execution-order)

---

## 3. Run toy smoke experiments

Before running a large campaign, run the toy smoke experiments.

Script:

```txt
scripts/execution/run_toy_smoke_experiments.sh
```

README:

```txt
scripts/execution/run_toy_smoke_experiments.README.md
```

Toy smoke folder:

```txt
toy_smoke_experiments/
```

The toy smoke campaign validates:

- distributed Flower execution;
- server-only execution;
- runtime config patching;
- gRPC patching;
- remote folder synchronization;
- result gathering;
- output merging;
- campaign state markers;
- resume behavior.

Example Grid'5000 smoke test:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --campaign-id toy_smoke_preflight_001
```

Expected summary:

```bash
cat campaign_runs/toy_smoke_preflight_001/summaries/campaign_summary.csv
```

Expected output:

```csv
campaign_id,total,completed,failed,running
toy_smoke_preflight_001,2,2,0,0
```

Only proceed to full campaigns after the smoke test succeeds.

[Back to recommended execution order](#recommended-execution-order)

---

## 4. Smoke test decision

After running the toy smoke experiments, decide whether to continue or debug the setup.

- If the smoke test fails, inspect logs and fix the environment/configuration.
- If the smoke test passes, continue to the experiment campaign.

[Back to recommended execution order](#recommended-execution-order)

---

### 4.1 If the smoke test fails: inspect logs and fix environment/configs

#### Check campaign status

```bash
cat campaign_runs/<campaign_id>/summaries/campaign_summary.csv
```

#### List completed experiments

```bash
ls campaign_runs/<campaign_id>/state/completed/
```

#### List failed experiments

```bash
ls campaign_runs/<campaign_id>/state/failed/
```

#### Inspect campaign logs

```bash
tail -f campaign_runs/<campaign_id>/logs/*.log
```

#### Inspect distributed logs

```bash
tail -f distributed_launch_logs/<run_id>/*.out
tail -f distributed_launch_logs/<run_id>/*.err
```

#### Inspect server-only logs

```bash
tail -f server_only_logs/<run_id>/server_only.out
tail -f server_only_logs/<run_id>/server_only.err
```

#### Check runtime config patching

For distributed runs, inspect:

```bash
grep -n "Runtime executor config patch result" campaign_runs/<campaign_id>/logs/<experiment_id>.log
```

Expected:

```txt
Runtime executor config patch result: patched=true
```

If it prints `patched=false`, check whether the executor config uses supported option names such as:

```txt
base_server_config_file
base_client_config_file
```


#### Check for out-of-memory (OOM) kills

When running many Flower clients on the same physical node, an experiment can fail even when the MetaCS-FL configuration and Flower setup are correct. This can happen when the selected clients, model architecture, dataset, batch size, local epochs, or assigned local workload cause the node to run out of RAM.

A common symptom is that the Flower server log stops during a training round, for example after:

```txt
[Server <id> | Round <r>] Starting the training phase...
```

and does not later show the aggregation message:

```txt
[Server <id> | Round <r> | Training Phase] Received <n> results and <m> failures.
```

Clients may then report gRPC connection errors such as:

```txt
StatusCode.UNAVAILABLE
failed to connect to all addresses
Connection refused
```

This does not necessarily mean that Flower timed out. It may mean that the Linux kernel killed one of the Python processes because the node ran out of memory.

Check the kernel log with:

```bash
dmesg -T | grep -i -E "killed process|out of memory|oom"
```

If the node ran out of memory, the output may contain messages similar to:

```txt
oom-kill:constraint=CONSTRAINT_NONE,...,task=python3
Out of memory: Killed process <pid> (python3) ...
```

The `anon-rss` value indicates how much resident anonymous memory the killed process was using. For example:

```txt
anon-rss:77538528kB
```

corresponds to roughly 74 GiB of RAM.

OOM failures are more likely in local or single-node executions with many concurrent clients, especially with memory-intensive models such as recurrent, attention-based, or deep neural architectures.

Possible mitigations:

- reduce the number of clients selected per round;
- reduce the number of clients executed on the same node;
- distribute clients across more nodes;
- reduce `batch_size`;
- reduce the number of local `epochs`;
- reduce the number of local samples/tasks assigned per selected client;
- increase the delay between launching client processes, for example `process_wait_time`;
- monitor memory usage while the experiment is running.

Useful monitoring command:

```bash
watch -n 1 'free -h; ps -eo pid,ppid,rss,cmd --sort=-rss | head -20'
```

If the kernel log reports an OOM kill, treat the client/server gRPC error as a consequence of memory pressure, not as the primary cause. Reduce peak parallel memory usage before assuming a Flower timeout or configuration bug.

After fixing the environment or configuration, re-run the toy smoke experiments before starting a full campaign.

[Back to recommended execution order](#recommended-execution-order)

---

### 4.2 If the smoke test passes: continue to the experiment campaign

If the toy smoke campaign finishes successfully, continue to the full experiment campaign.

Expected smoke summary:

```csv
campaign_id,total,completed,failed,running
toy_smoke_preflight_001,2,2,0,0
```

Proceed to:

[Generate or run the experiment campaign](#5-generate-or-run-the-experiment-campaign)

[Back to recommended execution order](#recommended-execution-order)

---

## 5. Generate or run the experiment campaign

Use this stage to generate a manifest, dry-run it, execute a full experiment pack, or resume an interrupted campaign.

[Back to recommended execution order](#recommended-execution-order)

---

### 5.1 Experiment packs

The current full static campaign pack is:

```txt
fgcs_2026_experiments
```

It includes:

```txt
static_client_availability/dp_impact_experiments
static_client_availability/performance_experiments
static_client_availability/scalability_experiments
static_client_availability/sensitivity_experiments
```

It excludes:

```txt
dynamic_client_availability/late_join_clients_experiments
Emotion-specific experiments
```

Distributed backend:

```txt
dp_impact_experiments
performance_experiments
```

Server-only backend:

```txt
scalability_experiments
sensitivity_experiments
```

[Back to recommended execution order](#recommended-execution-order)

---

### 5.2 Generate experiment packs

Experiment packs are generated separately from the campaign runner.

Script:

```txt
scripts/execution/generate_experiment_pack.py
```

README:

```txt
scripts/execution/generate_experiment_pack.README.md
```

Example:

```bash
python3 scripts/execution/generate_experiment_pack.py \
  --pack fgcs_2026_experiments \
  --experiments-root experiments \
  --output campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv \
  --validate true
```

The generated manifest defines exactly which experiments should run. This prevents accidental changes if files are later added to or removed from the `experiments/` folder.

[Back to recommended execution order](#recommended-execution-order)

---

### 5.3 Dry-run the FGCS 2026 pack

Use this to generate and validate the campaign manifest without executing experiments.

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_dry_run_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --dry-run true
```

This generates:

```txt
campaign_runs/fgcs_2026_dry_run_001/campaign_manifest.csv
campaign_runs/fgcs_2026_dry_run_001/campaign_manifest.summary.json
campaign_runs/fgcs_2026_dry_run_001/campaign_manifest.rows.usv
```

[Back to recommended execution order](#recommended-execution-order)

---

### 5.4 Run many experiments as a campaign

Use this to execute a full experiment pack with fault tolerance.

Script:

```txt
scripts/execution/run_many_distributed_experiments.sh
```

README:

```txt
scripts/execution/run_many_distributed_experiments.README.md
```

The runner uses a manifest-driven design. Each campaign has:

```txt
campaign_runs/<campaign_id>/
├── campaign_manifest.csv
├── campaign_manifest.rows.usv
├── logs/
├── state/
│   ├── completed/
│   ├── failed/
│   └── running/
└── summaries/
```

Completed experiments are skipped if the same campaign is resumed.

Run the FGCS 2026 campaign:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install false \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

[Back to recommended execution order](#recommended-execution-order)

---

### 5.5 Resume the campaign

If the campaign crashes or is interrupted, rerun the same command with the same campaign id:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install false \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 8
```

Completed experiments will be skipped. Failed or interrupted experiments will be retried by default.

[Back to recommended execution order](#recommended-execution-order)

---

## 6. Merge distributed outputs

Distributed runs gather per-node outputs under:

```txt
gathered_results/<run_id>/
```

The merge script consolidates them into:

```txt
merged_results/<run_id>/
```

Script:

```txt
scripts/post_execution/merge_distributed_outputs.py
```

README:

```txt
scripts/post_execution/merge_distributed_outputs.README.md
```

Example:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/g5k_test_001 \
  merged_results/g5k_test_001 \
  --clean-output true
```

The one-experiment and many-experiment runners call this automatically for distributed experiments.

[Back to recommended execution order](#recommended-execution-order)

---

## 7. Run analysis scripts

Analysis scripts are under:

```txt
scripts/analysis/
```

Folder README:

```txt
scripts/analysis/README.md
```

Available script READMEs:

```txt
scripts/analysis/plot_dp_impact_distributions.README.md
scripts/analysis/plot_performance_results.README.md
scripts/analysis/scalability_analysis.README.md
scripts/analysis/summarize_performance_results.README.md
```

[Back to recommended execution order](#recommended-execution-order)

---

### 7.1 DP impact analysis

Script:

```txt
scripts/analysis/plot_dp_impact_distributions.py
```

Example:

```bash
python3 scripts/analysis/plot_dp_impact_distributions.py \
  --non-private-data-distribution-folder results/static_client_availability/dp_impact_results/no_privacy/data_distribution \
  --differentially-private-data-distribution-folder results/static_client_availability/dp_impact_results/differential_privacy/data_distribution \
  --root-analysis-folder analysis_results
```

[Back to recommended execution order](#recommended-execution-order)

---

### 7.2 Performance plots

Script:

```txt
scripts/analysis/plot_performance_results.py
```

Example:

```bash
python3 scripts/analysis/plot_performance_results.py \
  --performance-results-folder results/static_client_availability/performance_results \
  --root-analysis-folder analysis_results \
  --experiment-tag performance \
  --dataset-name cifar_10 \
  --dataset-distribution iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,oort,mec,ecmtc,divfl,ecsm,metacsfl
```

[Back to recommended execution order](#recommended-execution-order)

---

### 7.3 Scalability analysis

Script:

```txt
scripts/analysis/scalability_analysis.py
```

Example:

```bash
python3 scripts/analysis/scalability_analysis.py \
  --root-results-folder results/static_client_availability \
  --root-analysis-folder analysis_results
```

[Back to recommended execution order](#recommended-execution-order)

---

### 7.4 Performance summary

Script:

```txt
scripts/analysis/summarize_performance_results.py
```

Example:

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder results/static_client_availability/performance_results \
  --dataset-name cifar_10 \
  --dataset-distribution iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,oort,mec,ecmtc,divfl,ecsm,metacsfl \
  --baseline fedavg
```

[Back to recommended execution order](#recommended-execution-order)

---

## 8. Export plots, tables, and metrics

Use the outputs generated by the analysis scripts to export figures, tables, summaries, and metrics.

Typical final outputs are written under:

```txt
analysis_results/
```

This stage should include:

- final DP impact plots;
- performance plots;
- scalability plots or tables;
- performance summary tables;
- metrics and comparisons.

The exact files depend on which analysis scripts were executed and which experiment results are available.

[Back to recommended execution order](#recommended-execution-order)

---

## Optional execution modes

The following modes are useful for debugging, isolated tests, or experiments that do not require the full campaign runner.

These modes are intentionally outside the main numbered execution workflow.

- [Run one distributed experiment](#run-one-distributed-experiment)
- [Run server-only experiments](#run-server-only-experiments)
- [Launch distributed Flower directly](#launch-distributed-flower-directly)

---

### Run one distributed experiment

Use this when you want to run a single experiment configuration.

Script:

```txt
scripts/execution/run_one_distributed_experiment.sh
```

README:

```txt
scripts/execution/run_one_distributed_experiment.README.md
```

This wrapper runs:

1. optional setup;
2. distributed Flower launch;
3. merge of gathered outputs.

Example:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --custom-flower-executor-cfg metacs_fl/flower_executor/config/flower_executor.cfg \
  --custom-flower-server-cfg metacs_fl/server/config/flower_server.cfg \
  --custom-flower-client-cfg metacs_fl/client/config/flower_client.cfg \
  --patch-executor-config true \
  --patch-grpc-config true \
  --server-port 8080 \
  --max-parallel-remote-ops 8 \
  --repetitions 1 \
  --run-id g5k_test_001 \
  --clean-merge-output true
```

Outputs:

```txt
distributed_launch_logs/g5k_test_001/
gathered_results/g5k_test_001/
merged_results/g5k_test_001/
```

[Back to optional execution modes](#optional-execution-modes)

---

### Run server-only experiments

Some experiments do not require distributed Flower clients. For example, scalability and sensitivity experiments may run only on the server node.

Script:

```txt
scripts/execution/run_on_server_only.sh
```

README:

```txt
scripts/execution/run_on_server_only.README.md
```

Example:

```bash
bash scripts/execution/run_on_server_only.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --script toy_smoke_experiments/server_only/scalability/toy_scalability_smoke.py \
  --config-file toy_smoke_experiments/server_only/scalability/scalability_experiments_fedavg_smoke.cfg \
  --remote-output-dir results/toy_smoke_experiments/server_only/scalability_results/fedavg_smoke \
  --run-id server_only_test_001
```

[Back to optional execution modes](#optional-execution-modes)

---

### Launch distributed Flower directly

For lower-level control, use the launcher directly.

Script:

```txt
scripts/execution/launch_distributed_flower.sh
```

README:

```txt
scripts/execution/launch_distributed_flower.README.md
```

Example:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --patch-executor-config true \
  --patch-grpc-config true \
  --server-port 8080 \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

This is useful for debugging, but for most experiments prefer:

```txt
run_one_distributed_experiment.sh
```

or:

```txt
run_many_distributed_experiments.sh
```

[Back to optional execution modes](#optional-execution-modes)

---

## Quick command summary

### Setup

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --repo-auth token \
  --prompt-github-token true \
  --branch v0.2.0 \
  --force-reclone true \
  --max-parallel-installs 8
```

### Smoke test

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --campaign-id toy_smoke_preflight_001
```

### Full campaign dry-run

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_dry_run_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --dry-run true
```

### Full campaign

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install false \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

### Resume campaign

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install false \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 8
```

[Back to table of contents](#table-of-contents)

---

## Notes for Grid'5000

For Grid'5000 runs:

1. Make sure SSH works from your local machine:

```bash
ssh root@paradoxe-1.rennes.g5k
```

2. Make sure the runtime hostnames in `scripts/nodes.g5k.txt` are reachable from inside the Grid'5000 allocation.

3. Use `paradoxe-*.rennes.grid5000.fr` as runtime hostnames when full internal hostnames are needed.

4. Use short runtime hostnames such as `paradoxe-1` if the executor matches nodes using short hostnames.

5. Always run the toy smoke test first.

[Back to table of contents](#table-of-contents)

---

## Results

### FGCS 2026

The FGCS 2026 experiment results are available for download here:

- [Download `fgcs_2026_results.tar.xz`](https://osf.io/rejq9/files/sd4a6)

To download the archive directly from a terminal, run one of the following commands:

```bash
# Using curl
curl -L -o fgcs_2026_results.tar.xz https://osf.io/sd4a6/download

# Or using wget
wget -O fgcs_2026_results.tar.xz https://osf.io/sd4a6/download
```

The results folder was compressed as a `.tar.xz` archive using `tar` and `xz` with all available CPU threads enabled by `xz -T0`:

```bash
tar -c fgcs_2026_results | xz -T0 -v > fgcs_2026_results.tar.xz
```

The downloadable compressed archive has approximately **435.2 MiB**, while the uncompressed folder has approximately **4416.2 MiB**.

To extract the archive, run:

```bash
xz -dc fgcs_2026_results.tar.xz | tar -xv
```

For reference, the generic commands are:

```bash
# Compress a folder into a .tar.xz archive
tar -c folder_name | xz -T0 -v > folder_name.tar.xz

# Extract a .tar.xz archive with progress from tar
xz -dc folder_name.tar.xz | tar -xv
```

[Back to table of contents](#table-of-contents)

---

## Scientific Productions

### 1. MetaCS-FL: A Metaheuristic-Based Framework for Client Selection in Federated Learning Systems

#### Authors

- Alan L. Nunes
- Cristina Boeres
- Laércio L. Pilla
- Lúcia M. A. Drummond

#### Affiliations

- Computing Institute, Fluminense Federal University, Niterói, Brazil
- University of Bordeaux, CNRS, Bordeaux INP, Inria, LaBRI, Talence, France

#### Abstract

Federated Learning (FL) enables collaborative training of distributed machine learning models, with each participant (client) using their own private data. In Cross-Device FL, clients are typically heterogeneous mobile or edge devices, often unreliable and with small and highly imbalanced local datasets. Selecting which clients participate is therefore critical, as poor choices can increase execution time, energy consumption, and reduce model accuracy. In this work, we propose MetaCS-FL, a client selection framework that supports different metaheuristics, initial solution strategies, and user-defined triggers for new selections. It leverages client profiling along with historical and current performance data to make efficient choices for both client participation and local data allocation. We evaluated MetaCS-FL in an extensive set of experiments, including comparisons with state-of-the-art algorithms. Using FedAvg as a baseline, MetaCS-FL reduced total time by up to 88.85% and energy consumption by 84.45% on CIFAR-10, and by 85.78% and 82.99%, respectively, on Fashion-MNIST, while achieving the target testing accuracy.

#### Citation

```bibtex
@misc{nunes2025metacsfl,
  title        = {{MetaCS-FL: A Metaheuristic-Based Framework for Client Selection in Federated Learning Systems}},
  author       = {Nunes, Alan L. and Boeres, Cristina and Pilla, Laércio L. and Drummond, Lúcia M. A.},
  year         = {2025},
  howpublished = {HAL},
  hal_id       = {hal-05170215},
  url          = {https://hal.science/hal-05170215}
}
```

[Back to table of contents](#table-of-contents)

---

## License

The source code in this repository is distributed under the CeCILL-C Free Software License Agreement, Version 1.0.

CeCILL-C is a free software license governed by French law. It grants users broad rights to use, modify, and redistribute the software, subject to the terms of the license. The full license text is provided in the repository license file.

[Back to table of contents](#table-of-contents)