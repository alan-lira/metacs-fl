# Many-Experiment Distributed Campaign Runner

This document explains how to use:

```txt
scripts/execution/run_many_distributed_experiments.sh
```

to run a resumable campaign composed of many MetaCS-FL experiments.

The script is **manifest-driven**. It does not define experiment packs itself. Instead, it either:

1. reads an existing `campaign_manifest.csv`; or
2. calls `scripts/execution/generate_experiment_pack.py` to generate one.

The runner then executes each manifest row using one of two backends:

```txt
distributed_flower
server_only_python
```

---

## 1. Script location

The script is located at:

```txt
scripts/execution/run_many_distributed_experiments.sh
```

Related scripts:

```txt
scripts/execution/generate_experiment_pack.py
scripts/execution/run_one_distributed_experiment.sh
scripts/execution/run_on_server_only.sh
scripts/execution/launch_distributed_flower.sh
scripts/post_execution/merge_distributed_outputs.py
scripts/setup/setup_remote_metacsfl_node.sh
```

All examples in this document assume commands are executed from the project root folder.

---

## 2. Main idea

The runner executes a campaign from a manifest.

A campaign has:

```txt
campaign_runs/<campaign_id>/
├── campaign_manifest.csv
├── campaign_manifest.rows.usv
├── campaign_manifest.summary.json
├── logs/
├── state/
│   ├── completed/
│   ├── failed/
│   └── running/
└── summaries/
```

The manifest defines exactly which experiments should run.

The runner handles:

- reading the manifest;
- preserving empty manifest fields correctly;
- skipping completed experiments;
- retrying failed experiments if configured;
- retrying interrupted/running experiments if configured;
- launching distributed experiments through `run_one_distributed_experiment.sh`;
- launching server-only experiments through `run_on_server_only.sh`;
- writing per-experiment logs;
- writing completion/failure markers;
- writing a campaign summary.

---

## 3. Manifest-driven execution

The runner expects a CSV manifest with this exact header:

```csv
experiment_id,backend,group,num_clients,dataset,distribution,approach,executor_cfg,server_cfg,client_cfg,server_script,server_config,remote_output_dir
```

Each row represents one experiment.

### 3.1 Distributed Flower rows

Distributed rows use:

```txt
backend = distributed_flower
```

Required fields:

```txt
experiment_id
backend
group
num_clients
dataset
distribution
approach
executor_cfg
server_cfg
client_cfg
```

Unused server-only fields must remain empty:

```txt
server_script
server_config
remote_output_dir
```

Example:

```csv
performance__50_clients__cifar_10__iid__fedavg,distributed_flower,performance_experiments,50_clients,cifar_10,iid,fedavg,experiments/static_client_availability/performance_experiments/50_clients/cifar_10/iid/fedavg.cfg,experiments/static_client_availability/performance_experiments/50_clients/cifar_10/iid/fedavg_server.cfg,experiments/static_client_availability/performance_experiments/50_clients/cifar_10/iid/client.cfg,,,
```

Distributed rows are executed through:

```txt
scripts/execution/run_one_distributed_experiment.sh
```

which internally calls:

```txt
scripts/execution/launch_distributed_flower.sh
scripts/post_execution/merge_distributed_outputs.py
```

---

### 3.2 Server-only rows

Server-only rows use:

```txt
backend = server_only_python
```

Required fields:

```txt
experiment_id
backend
group
approach
server_script
```

Optional fields:

```txt
server_config
remote_output_dir
```

Distributed config fields must remain empty:

```txt
num_clients
dataset
distribution
executor_cfg
server_cfg
client_cfg
```

Example:

```csv
scalability__fedavg,server_only_python,scalability_experiments,,,,fedavg,,,,experiments/static_client_availability/scalability_experiments/scalability_experiments.py,experiments/static_client_availability/scalability_experiments/scalability_experiments_fedavg.cfg,results/static_client_availability/scalability_results/fedavg
```

Server-only rows are executed through:

```txt
scripts/execution/run_on_server_only.sh
```

The runner passes:

```bash
--script <server_script>
--config-file <server_config>
--remote-output-dir <remote_output_dir>
```

when those fields are present.

---

## 4. Internal manifest row parsing

The runner converts the CSV manifest into an internal file:

```txt
campaign_runs/<campaign_id>/campaign_manifest.rows.usv
```

This file uses the ASCII **Unit Separator** character:

```txt
\x1f
```

as the internal delimiter.

This is important because server-only rows intentionally contain empty CSV fields. Older versions used TSV and Bash `IFS=$'\t'`, but Bash treats tabs as whitespace and collapses repeated separators. That caused empty fields to be lost and server-only rows to shift, producing errors such as:

```txt
--script ''
```

The current runner uses `campaign_manifest.rows.usv` so empty manifest fields are preserved correctly.

You normally do not edit this `.usv` file manually. It is generated from the CSV manifest each time the campaign runner starts.

---

## 5. Fault tolerance and resume behavior

The campaign runner uses marker files under:

```txt
campaign_runs/<campaign_id>/state/
```

Structure:

```txt
state/
├── completed/
├── failed/
└── running/
```

Each experiment gets marker files such as:

```txt
state/completed/<experiment_id>.done.json
state/failed/<experiment_id>.failed.json
state/running/<experiment_id>.running.json
```

### 5.1 Completed experiments

If a completed marker exists, the experiment is skipped by default.

Example:

```txt
[SKIP] performance__50_clients__cifar_10__iid__fedavg already completed.
```

### 5.2 Failed experiments

Failed experiments are retried by default:

```bash
--rerun-failed true
```

To prevent retrying failed experiments:

```bash
--rerun-failed false
```

### 5.3 Interrupted/running experiments

If the campaign was interrupted while an experiment was running, a `.running.json` marker may remain.

Such experiments are retried by default:

```bash
--rerun-running true
```

To prevent retrying interrupted experiments:

```bash
--rerun-running false
```

### 5.4 Force rerun

To rerun everything regardless of markers:

```bash
--force-rerun true
```

---

## 6. Command-line usage

Basic usage with an existing manifest:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest campaign_runs/<campaign_id>/campaign_manifest.csv \
  --campaign-id <campaign_id> \
  --nodes-file scripts/nodes.g5k.txt
```

Usage with pack generation:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id <campaign_id> \
  --nodes-file scripts/nodes.g5k.txt
```

The pack generation mode internally calls:

```txt
scripts/execution/generate_experiment_pack.py
```

---

## 7. Parameters

### 7.1 Manifest parameters

| Parameter | Description | Default |
|---|---|---|
| `--manifest FILE` | Existing campaign manifest CSV. | Not set |
| `--pack PACK` | Pack name to generate using `generate_experiment_pack.py`. | Not set |
| `--experiments-root DIR` | Experiments root used by the generator. | `experiments` |
| `--generate-pack-script FILE` | Pack generator script. | `scripts/execution/generate_experiment_pack.py` |

You must provide either:

```bash
--manifest FILE
```

or:

```bash
--pack PACK
```

but not both.

---

### 7.2 Required campaign parameters

| Parameter | Description | Default |
|---|---|---|
| `--campaign-id ID` | Campaign identifier. | Required |
| `--nodes-file FILE` | Node description file shared by setup and launch scripts. | Required |

---

### 7.3 Main execution parameters

| Parameter | Description | Default |
|---|---|---|
| `--remote-project-dir DIR` | Project directory on target nodes. | `/root/metacs-fl` |
| `--remote-venv-activate FILE` | Virtual environment activation script on target nodes. | `<remote-project-dir>/.venv/bin/activate` |
| `--install true\|false` | Whether `run_one_distributed_experiment.sh` should run setup before distributed experiments. | `false` |
| `--dry-run true\|false` | Generate/validate the manifest and print summary without executing experiments. | `false` |

---

### 7.4 Fault-tolerance parameters

| Parameter | Description | Default |
|---|---|---|
| `--force-rerun true\|false` | Run experiments even when completed markers exist. | `false` |
| `--rerun-failed true\|false` | Retry experiments with failed markers. | `true` |
| `--rerun-running true\|false` | Retry experiments left in running state by a previous interrupted campaign. | `true` |

---

### 7.5 Parallelism parameters

| Parameter | Description | Default |
|---|---|---|
| `--max-parallel-experiments N` | Number of experiments to run in parallel. | `1` |
| `--repetitions N` | Passed to distributed experiments. | `1` |
| `--max-parallel-installs N` | Passed to setup through `run_one_distributed_experiment.sh`. | `4` |
| `--max-parallel-remote-ops N` | Passed to distributed launcher through `run_one_distributed_experiment.sh`. | `8` |

For distributed Flower experiments, the recommended default is:

```bash
--max-parallel-experiments 1
```

because each distributed experiment typically consumes the full node allocation.

---

### 7.6 Script path parameters

| Parameter | Description | Default |
|---|---|---|
| `--run-one-script FILE` | One-experiment pipeline script. | `scripts/execution/run_one_distributed_experiment.sh` |
| `--run-server-only-script FILE` | Server-only execution script. | `scripts/execution/run_on_server_only.sh` |

---

### 7.7 Common parameters

| Parameter | Description | Default |
|---|---|---|
| `--python-bin BIN` | Python executable. | `python3` |
| `--ssh-options "OPTIONS"` | Extra SSH options. | Empty |
| `--rsync-options "OPTIONS"` | Extra rsync options. | `-az` |

---

### 7.8 Repository/setup parameters for distributed experiments

These are passed to `run_one_distributed_experiment.sh` and then to the setup script when:

```bash
--install true
```

| Parameter | Description | Default |
|---|---|---|
| `--repo-url URL` | Repository URL. | `https://github.com/alan-lira/metacs-fl.git` |
| `--branch BRANCH` | Git branch. | `main` |
| `--repo-auth none\|token\|ssh` | Repository authentication mode. | `none` |
| `--prompt-github-token true\|false` | Whether setup prompts for a GitHub token. | `true` |
| `--force-reclone true\|false` | Whether setup reclones the repository. | `true` |
| `--install-powerjoular true\|false` | Whether setup installs PowerJoular. | `true` |
| `--append-venv-to-bashrc true\|false` | Whether setup appends venv activation to `.bashrc`. | `true` |
| `--install-metacsfl-package true\|false` | Whether setup installs MetaCS-FL with `pip3 install -e`. | `true` |

---

### 7.9 Output root parameters

| Parameter | Description | Default |
|---|---|---|
| `--output-dir DIR` | Base directory for all local campaign artifacts. If provided, the default roots become `DIR/campaign_runs`, `DIR/distributed_launch_logs`, `DIR/gathered_results`, `DIR/merged_results`, `DIR/server_only_logs`, and `DIR/server_only_results`. Explicit root arguments below override this default. | Not set |
| `--campaign-root DIR` | Root for campaign state and logs. | `campaign_runs` |
| `--distributed-log-root DIR` | Root for distributed launch logs. | `distributed_launch_logs` |
| `--gather-root DIR` | Root for gathered distributed outputs. | `gathered_results` |
| `--merge-root DIR` | Root for merged distributed outputs. | `merged_results` |
| `--server-only-log-root DIR` | Root for server-only logs. | `server_only_logs` |
| `--server-only-output-root DIR` | Root for collected server-only outputs. | `server_only_results` |

---

## 8. Output structure

For campaign id:

```txt
fgcs_2026_experiments_001
```

the campaign runner creates the campaign directory under `<campaign-root>`:

```txt
<campaign-root>/fgcs_2026_experiments_001/
├── campaign_manifest.csv
├── campaign_manifest.rows.usv
├── campaign_manifest.summary.json
├── logs/
│   ├── <experiment_id>.log
│   └── ...
├── state/
│   ├── completed/
│   │   ├── <experiment_id>.done.json
│   │   └── ...
│   ├── failed/
│   │   ├── <experiment_id>.failed.json
│   │   └── ...
│   └── running/
└── summaries/
    └── campaign_summary.csv
```

Distributed experiments also produce:

```txt
<distributed-log-root>/<campaign_id>__<experiment_id>/
<gather-root>/<campaign_id>__<experiment_id>/
<merge-root>/<campaign_id>__<experiment_id>/
```

Server-only experiments produce:

```txt
<server-only-log-root>/<campaign_id>__<experiment_id>/
<server-only-output-root>/<campaign_id>__<experiment_id>/
```

if `remote_output_dir` is provided in the manifest.

By default, these roots are:

```txt
campaign_runs
distributed_launch_logs
gathered_results
merged_results
server_only_logs
server_only_results
```

If `--output-dir DIR` is provided and no explicit root overrides are provided, the paths become:

```txt
DIR/campaign_runs/<campaign_id>/
DIR/distributed_launch_logs/<campaign_id>__<experiment_id>/
DIR/gathered_results/<campaign_id>__<experiment_id>/
DIR/merged_results/<campaign_id>__<experiment_id>/
DIR/server_only_logs/<campaign_id>__<experiment_id>/
DIR/server_only_results/<campaign_id>__<experiment_id>/
```

---

## 9. Dry-run mode

Use dry-run mode to generate or validate the manifest without executing anything.

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_dry_run_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --dry-run true
```

This creates:

```txt
<campaign-root>/fgcs_2026_dry_run_001/campaign_manifest.csv
<campaign-root>/fgcs_2026_dry_run_001/campaign_manifest.rows.usv
<campaign-root>/fgcs_2026_dry_run_001/campaign_manifest.summary.json
```

and exits without running experiments.

---

## 10. Supported pack: `fgcs_2026_experiments`

When using:

```bash
--pack fgcs_2026_experiments
```

the runner calls:

```txt
scripts/execution/generate_experiment_pack.py
```

to generate the campaign manifest.

This pack includes the static experiments for the FGCS 2026 campaign:

### 10.1 Distributed Flower experiments

#### Performance experiments

Included dimensions:

```txt
num_clients:
  50_clients
  100_clients

datasets:
  cifar_10
  fashion_mnist

distributions:
  iid
  non_iid

approaches:
  fedavg
  oort
  mec
  ecmtc
  divfl
  ecsm
  metacsfl
```

Emotion experiments are excluded.

#### DP impact experiments

Included dimensions:

```txt
num_clients:
  100_clients

dataset:
  cifar_10

distribution:
  non_iid

approaches:
  metacsfl
  metacsfl_no_privacy
```

### 10.2 Server-only experiments

#### Scalability experiments

Included approaches:

```txt
fedavg
oort
mec
ecmtc
divfl
ecsm
metacsfl_015
metacsfl_025
metacsfl_050
```

#### Sensitivity experiments

Included approach:

```txt
metacsfl
```

### 10.3 Excluded experiments

The pack excludes:

```txt
dynamic_client_availability/late_join_clients_experiments
performance_experiments involving Emotion
```

---

## 11. Example calls

### 11.1 Generate manifest and dry-run

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_dry_run_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --dry-run true
```

---

### 11.2 Run full campaign using generated pack

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

---

### 11.3 Run full campaign using a custom local output directory

Use `--output-dir` when you want all campaign state, logs, gathered results, merged results, and server-only outputs under the same local base directory.

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --output-dir /mnt/d/results \
  --install false \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

This creates paths such as:

```txt
/mnt/d/results/campaign_runs/fgcs_2026_experiments_001/
/mnt/d/results/distributed_launch_logs/fgcs_2026_experiments_001__<experiment_id>/
/mnt/d/results/gathered_results/fgcs_2026_experiments_001__<experiment_id>/
/mnt/d/results/merged_results/fgcs_2026_experiments_001__<experiment_id>/
```

---

### 11.4 Run full campaign with setup enabled

Use this for a fresh Grid'5000 allocation:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install true \
  --repo-auth token \
  --prompt-github-token true \
  --branch main \
  --force-reclone true \
  --max-parallel-installs 8 \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

---

### 11.5 Run using an existing manifest

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

---

### 11.6 Resume a campaign after interruption

Rerun the exact same command with the same campaign id:

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

---

### 11.7 Force rerun everything

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --force-rerun true
```

---

### 11.8 Do not retry failed experiments

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --rerun-failed false
```

---

## 12. Recommended workflow

### 12.1 First: dry-run the pack

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_dry_run_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --dry-run true
```

Inspect:

```bash
cat <campaign-root>/fgcs_2026_dry_run_001/campaign_manifest.csv
cat <campaign-root>/fgcs_2026_dry_run_001/campaign_manifest.summary.json
```

### 12.2 Then run the real campaign

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

### 12.3 Resume if needed

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

---

## 13. Monitoring progress

Campaign-level logs:

```bash
tail -f <campaign-root>/<campaign_id>/logs/*.log
```

Distributed launch logs:

```bash
tail -f <distributed-log-root>/<campaign_id>__<experiment_id>/*.out
tail -f <distributed-log-root>/<campaign_id>__<experiment_id>/*.err
```

Server-only logs:

```bash
tail -f <server-only-log-root>/<campaign_id>__<experiment_id>/server_only.out
tail -f <server-only-log-root>/<campaign_id>__<experiment_id>/server_only.err
```

Summary:

```bash
cat <campaign-root>/<campaign_id>/summaries/campaign_summary.csv
```

Completed markers:

```bash
ls <campaign-root>/<campaign_id>/state/completed/
```

Failed markers:

```bash
ls <campaign-root>/<campaign_id>/state/failed/
```

---

## 14. Common issues

### 14.1 `--script ''` in server-only command

This indicates either:

1. the manifest row is malformed; or
2. an old version of the runner parsed empty fields incorrectly.

The current runner uses:

```txt
campaign_manifest.rows.usv
```

with Unit Separator (`\x1f`) to preserve empty fields.

Check the generated command in:

```txt
campaign_runs/<campaign_id>/logs/<experiment_id>.log
```

A correct server-only command should contain:

```bash
--script experiments/.../some_script.py
```

not:

```bash
--script ''
```

Also inspect the manifest row:

```bash
python3 - <<'PY'
import csv
with open("campaign_runs/<campaign_id>/campaign_manifest.csv", newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        print(r)
PY
```

---

### 14.2 Server-only script not found on remote node

Example:

```txt
python3: can't open file '/root/metacs-fl/experiments/.../scalability_experiments.py': [Errno 2] No such file or directory
```

This means the path in the manifest is wrong for the remote repository.

Check locally:

```bash
find experiments -type f -name "*scalability*.py" | sort
```

Check remotely:

```bash
ssh root@paradoxe-16.rennes.g5k \
  "cd /root/metacs-fl && find experiments -type f -name '*scalability*.py' | sort"
```

Then update the manifest `server_script` column.

---

### 14.3 Distributed experiment fails but setup and SSH are fine

Inspect the launch logs:

```bash
cat distributed_launch_logs/<run_id>/*.out
cat distributed_launch_logs/<run_id>/*.err
```

If the launcher prints:

```txt
Runtime executor config patch result: patched=false
```

then the custom server/client config files may have been copied but not used by the executor. Check the executor config option names.

Current launcher supports:

```txt
base_server_config_file
base_client_config_file
```

as well as several generic server/client config option names.

---

### 14.4 Completed experiment is skipped

This is expected if a completed marker exists:

```txt
campaign_runs/<campaign_id>/state/completed/<experiment_id>.done.json
```

To rerun it:

```bash
--force-rerun true
```

or delete the marker manually.

---

### 14.5 Failed experiment is retried

This is the default:

```bash
--rerun-failed true
```

To skip failed experiments:

```bash
--rerun-failed false
```

---

### 14.6 Campaign was interrupted

If the campaign was interrupted, some experiments may have:

```txt
state/running/<experiment_id>.running.json
```

By default, they are retried:

```bash
--rerun-running true
```

---

### 14.7 Parallel experiments overload the allocation

The default is:

```bash
--max-parallel-experiments 1
```

Keep this default for distributed Flower campaigns unless you intentionally partition the node allocation.

Running multiple distributed Flower experiments at once on the same node set can cause:

- port conflicts;
- resource contention;
- overlapping output folders;
- confusing logs;
- failed gRPC connections.

---

## 15. Summary

For a dry-run:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_dry_run_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --dry-run true
```

For the real campaign:

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

To resume:

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
