# Toy Smoke Experiment Runner

This document explains how to use:

```txt
scripts/execution/run_toy_smoke_experiments.sh
```

to run a small validation campaign before launching larger MetaCS-FL experiment campaigns.

The toy smoke runner is a wrapper around:

```txt
scripts/execution/run_many_distributed_experiments.sh
```

It runs the manifest:

```txt
toy_smoke_experiments/campaign_manifest.csv
```

and validates both execution backends:

```txt
distributed_flower
server_only_python
```

---

## 1. Purpose

Use this script as a preflight check before running full experiments.

It helps validate that:

- node files are correct;
- SSH access works;
- the remote project exists;
- the virtual environment works;
- custom Flower configs are copied correctly;
- runtime gRPC config patching works;
- runtime executor config patching works;
- distributed Flower execution works;
- result gathering and merging work;
- server-only execution works;
- campaign state markers work;
- campaign resume behavior works.

This script is especially useful before running:

```txt
fgcs_2026_experiments
```

or any larger campaign.

---

## 2. Required folder

The repository should contain:

```txt
toy_smoke_experiments/
├── README.md
├── campaign_manifest.csv
├── campaign_manifest.distributed_only.csv
├── campaign_manifest.server_only_only.csv
├── campaign_manifest.summary.json
├── distributed/
│   └── cifar_10_iid_fedavg/
│       ├── fedavg.cfg
│       ├── fedavg_server.cfg
│       └── client.cfg
└── server_only/
    └── scalability/
        ├── scalability_experiments_fedavg_smoke.cfg
        └── toy_scalability_smoke.py
```

This folder is intentionally outside:

```txt
experiments/
```

because it is not part of the paper experiment set.

Keep this folder tracked in Git.

Do not ignore it in `.gitignore`.

---

## 3. Generated folders

The toy smoke runner may generate:

```txt
campaign_runs/
distributed_launch_logs/
gathered_results/
merged_results/
server_only_logs/
server_only_results/
```

These are runtime outputs and should be ignored by Git.

Recommended `.gitignore` block:

```gitignore
# Experiment/runtime outputs
output/
output_execution/
output_simulation/
setup_logs/
distributed_launch_logs/
campaign_runs/
gathered_results/
merged_results/
server_only_logs/
server_only_results/
.distributed_runtime_configs/
```

---

## 4. Script location

```txt
scripts/execution/run_toy_smoke_experiments.sh
```

Run it from the project root folder.

---

## 5. Usage

Basic form:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file <nodes-file>
```

Example:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate
```

---

## 6. Parameters

| Parameter | Description | Default |
|---|---|---|
| `--nodes-file FILE` | Node description file. | Required |
| `--campaign-id ID` | Campaign id for the smoke run. | `toy_smoke_<timestamp>` |
| `--toy-folder DIR` | Local toy smoke experiments folder. | `toy_smoke_experiments` |
| `--manifest FILE` | Smoke campaign manifest. | `<toy-folder>/campaign_manifest.csv` |
| `--remote-project-dir DIR` | Project directory on target nodes. | `/root/metacs-fl` |
| `--remote-venv-activate FILE` | Virtual environment activation script. | `<remote-project-dir>/.venv/bin/activate` |
| `--install true\|false` | Whether distributed experiments should run setup first. | `false` |
| `--sync-toy-folder true\|false` | Whether to sync `toy_smoke_experiments/` to remote nodes before running. | `true` |
| `--max-parallel-remote-ops N` | Parallelism for distributed launcher remote operations. | `8` |
| `--max-parallel-syncs N` | Parallelism for syncing the toy folder. | `8` |
| `--repetitions N` | Number of repetitions for distributed smoke experiment. | `1` |
| `--dry-run true\|false` | Validate/print commands without executing. | `false` |
| `--run-many-script FILE` | Path to `run_many_distributed_experiments.sh`. | `scripts/execution/run_many_distributed_experiments.sh` |
| `--python-bin BIN` | Python executable. | `python3` |
| `--ssh-options "OPTIONS"` | Extra SSH options. | Empty |
| `--rsync-options "OPTIONS"` | Extra rsync options. | `-az --delete` |

---

## 7. Local smoke test

Use this to test on the local machine:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --campaign-id toy_smoke_local_001
```

In local mode:

- no SSH is used;
- the toy folder is not synced;
- the existing local `toy_smoke_experiments/` folder is used.

---

## 8. Grid'5000 smoke test

Use this to test on Grid'5000:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --campaign-id toy_smoke_g5k_001
```

By default, the script syncs:

```txt
toy_smoke_experiments/
```

to each unique remote target under:

```txt
/root/metacs-fl/toy_smoke_experiments/
```

This is necessary because the server-only smoke script is executed on the server node.

---

## 9. Generic remote smoke test

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.remote.txt \
  --remote-project-dir /home/ubuntu/metacs-fl \
  --remote-venv-activate /home/ubuntu/metacs-fl/.venv/bin/activate \
  --campaign-id toy_smoke_remote_001
```

---

## 10. Dry-run

Use dry-run mode to check the command and manifest without running experiments.

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --dry-run true
```

In dry-run mode:

- remote toy-folder sync is skipped;
- the campaign runner is called with `--dry-run true`;
- no experiment is executed.

---

## 11. Running only distributed or only server-only smoke tests

The toy smoke folder contains optional manifests:

```txt
toy_smoke_experiments/campaign_manifest.distributed_only.csv
toy_smoke_experiments/campaign_manifest.server_only_only.csv
```

Run only the distributed smoke test:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --manifest toy_smoke_experiments/campaign_manifest.distributed_only.csv \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --campaign-id toy_smoke_distributed_only_001
```

Run only the server-only smoke test:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --manifest toy_smoke_experiments/campaign_manifest.server_only_only.csv \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --campaign-id toy_smoke_server_only_001
```

---

## 12. Resume behavior

The smoke runner uses the same campaign state system as the full campaign runner.

For a campaign id:

```txt
toy_smoke_g5k_001
```

state markers are written under:

```txt
campaign_runs/toy_smoke_g5k_001/state/
```

If you rerun the same command with the same campaign id:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --campaign-id toy_smoke_g5k_001
```

completed experiments are skipped and failed experiments are retried.

---

## 13. Output locations

For campaign id:

```txt
toy_smoke_g5k_001
```

campaign state and logs:

```txt
campaign_runs/toy_smoke_g5k_001/
```

distributed logs:

```txt
distributed_launch_logs/toy_smoke_g5k_001__<distributed_experiment_id>/
```

gathered distributed results:

```txt
gathered_results/toy_smoke_g5k_001__<distributed_experiment_id>/
```

merged distributed results:

```txt
merged_results/toy_smoke_g5k_001__<distributed_experiment_id>/
```

server-only logs:

```txt
server_only_logs/toy_smoke_g5k_001__<server_only_experiment_id>/
```

server-only results:

```txt
server_only_results/toy_smoke_g5k_001__<server_only_experiment_id>/
```

---

## 14. Checking success

After the smoke run, check:

```bash
cat campaign_runs/<campaign_id>/summaries/campaign_summary.csv
```

Expected result:

```csv
campaign_id,total,completed,failed,running
<campaign_id>,2,2,0,0
```

For distributed-only smoke tests:

```csv
<campaign_id>,1,1,0,0
```

For server-only-only smoke tests:

```csv
<campaign_id>,1,1,0,0
```

---

## 15. Common issues

### 15.1 Toy folder missing on remote node

If the server-only experiment fails with:

```txt
python3: can't open file '/root/metacs-fl/toy_smoke_experiments/...': No such file or directory
```

then the toy folder was not synced.

Use:

```bash
--sync-toy-folder true
```

This is the default for remote and Grid'5000 modes.

---

### 15.2 SSH works manually but sync fails

Check the first hostname column in the node file.

For example:

```txt
server root@paradoxe-16.rennes.g5k paradoxe-16.rennes.grid5000.fr
```

requires this to work:

```bash
ssh root@paradoxe-16.rennes.g5k
```

You can pass SSH options:

```bash
--ssh-options "-o StrictHostKeyChecking=no"
```

---

### 15.3 Distributed smoke fails during server startup

Check:

```bash
cat distributed_launch_logs/<run_id>/*.out
cat distributed_launch_logs/<run_id>/*.err
```

Common causes:

- runtime executor config patching failed;
- runtime gRPC config patching failed;
- server config is missing required logging keys;
- clients cannot reach the server runtime hostname.

The distributed smoke test is useful because it catches these issues before the full campaign.

---

### 15.4 Server-only smoke fails

Check:

```bash
cat server_only_logs/<run_id>/server_only.out
cat server_only_logs/<run_id>/server_only.err
```

Common causes:

- toy folder was not synced to the remote project root;
- virtual environment path is wrong;
- server-only script path is wrong;
- output directory collection path is wrong.

---

## 16. Recommended preflight workflow

Before running a full campaign:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --campaign-id toy_smoke_preflight_001
```

Then confirm:

```bash
cat campaign_runs/toy_smoke_preflight_001/summaries/campaign_summary.csv
```

Expected:

```csv
campaign_id,total,completed,failed,running
toy_smoke_preflight_001,2,2,0,0
```

Only after this succeeds, run the larger campaign:

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
