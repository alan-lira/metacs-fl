# Experiment Pack Generator

This document explains how to use:

```txt
scripts/execution/generate_experiment_pack.py
```

to generate explicit campaign manifests for MetaCS-FL experiment packs.

The generator is intentionally separate from the campaign runner. It defines **what experiments belong to a pack**, while `run_many_distributed_experiments.sh` defines **how to execute and resume them**.

All examples assume commands are executed from the project root.

---

## 1. Script location

```txt
scripts/execution/generate_experiment_pack.py
```

Make it executable if needed:

```bash
chmod +x scripts/execution/generate_experiment_pack.py
```

---

## 2. Main idea

The generator produces a CSV manifest such as:

```txt
campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv
```

The manifest is the exact experimental plan. It explicitly lists:

- experiment id;
- backend type;
- experiment group;
- number of clients, dataset, distribution, and approach;
- config files for distributed Flower experiments;
- script/config/output information for server-only experiments.

This avoids accidentally running new, temporary, or deprecated folders just because they were added under `experiments/`.

---

## 3. Supported packs

Currently supported:

```txt
fgcs_2026_experiments
```

This pack includes:

### Distributed Flower experiments

```txt
static_client_availability/performance_experiments
static_client_availability/dp_impact_experiments
```

### Server-only experiments

```txt
static_client_availability/scalability_experiments
static_client_availability/sensitivity_experiments
```

### Excluded experiments

```txt
dynamic_client_availability/intermittent_client_availability_experiments
dynamic_client_availability/late_joining_clients_experiments
performance experiments involving Emotion
```

The excluded dynamic families can still be run through a custom manifest or with `run_one_distributed_experiment.sh`. See `experiments/README.md` for their directory structure and example commands.

---

## 4. `fgcs_2026_experiments` contents

### 4.1 Performance experiments

The pack explicitly includes:

```txt
client systems:
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

This gives:

```txt
2 client systems × 2 datasets × 2 distributions × 7 approaches = 56 experiments
```

Each performance experiment is a `distributed_flower` row.

---

### 4.2 DP impact experiments

The pack explicitly includes:

```txt
100_clients/cifar_10/non_iid

approaches:
  metacsfl
  metacsfl_no_privacy
```

This gives:

```txt
2 experiments
```

Each DP-impact experiment is a `distributed_flower` row.

---

### 4.3 Scalability experiments

The pack explicitly includes:

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

This gives:

```txt
9 experiments
```

Each scalability experiment is a `server_only_python` row.

---

### 4.4 Sensitivity experiments

The pack includes:

```txt
metacsfl
```

This gives:

```txt
1 experiment
```

The sensitivity experiment is a `server_only_python` row.

---

## 5. Expected manifest size

For `fgcs_2026_experiments`, the expected total is:

```txt
performance_experiments: 56
dp_impact_experiments:   2
scalability_experiments: 9
sensitivity_experiments: 1
--------------------------------
total:                  68
```

---

## 6. Command-line usage

Basic form:

```bash
python3 scripts/execution/generate_experiment_pack.py \
  --pack fgcs_2026_experiments \
  --experiments-root experiments \
  --output campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv \
  --validate true
```

Full form:

```bash
python3 scripts/execution/generate_experiment_pack.py \
  --pack fgcs_2026_experiments \
  --experiments-root experiments \
  --output campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv \
  --summary-output campaign_runs/fgcs_2026_experiments_001/campaign_manifest.summary.json \
  --validate true
```

---

## 7. Parameters

| Parameter | Description | Default |
|---|---|---|
| `--pack` | Experiment pack to generate. | Required |
| `--experiments-root` | Root folder containing experiment definitions. | `experiments` |
| `--output` | Output manifest CSV path. | Required |
| `--summary-output` | Optional JSON summary path. | `<output>.summary.json` |
| `--validate true\|false` | Whether to fail if referenced files are missing. | `true` |

---

## 8. Manifest columns

The generated CSV has these columns:

```csv
experiment_id,backend,group,num_clients,dataset,distribution,approach,executor_cfg,server_cfg,client_cfg,server_script,server_config,remote_output_dir
```

### Distributed rows

For `backend=distributed_flower`, these fields are used:

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

These rows are later executed by:

```txt
scripts/execution/run_one_distributed_experiment.sh
```

### Server-only rows

For `backend=server_only_python`, these fields are used:

```txt
experiment_id
backend
group
approach
server_script
server_config
remote_output_dir
```

These rows are later executed by:

```txt
scripts/execution/run_on_server_only.sh
```

---

## 9. Output files

The generator writes:

```txt
campaign_manifest.csv
campaign_manifest.csv.summary.json
```

Example:

```txt
campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv
campaign_runs/fgcs_2026_experiments_001/campaign_manifest.csv.summary.json
```

The summary JSON includes:

- pack name;
- experiment root;
- total number of rows;
- count by backend;
- count by group;
- validation problems, if any.

---

## 10. Validation

With:

```bash
--validate true
```

the generator fails if any referenced file does not exist.

For distributed rows, it checks:

```txt
executor_cfg
server_cfg
client_cfg
```

For server-only rows, it checks:

```txt
server_script
server_config, when provided
```

To generate a manifest even if some files are missing:

```bash
--validate false
```

This is useful only for debugging or while creating new experiment folders.

---

## 11. Recommended dry-run workflow

Generate and inspect the manifest:

```bash
mkdir -p campaign_runs/fgcs_2026_dry_run_001

python3 scripts/execution/generate_experiment_pack.py \
  --pack fgcs_2026_experiments \
  --experiments-root experiments \
  --output campaign_runs/fgcs_2026_dry_run_001/campaign_manifest.csv \
  --validate true
```

Inspect:

```bash
column -s, -t < campaign_runs/fgcs_2026_dry_run_001/campaign_manifest.csv | less -S
cat campaign_runs/fgcs_2026_dry_run_001/campaign_manifest.csv.summary.json
```

Then execute with:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest campaign_runs/fgcs_2026_dry_run_001/campaign_manifest.csv \
  --campaign-id fgcs_2026_dry_run_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --install false \
  --max-parallel-experiments 1
```

---

## 12. Common issues

### Missing config files

If validation fails, the generator prints every missing file. Fix the expected folder/config names or run with `--validate false` only for debugging.

### Pack changed unintentionally

The pack is explicit in the generator code. Adding files under `experiments/` does not automatically add them to the pack. Update `generate_experiment_pack.py` intentionally when the scientific experiment set changes.

### Existing manifest should be preserved

For reproducibility, keep the generated manifest under `campaign_runs/<campaign_id>/`. The campaign runner can execute an existing manifest using `--manifest`, independent of later folder changes.
