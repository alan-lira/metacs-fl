# Scripts

This folder contains setup, execution, post-execution, and analysis utilities for MetaCS-FL experiments.

All commands in the script READMEs assume they are executed from the project root.

## Shared node files

The node files are shared by setup and execution scripts:

```txt
scripts/nodes.local.txt
scripts/nodes.remote.txt
scripts/nodes.g5k.txt
```

Each node file uses this format:

```txt
mode <local|remote|g5k>

<role> <ssh_target> <runtime_host>
```

- `role` is normally `server` or `client`.
- `ssh_target` is used by setup and execution scripts to reach a node.
- `runtime_host` is used by Flower/gRPC during distributed execution.
- In `local` mode, use `-` as the SSH target.

## Subfolders

- `setup/`: install and configure MetaCS-FL on local, remote, or Grid'5000 nodes.
- `execution/`: run toy smoke tests, launch one or many experiments, run server-only experiments, and generate campaign manifests.
- `post_execution/`: merge gathered per-node outputs.
- `analysis/`: generate plots, numerical summaries, CSV exports, and LaTeX tables from experiment results.

## Recommended workflow

### 1. Set up the nodes

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --max-parallel-installs 8
```

### 2. Run the toy smoke campaign

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate
```

Use the smoke test to verify node access, environment setup, distributed execution, output gathering/merging, server-only execution, and campaign state handling before a large run.

### 3. Run one experiment or a campaign

One distributed experiment:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id g5k_test_001 \
  --custom-flower-executor-cfg metacs_fl/flower_executor/config/flower_executor.cfg \
  --custom-flower-server-cfg metacs_fl/server/config/flower_server.cfg \
  --custom-flower-client-cfg metacs_fl/client/config/flower_client.cfg
```

Manifest-driven campaign:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --pack fgcs_2026_experiments \
  --campaign-id fgcs_2026_experiments_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --install false \
  --max-parallel-experiments 1
```

Dynamic experiments that use multiple executor blocks can select a block with:

```bash
--execution-blocks Execution_1_N
```

Omit this option to run all execution blocks defined in the executor configuration.

### 4. Merge outputs when not using the one-experiment pipeline

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/<run_id> \
  merged_results/<run_id> \
  --clean-output true
```

### 5. Analyze the results

Use the scripts under `scripts/analysis/`. Performance analysis supports static client availability, intermittent availability, and late-joining clients.

## README map

### Setup

- `setup/README.md`: setup folder overview.
- `setup/setup_remote_metacsfl_node.README.md`: local, remote, and Grid'5000 setup script.

### Execution

- `execution/README.md`: execution folder overview.
- `execution/run_toy_smoke_experiments.README.md`: recommended preflight smoke campaign.
- `execution/launch_distributed_flower.README.md`: low-level distributed Flower launcher.
- `execution/run_one_distributed_experiment.README.md`: one-experiment pipeline: optional setup, launch, and merge.
- `execution/generate_experiment_pack.README.md`: explicit campaign-manifest generator.
- `execution/run_many_distributed_experiments.README.md`: resumable manifest-driven campaign runner.
- `execution/run_on_server_only.README.md`: server-only execution backend.

### Post-execution

- `post_execution/README.md`: post-execution folder overview.
- `post_execution/merge_distributed_outputs.README.md`: safe output merger.

### Analysis

- `analysis/README.md`: analysis folder overview and result layouts.
- `analysis/plot_dp_impact_distributions.README.md`: differential-privacy distribution plots.
- `analysis/plot_performance_results.README.md`: static and dynamic performance plots.
- `analysis/scalability_analysis.README.md`: scalability plots and tables.
- `analysis/summarize_performance_results.README.md`: static and dynamic performance summaries, dropout metrics, and late-join engagement exports.

### Experiment configurations and smoke-test data

- `../experiments/README.md`: static and dynamic experiment families, configuration roles, and custom campaign examples.
- `../toy_smoke_experiments/README.md`: toy campaign contents, expected outputs, and direct/manual alternatives.
