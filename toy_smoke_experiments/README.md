# Toy Smoke Experiments

This folder contains a tiny smoke-test campaign for validating the MetaCS-FL execution orchestration stack without running the full experiment set.

It is intentionally placed outside `experiments/` so it is not accidentally included in paper campaigns.

## Contents

```txt
toy_smoke_experiments/
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

## What it tests

The default manifest runs two experiments:

1. `distributed_flower`: a tiny CIFAR-10 IID FedAvg execution with 2 clients and 1 round.
2. `server_only_python`: a self-contained server-only smoke script that writes a small CSV and JSON manifest.

The distributed test validates:

- `run_many_distributed_experiments.sh` manifest parsing and campaign markers;
- `run_one_distributed_experiment.sh` orchestration;
- `launch_distributed_flower.sh` runtime config staging;
- executor config patching through `base_server_config_file` and `base_client_config_file`;
- gRPC patching;
- remote execution;
- gather and merge.

The server-only test validates:

- server-node selection from the node file;
- SSH/local server-only execution;
- `--config-file` passing;
- output collection through `--remote-output-dir`.

## Important

For remote/Grid'5000 runs, this folder must exist under the remote project root, for example:

```txt
/root/metacs-fl/toy_smoke_experiments/
```

So either commit this folder to the repository or copy it to the remote nodes before running.

## Run both smoke tests

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest toy_smoke_experiments/campaign_manifest.csv \
  --campaign-id toy_smoke_check_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install false \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

## Run only the distributed smoke test

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest toy_smoke_experiments/campaign_manifest.distributed_only.csv \
  --campaign-id toy_smoke_distributed_check_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install false \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 8 \
  --repetitions 1
```

## Run only the server-only smoke test

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest toy_smoke_experiments/campaign_manifest.server_only_only.csv \
  --campaign-id toy_smoke_server_only_check_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install false \
  --max-parallel-experiments 1
```

## Local run

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest toy_smoke_experiments/campaign_manifest.csv \
  --campaign-id toy_smoke_local_check_001 \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --install false \
  --max-parallel-experiments 1 \
  --max-parallel-remote-ops 1 \
  --repetitions 1
```

## Expected outputs

For campaign id `toy_smoke_check_001`, expect:

```txt
campaign_runs/toy_smoke_check_001/
distributed_launch_logs/toy_smoke_check_001__toy_distributed__cifar_10__iid__fedavg_2clients_1round/
gathered_results/toy_smoke_check_001__toy_distributed__cifar_10__iid__fedavg_2clients_1round/
merged_results/toy_smoke_check_001__toy_distributed__cifar_10__iid__fedavg_2clients_1round/
server_only_logs/toy_smoke_check_001__toy_server_only__scalability__fedavg_10clients_10tasks_1round/
server_only_results/toy_smoke_check_001__toy_server_only__scalability__fedavg_10clients_10tasks_1round/
```

## Resume behavior

Rerun the same command with the same `--campaign-id`. Completed experiments should be skipped and failed experiments retried.
