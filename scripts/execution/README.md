# Execution Scripts

This folder contains the scripts used to run MetaCS-FL experiments.

All commands below assume they are launched from the project root.

## Files

| File | Purpose | README |
|---|---|---|
| `launch_distributed_flower.sh` | Low-level distributed Flower launcher. It stages configs, patches gRPC settings, launches remote processes, gathers outputs. | `launch_distributed_flower.README.md` |
| `run_one_distributed_experiment.sh` | One-experiment pipeline. Optionally runs setup, launches distributed Flower execution, then merges outputs. | `run_one_distributed_experiment.README.md` |
| `generate_experiment_pack.py` | Generates explicit campaign manifests such as `fgcs_2026_experiments`. | `generate_experiment_pack.README.md` |
| `run_many_distributed_experiments.sh` | Manifest-driven campaign runner with fault tolerance. | `run_many_distributed_experiments.README.md` |
| `run_on_server_only.sh` | Executes one server-only experiment on the server node from a node file. | `run_on_server_only.README.md` |

## Recommended usage levels

Use the highest-level script that matches your task:

1. Use `run_many_distributed_experiments.sh` for full campaigns.
2. Use `run_one_distributed_experiment.sh` for one distributed experiment.
3. Use `run_on_server_only.sh` for one server-only experiment.
4. Use `launch_distributed_flower.sh` only when you need direct control over distributed launch details.
5. Use `generate_experiment_pack.py` when you want to inspect or version a campaign manifest before execution.

## Common output roots

The scripts commonly create:

```txt
campaign_runs/<campaign_id>/
distributed_launch_logs/<run_id>/
gathered_results/<run_id>/
merged_results/<run_id>/
server_only_logs/<run_id>/
server_only_results/<run_id>/
```

## Common node files

The execution scripts use the shared node files from `scripts/`:

```txt
scripts/nodes.local.txt
scripts/nodes.remote.txt
scripts/nodes.g5k.txt
```

The node file format is:

```txt
mode <local|remote|g5k>

server <ssh_target|-> <runtime_host>
client <ssh_target|-> <runtime_host>
```

For Grid'5000, the SSH target and runtime host may differ. Example:

```txt
mode g5k

server root@paradoxe-16.rennes.g5k paradoxe-16.rennes.grid5000.fr
client root@paradoxe-17.rennes.g5k paradoxe-17.rennes.grid5000.fr
client root@paradoxe-18.rennes.g5k paradoxe-18.rennes.grid5000.fr
```

If your executor matches short hostnames, use short runtime hosts:

```txt
mode g5k

server root@paradoxe-16.rennes.g5k paradoxe-16
client root@paradoxe-17.rennes.g5k paradoxe-17
client root@paradoxe-18.rennes.g5k paradoxe-18
```
