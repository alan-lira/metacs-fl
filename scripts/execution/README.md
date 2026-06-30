# Execution Scripts

This folder contains the scripts used to validate and run MetaCS-FL experiments.

All commands below assume they are launched from the project root.

## Files

| File | Purpose | README |
|---|---|---|
| `run_toy_smoke_experiments.sh` | Recommended preflight wrapper. Validates and optionally synchronizes the toy campaign, then invokes the campaign runner. | `run_toy_smoke_experiments.README.md` |
| `launch_distributed_flower.sh` | Low-level distributed Flower launcher. It stages configs, patches executor/gRPC settings, launches processes, and gathers outputs. | `launch_distributed_flower.README.md` |
| `run_one_distributed_experiment.sh` | One-experiment pipeline. It can run setup, launch distributed Flower, and merge gathered outputs. | `run_one_distributed_experiment.README.md` |
| `generate_experiment_pack.py` | Generates explicit campaign manifests such as `fgcs_2026_experiments`. | `generate_experiment_pack.README.md` |
| `run_many_distributed_experiments.sh` | Manifest-driven campaign runner with completion markers, retries, and resume behavior. | `run_many_distributed_experiments.README.md` |
| `run_on_server_only.sh` | Executes one server-only experiment on the server node selected from a node file. | `run_on_server_only.README.md` |

## Recommended usage levels

Use the highest-level script that matches the task:

1. Use `run_toy_smoke_experiments.sh` before a new allocation, environment, branch, or large campaign.
2. Use `run_many_distributed_experiments.sh` for a manifest-driven campaign.
3. Use `run_one_distributed_experiment.sh` for one distributed experiment.
4. Use `run_on_server_only.sh` for one server-only experiment.
5. Use `launch_distributed_flower.sh` only when direct control over launch details is required.
6. Use `generate_experiment_pack.py` to inspect or version a generated campaign manifest before execution.

## Execution-block selection

Both `run_many_distributed_experiments.sh` and `launch_distributed_flower.sh` support:

```bash
--execution-blocks <block>
```

Examples:

```bash
--execution-blocks Execution_1_N
--execution-blocks Execution_2_N
```

The campaign runner forwards this value through `run_one_distributed_experiment.sh` to the distributed launcher and then to `main.py`. When omitted, all execution blocks configured in the selected executor file may run.

This option applies to distributed Flower rows. Server-only rows do not use it.

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

The `--output-dir` option available in the launcher/campaign workflow can place these generated roots below a common local base directory. Explicit root arguments override the derived defaults.

## Common node files

The execution scripts use the shared node files from `scripts/`:

```txt
scripts/nodes.local.txt
scripts/nodes.remote.txt
scripts/nodes.g5k.txt
```

The node-file format is:

```txt
mode <local|remote|g5k>

server <ssh_target|-> <runtime_host>
client <ssh_target|-> <runtime_host>
```

For Grid'5000, the SSH target and runtime host may differ:

```txt
mode g5k

server root@paradoxe-16.rennes.g5k paradoxe-16.rennes.grid5000.fr
client root@paradoxe-17.rennes.g5k paradoxe-17.rennes.grid5000.fr
client root@paradoxe-18.rennes.g5k paradoxe-18.rennes.grid5000.fr
```

If the executor configuration matches short hostnames, use short runtime hosts:

```txt
mode g5k

server root@paradoxe-16.rennes.g5k paradoxe-16
client root@paradoxe-17.rennes.g5k paradoxe-17
client root@paradoxe-18.rennes.g5k paradoxe-18
```

## Recommended first command

For Grid'5000 or generic remote nodes:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate
```

For a local smoke test:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.local.txt \
  --remote-project-dir "$PWD" \
  --remote-venv-activate "$PWD/.venv/bin/activate" \
  --campaign-id toy_smoke_local_001
```

## Related configuration guides

- `../../experiments/README.md`: static and dynamic experiment layouts, one-run examples, and custom manifest examples.
- `../../toy_smoke_experiments/README.md`: smoke-test files, expected outputs, and direct campaign-runner alternative.
