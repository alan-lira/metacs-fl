# Experiment Configurations

This folder contains the configuration sets used to run MetaCS-FL experiments.

The configurations are organized first by client-availability model and then by experiment family, client population, dataset, and data distribution.

All paths and commands below are relative to the repository root.

## Top-level structure

```txt
experiments/
├── static_client_availability/
│   ├── performance_experiments/
│   ├── dp_impact_experiments/
│   ├── scalability_experiments/
│   └── sensitivity_experiments/
└── dynamic_client_availability/
    ├── intermittent_client_availability_experiments/
    └── late_joining_clients_experiments/
```

## Static client availability

Static experiments assume that the client population available to the experiment does not change according to an intermittent-availability or late-entry schedule.

### Performance experiments

Typical layout:

```txt
experiments/static_client_availability/performance_experiments/
└── <num_clients>_clients/
    └── <dataset>/
        └── <distribution>/
            ├── <approach>.cfg
            ├── <approach>_server.cfg
            └── client.cfg
```

The repository contains configurations for client populations such as `50_clients` and `100_clients`, datasets such as `cifar_10`, `fashion_mnist`, and `emotion`, and `iid`/`non_iid` distributions where provided.

### Differential-privacy impact experiments

Located under:

```txt
experiments/static_client_availability/dp_impact_experiments/
```

These configurations are used with the DP impact analysis workflow.

### Scalability experiments

Located under:

```txt
experiments/static_client_availability/scalability_experiments/
```

These are server-only Python experiments. They are launched through `run_on_server_only.sh` directly or through `server_only_python` rows in a campaign manifest.

### Sensitivity experiments

Located under:

```txt
experiments/static_client_availability/sensitivity_experiments/
```

These are also handled as server-only experiments in the generated FGCS campaign pack.

## Dynamic client availability

Dynamic experiment configurations are kept separate from the static experiment families.

They are not currently included in the generated `fgcs_2026_experiments` pack. Run them with a custom manifest or with `run_one_distributed_experiment.sh`.

### Intermittent client availability

Layout:

```txt
experiments/dynamic_client_availability/intermittent_client_availability_experiments/
└── <num_clients>_clients/
    └── <dataset>/
        └── <distribution>/
            ├── client.cfg
            ├── <approach>.cfg
            └── <approach>_server.cfg
```

For example:

```txt
experiments/dynamic_client_availability/intermittent_client_availability_experiments/
└── 50_clients/
    └── cifar_10/
        └── iid/
            ├── client.cfg
            ├── fedavg.cfg
            ├── fedavg_server.cfg
            ├── oort.cfg
            ├── oort_server.cfg
            ├── rifles.cfg
            ├── rifles_server.cfg
            ├── metacsfl.cfg
            └── metacsfl_server.cfg
```

The approach executor configuration defines the available execution blocks and availability behavior. Use `--execution-blocks <block>` when only one named block should run. Omit the option to allow all blocks configured in the executor file to run.

### Late-joining clients

Layout:

```txt
experiments/dynamic_client_availability/late_joining_clients_experiments/
└── <num_clients>_clients/
    └── <dataset>/
        └── <distribution>/
            ├── client.cfg
            ├── <approach>_<num_late_clients>_late_clients.cfg
            └── <approach>_server.cfg
```

Examples of executor filenames include:

```txt
fedavg_10_late_clients.cfg
fedavg_25_late_clients.cfg
fedavg_50_late_clients.cfg
metacsfl_10_late_clients.cfg
metacsfl_25_late_clients.cfg
metacsfl_50_late_clients.cfg
```

The exact entry-round and client-performance variants are defined by the execution blocks inside the selected executor configuration. Select one with `--execution-blocks <block>` or omit the selector to run all configured blocks.

## Configuration roles

A distributed experiment normally uses three files:

| File | Role |
|---|---|
| `<approach>.cfg` or `<approach>_<variant>.cfg` | Flower executor configuration. Defines the experiment executions and references the server/client configs. |
| `<approach>_server.cfg` | Server and strategy configuration for the approach. |
| `client.cfg` | Shared client configuration for the dataset, distribution, and client population. |

The execution scripts stage these files into a per-run runtime directory. By default they also patch the runtime executor config to reference the selected server/client files and patch the gRPC addresses for the current node allocation.

## Run one distributed experiment

### Static example

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id static_cifar10_iid_fedavg_001 \
  --custom-flower-executor-cfg experiments/static_client_availability/performance_experiments/50_clients/cifar_10/iid/fedavg.cfg \
  --custom-flower-server-cfg experiments/static_client_availability/performance_experiments/50_clients/cifar_10/iid/fedavg_server.cfg \
  --custom-flower-client-cfg experiments/static_client_availability/performance_experiments/50_clients/cifar_10/iid/client.cfg
```

### Intermittent-availability example

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id intermittent_cifar10_iid_metacsfl_001 \
  --custom-flower-executor-cfg experiments/dynamic_client_availability/intermittent_client_availability_experiments/50_clients/cifar_10/iid/metacsfl.cfg \
  --custom-flower-server-cfg experiments/dynamic_client_availability/intermittent_client_availability_experiments/50_clients/cifar_10/iid/metacsfl_server.cfg \
  --custom-flower-client-cfg experiments/dynamic_client_availability/intermittent_client_availability_experiments/50_clients/cifar_10/iid/client.cfg \
  --launch-arg --execution-blocks \
  --launch-arg Execution_1_N
```

`run_one_distributed_experiment.sh` forwards additional launcher options as repeated `--launch-arg` pairs. Confirm the intended block name in the selected executor config.

### Late-joining example

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id latejoin_cifar10_iid_metacsfl_10_001 \
  --custom-flower-executor-cfg experiments/dynamic_client_availability/late_joining_clients_experiments/50_clients/cifar_10/iid/metacsfl_10_late_clients.cfg \
  --custom-flower-server-cfg experiments/dynamic_client_availability/late_joining_clients_experiments/50_clients/cifar_10/iid/metacsfl_server.cfg \
  --custom-flower-client-cfg experiments/dynamic_client_availability/late_joining_clients_experiments/50_clients/cifar_10/iid/client.cfg \
  --launch-arg --execution-blocks \
  --launch-arg Execution_1_N
```

## Run a custom campaign

Dynamic experiments should be listed in a custom CSV manifest with this exact header:

```csv
experiment_id,backend,group,num_clients,dataset,distribution,approach,executor_cfg,server_cfg,client_cfg,server_script,server_config,remote_output_dir
```

Example distributed row:

```csv
intermittent__50_clients__cifar_10__iid__metacsfl,distributed_flower,intermittent_client_availability_experiments,50_clients,cifar_10,iid,metacsfl,experiments/dynamic_client_availability/intermittent_client_availability_experiments/50_clients/cifar_10/iid/metacsfl.cfg,experiments/dynamic_client_availability/intermittent_client_availability_experiments/50_clients/cifar_10/iid/metacsfl_server.cfg,experiments/dynamic_client_availability/intermittent_client_availability_experiments/50_clients/cifar_10/iid/client.cfg,,,
```

Run the manifest with:

```bash
bash scripts/execution/run_many_distributed_experiments.sh \
  --manifest path/to/dynamic_campaign_manifest.csv \
  --campaign-id dynamic_campaign_001 \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate \
  --install false \
  --execution-blocks Execution_1_N \
  --max-parallel-experiments 1
```

The campaign-level `--execution-blocks` value is forwarded to every `distributed_flower` row and ignored for `server_only_python` rows. Use separate campaigns when different rows need different block selectors.

## Generated FGCS pack

The currently supported generated pack is:

```txt
fgcs_2026_experiments
```

It includes selected static performance, DP impact, scalability, and sensitivity experiments. It excludes:

```txt
dynamic_client_availability/intermittent_client_availability_experiments
dynamic_client_availability/late_joining_clients_experiments
performance experiments involving Emotion
```

See:

```txt
scripts/execution/generate_experiment_pack.README.md
scripts/execution/run_many_distributed_experiments.README.md
```

for the complete pack contents and campaign-runner behavior.

## Outputs and analysis

Distributed runs normally create:

```txt
distributed_launch_logs/<run_id>/
gathered_results/<run_id>/
merged_results/<run_id>/
```

Campaign runs prefix each experiment run ID with the campaign ID.

The performance analysis scripts support execution folders named as follows:

```txt
# Static/default
<approach>_exec_<id>/

# Intermittent availability
<approach>_<scenario>_exec_<id>/

# Late joining
<approach>_<num_late_clients>_late_clients_entry_round_<entry_round>_<performance_type>_performance_exec_<id>/
```

See:

```txt
scripts/analysis/README.md
scripts/analysis/plot_performance_results.README.md
scripts/analysis/summarize_performance_results.README.md
```

for required CSV files, trial discovery, plotting options, dropout metrics, and late-join engagement outputs.

## Before a large run

Validate the environment and orchestration stack first:

```bash
bash scripts/execution/run_toy_smoke_experiments.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --remote-venv-activate /root/metacs-fl/.venv/bin/activate
```
