# Scripts

This folder contains setup, execution, and post-processing utilities for MetaCS-FL experiments.

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

- `ssh_target` is used by setup/execution scripts to reach a node.
- `runtime_host` is used by Flower/gRPC during distributed execution.
- In `local` mode, use `-` as the SSH target.

## Subfolders

- `setup/`: install MetaCS-FL on local, remote, or Grid'5000 nodes.
- `execution/`: launch one or many experiments, run server-only experiments, and generate campaign manifests.
- `post_execution/`: merge gathered per-node outputs.
- `analysis/`: analysis and plotting scripts. This folder is not covered by this README set.

## Main workflow

A typical Grid'5000 workflow is:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh   --nodes-file scripts/nodes.g5k.txt   --remote-project-dir /root/metacs-fl   --max-parallel-installs 8
```

Then run either one experiment:

```bash
bash scripts/execution/run_one_distributed_experiment.sh   --install false   --nodes-file scripts/nodes.g5k.txt   --remote-project-dir /root/metacs-fl   --run-id g5k_test_001   --custom-flower-executor-cfg metacs_fl/flower_executor/config/flower_executor.cfg   --custom-flower-server-cfg metacs_fl/server/config/flower_server.cfg   --custom-flower-client-cfg metacs_fl/client/config/flower_client.cfg
```

or a full campaign:

```bash
bash scripts/execution/run_many_distributed_experiments.sh   --pack fgcs_2026_experiments   --campaign-id fgcs_2026_experiments_001   --nodes-file scripts/nodes.g5k.txt   --remote-project-dir /root/metacs-fl   --install false   --max-parallel-experiments 1
```

## README map

- `setup/setup_remote_metacsfl_node.README.md`: setup/install script.
- `execution/launch_distributed_flower.README.md`: low-level distributed Flower launcher.
- `execution/run_one_distributed_experiment.README.md`: one-experiment pipeline: launch + merge, optionally setup.
- `execution/generate_experiment_pack.README.md`: manifest generator for explicit experiment packs.
- `execution/run_many_distributed_experiments.README.md`: resumable campaign runner.
- `execution/run_on_server_only.README.md`: server-only backend used by campaign runs.
- `post_execution/merge_distributed_outputs.README.md`: output merger.
