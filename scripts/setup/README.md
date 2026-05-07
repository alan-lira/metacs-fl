# Setup Scripts

This folder contains scripts for preparing machines before MetaCS-FL experiments.

## Files

| File | Purpose | README |
|---|---|---|
| `setup_remote_metacsfl_node.sh` | Installs MetaCS-FL on local, remote, or Grid'5000 nodes using a shared node file. | `setup_remote_metacsfl_node.README.md` |

## Main command

From the project root:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh   --nodes-file scripts/nodes.g5k.txt   --remote-project-dir /root/metacs-fl   --max-parallel-installs 8
```

For a private GitHub repository:

```bash
bash scripts/setup/setup_remote_metacsfl_node.sh   --nodes-file scripts/nodes.g5k.txt   --remote-project-dir /root/metacs-fl   --repo-auth token   --prompt-github-token true   --branch main   --max-parallel-installs 8
```

See `setup_remote_metacsfl_node.README.md` for the full documentation.
