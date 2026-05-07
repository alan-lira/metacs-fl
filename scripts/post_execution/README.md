# Post-Execution Scripts

This folder contains scripts used after distributed execution finishes.

## Files

| File | Purpose | README |
|---|---|---|
| `merge_distributed_outputs.py` | Merges per-node outputs gathered by the distributed launcher into a single consolidated output directory. | `merge_distributed_outputs.README.md` |

## Main command

From the project root:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py   gathered_results/<RUN_ID>   merged_results/<RUN_ID>   --clean-output true
```

The merger expects gathered directories such as:

```txt
gathered_results/<RUN_ID>/node_<node_name>/...
```

and writes:

```txt
merged_results/<RUN_ID>/
```

See `merge_distributed_outputs.README.md` for full details.
