# Distributed Output Merger

This document explains how to use:

```txt
scripts/post_execution/merge_distributed_outputs.py
```

to safely merge distributed MetaCS-FL output folders gathered from multiple nodes.

This script is intended to run **after** `scripts/execution/launch_distributed_flower.sh`, which gathers per-node output folders under:

```txt
gathered_results/<RUN_ID>/
```

The merge script reads those per-node folders and produces a single consolidated output directory, usually under:

```txt
merged_results/<RUN_ID>/
```

The script is designed to preserve data safely. CSV files with matching relative paths are appended row-wise when their headers match, while conflicting files are copied into a conflict directory instead of being overwritten. 

---

## 1. Script location

The script is located at:

```txt
scripts/post_execution/merge_distributed_outputs.py
```

Example call from the project root:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --clean-output true
```

---

## 2. Expected input structure

The script expects a gathered results directory containing one folder per node.

Expected structure:

```txt
gathered_results/<RUN_ID>/
├── node_<node_a>/
│   └── output_execution/
│       └── exec_a_1/
│           └── ...
├── node_<node_b>/
│   └── output_execution/
│       └── exec_a_1/
│           └── ...
├── node_<node_c>/
│   └── output_execution/
│       └── exec_b_1/
│           └── ...
├── distributed_run_manifest.txt
└── nodes_runtime_<RUN_ID>.txt
```

By default, the script looks for node folders whose names start with:

```txt
node_
```

This default can be changed with:

```bash
--node-prefix <prefix>
```

---

## 3. What the script does

The script:

1. finds all node directories under the gathered root;
2. scans all files inside each `node_*` directory;
3. groups files by their relative path after the node directory;
4. merges CSV files with the same relative path when their headers match;
5. sorts merged CSV rows when round-like columns exist;
6. copies non-CSV files safely;
7. stores conflicting files under `_merge_conflicts/`;
8. copies top-level gathered metadata into `_gather_metadata/`;
9. writes a detailed `merge_report.json`.

For example, these files:

```txt
gathered_results/<RUN_ID>/node_paradoxe-16/output_execution/exec_a_1/output/individual_fit_metrics_history.csv
gathered_results/<RUN_ID>/node_paradoxe-17/output_execution/exec_a_1/output/individual_fit_metrics_history.csv
gathered_results/<RUN_ID>/node_paradoxe-18/output_execution/exec_a_1/output/individual_fit_metrics_history.csv
```

become:

```txt
merged_results/<RUN_ID>/output_execution/exec_a_1/output/individual_fit_metrics_history.csv
```

---

## 4. Command-line usage

Basic usage:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  <gathered_root> \
  <output_root>
```

Full form:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  <gathered_root> \
  <output_root> \
  --node-prefix node_ \
  --add-source-node false \
  --deduplicate-rows false \
  --clean-output false
```

The script takes two positional arguments:

| Argument | Description |
|---|---|
| `gathered_root` | Path to `gathered_results/<RUN_ID>` containing `node_*` folders. |
| `output_root` | Path where the merged output should be written. |

It also supports optional arguments such as `--node-prefix`, `--add-source-node`, `--deduplicate-rows`, and `--clean-output`. :contentReference[oaicite:1]{index=1}

---

## 5. Parameters

| Parameter | Description | Default |
|---|---|---|
| `gathered_root` | Input gathered directory containing per-node folders. | Required |
| `output_root` | Output directory where merged files are written. | Required |
| `--node-prefix PREFIX` | Prefix used to identify node folders. | `node_` |
| `--add-source-node true\|false` | Add a `source_node` column to merged CSV rows. | `false` |
| `--deduplicate-rows true\|false` | Remove exact duplicate CSV rows after merging. | `false` |
| `--clean-output true\|false` | Delete `output_root` before writing merged results. | `false` |

Boolean values can be passed as:

```txt
true false
1 0
yes no
y n
on off
```

---

## 6. Output structure

A typical merged output directory looks like:

```txt
merged_results/<RUN_ID>/
├── output_execution/
│   └── exec_a_1/
│       └── ...
├── _gather_metadata/
│   ├── distributed_run_manifest.txt
│   └── nodes_runtime_<RUN_ID>.txt
├── _merge_conflicts/
│   └── ...
└── merge_report.json
```

The main merged files are written using the same relative paths found under each node directory.

Top-level files from the gathered root, such as manifests and runtime hostfiles, are copied into:

```txt
_gather_metadata/
```

Conflicting files are copied into:

```txt
_merge_results/<RUN_ID>/_merge_conflicts/
```

The detailed merge report is written to:

```txt
merge_report.json
```

The script writes a JSON report containing the gathered root, output root, discovered node folders, merge options, copied metadata, per-file actions, and a summary of merge actions. 

---

## 7. CSV merge behavior

CSV files are merged when they have the same relative path and matching headers.

For example:

```txt
node_a/output_execution/exec_a_1/output/metrics.csv
node_b/output_execution/exec_a_1/output/metrics.csv
```

become:

```txt
merged_results/<RUN_ID>/output_execution/exec_a_1/output/metrics.csv
```

Rows are appended from all matching source files.

If a CSV file contains one of the following columns:

```txt
comm_round
round
server_round
```

the merged rows are sorted by that column.

The script can also use secondary columns when present:

```txt
client_id
client
phase
timestamp
time
```

This helps keep merged round-level and client-level metrics ordered after aggregation. :contentReference[oaicite:3]{index=3}

---

## 8. Adding the source node to CSV rows

Use:

```bash
--add-source-node true
```

to add a column named:

```txt
source_node
```

to merged CSV files.

Example:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --add-source-node true \
  --clean-output true
```

This is useful when you want to know which node produced each row.

Example output columns:

```csv
comm_round,client_id,accuracy,loss,source_node
1,0,0.45,1.23,node_root_paradoxe-16.rennes.g5k
1,1,0.43,1.31,node_root_paradoxe-17.rennes.g5k
```

---

## 9. Deduplicating rows

Use:

```bash
--deduplicate-rows true
```

to remove exact duplicate rows after merging.

Example:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --deduplicate-rows true \
  --clean-output true
```

When `--add-source-node true` is also used, `source_node` participates in deduplication. That means two otherwise identical rows from different nodes are treated as different rows.

When `--add-source-node false`, deduplication uses only the original CSV columns.

---

## 10. Cleaning the output directory

By default, the script does not delete an existing output directory.

Use:

```bash
--clean-output true
```

to remove the output directory before writing merged results.

Example:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --clean-output true
```

This is recommended when rerunning a merge for the same run id.

Be careful: this deletes the existing output directory before writing the new merged results.

---

## 11. Conflict handling

The script avoids overwriting conflicting files.

Conflicts are stored under:

```txt
_merge_conflicts/
```

### 11.1 CSV header mismatch

If CSV files with the same relative path have different headers, the script does not merge them.

Instead, it copies all conflicting source files into:

```txt
_merge_conflicts/csv_header_mismatch/
```

Example:

```txt
merged_results/<RUN_ID>/_merge_conflicts/csv_header_mismatch/output_execution/exec_a_1/output/metrics.csv.from_node_a
merged_results/<RUN_ID>/_merge_conflicts/csv_header_mismatch/output_execution/exec_a_1/output/metrics.csv.from_node_b
```

The report includes the reference header and the mismatched headers.

### 11.2 Non-CSV filename conflict

If non-CSV files with the same relative path are identical, the script copies one copy.

If they differ, the script copies them into:

```txt
_merge_conflicts/non_csv_name_conflict/
```

This prevents accidental overwriting.

### 11.3 Merge exceptions

If an exception occurs while reading or merging a group of files, the files are copied into:

```txt
_merge_conflicts/merge_exception/
```

The error is recorded in `merge_report.json`.

---

## 12. Merge report

Every merge writes:

```txt
merge_report.json
```

The report contains:

```json
{
  "gathered_root": "...",
  "output_root": "...",
  "node_prefix": "node_",
  "nodes": ["node_a", "node_b"],
  "add_source_node": false,
  "deduplicate_rows": false,
  "clean_output": true,
  "top_level_metadata_copied": [],
  "files": [],
  "summary": {}
}
```

Each file entry records what happened to that relative path.

Typical actions include:

```txt
merged_csv
copied_unique_file
copied_identical_duplicate_file_once
copied_empty_or_headerless_csv
conflict_csv_header_mismatch
conflict_non_csv_name_conflict
conflict_merge_exception
```

The terminal output also prints a summary and the path to the report. :contentReference[oaicite:4]{index=4}

---

## 13. Example calls

### 13.1 Basic merge

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001
```

---

### 13.2 Recommended merge for most experiments

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --clean-output true
```

---

### 13.3 Merge and keep source node information

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --add-source-node true \
  --clean-output true
```

---

### 13.4 Merge and deduplicate rows

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --deduplicate-rows true \
  --clean-output true
```

---

### 13.5 Merge with source node column and deduplication

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --add-source-node true \
  --deduplicate-rows true \
  --clean-output true
```

---

### 13.6 Merge output from a custom launcher gather root

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  custom_gathered_results/my_run \
  custom_merged_results/my_run \
  --node-prefix node_ \
  --clean-output true
```

---

## 14. Use after distributed launch

A typical workflow is:

1. Launch distributed execution:

```bash
bash scripts/execution/launch_distributed_flower.sh \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id cifar10_iid_test_001
```

2. Merge gathered outputs:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/cifar10_iid_test_001 \
  merged_results/cifar10_iid_test_001 \
  --clean-output true
```

The launch script gathers per-node results under `gathered_results/<RUN_ID>/node_<node_name>/`; the merge script then consolidates those node folders into one merged output. :contentReference[oaicite:5]{index=5}

---

## 15. Use inside the one-experiment pipeline

If using the one-experiment pipeline script, the merge step can be run automatically after launch.

Example:

```bash
bash scripts/execution/run_one_distributed_experiment.sh \
  --install false \
  --nodes-file scripts/nodes.g5k.txt \
  --remote-project-dir /root/metacs-fl \
  --run-id cifar10_iid_test_001 \
  --custom-flower-executor-cfg experiments/my_run/flower_executor.cfg \
  --custom-flower-server-cfg experiments/my_run/flower_server.cfg \
  --custom-flower-client-cfg experiments/my_run/flower_client.cfg \
  --clean-merge-output true
```

The pipeline will use:

```txt
gathered_results/<RUN_ID>/
merged_results/<RUN_ID>/
```

---

## 16. Common issues

### 16.1 No node folders found

Error example:

```txt
RuntimeError: No node folders found in gathered_results/<RUN_ID> with prefix 'node_'
```

Check that the gathered directory exists and contains folders such as:

```txt
node_local/
node_root_paradoxe-16.rennes.g5k/
node_root_paradoxe-17.rennes.g5k/
```

If your folders use another prefix, pass:

```bash
--node-prefix <prefix>
```

---

### 16.2 Gathered root does not exist

Error example:

```txt
FileNotFoundError: Gathered root does not exist or is not a directory
```

Check that the distributed launcher completed result collection and that the run id is correct.

For example:

```bash
ls gathered_results/
ls gathered_results/<RUN_ID>/
```

---

### 16.3 CSV files were not merged

Check `merge_report.json`.

If the action is:

```txt
conflict_csv_header_mismatch
```

then the CSV headers differ. The script copies the conflicting files into:

```txt
_merge_conflicts/csv_header_mismatch/
```

You can inspect the copied files and decide whether to normalize their headers manually.

---

### 16.4 Non-CSV files appear in `_merge_conflicts`

If non-CSV files have the same relative path but different contents, the script stores all versions in:

```txt
_merge_conflicts/non_csv_name_conflict/
```

This is expected and prevents data loss.

---

### 16.5 Rows are not sorted as expected

The script sorts only when one of these columns exists:

```txt
comm_round
round
server_round
```

If none exists, rows remain in the discovered merge order.

Secondary sort columns are used when present:

```txt
client_id
client
phase
timestamp
time
```

---

### 16.6 Duplicate rows remain

Use:

```bash
--deduplicate-rows true
```

If `--add-source-node true` is also enabled, otherwise identical rows from different nodes remain distinct because the `source_node` value is different.

---

### 16.7 Existing merged files remain after rerun

Use:

```bash
--clean-output true
```

This deletes the output directory before writing the new merged output.

---

## 17. Recommended command

For most experiments, use:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/<RUN_ID> \
  merged_results/<RUN_ID> \
  --clean-output true
```

When debugging which node produced which rows, use:

```bash
python3 scripts/post_execution/merge_distributed_outputs.py \
  gathered_results/<RUN_ID> \
  merged_results/<RUN_ID> \
  --add-source-node true \
  --clean-output true
```
