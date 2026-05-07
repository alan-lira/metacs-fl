# Scalability Analysis

This document explains how to use:

```txt
scripts/analysis/scalability_analysis.py
```

to analyze MetaCS-FL scalability experiment results.

The script recursively loads scalability result CSV files, optionally aggregates LNS summary files, generates PDF plots for several overhead metrics, exports CSV and LaTeX summary tables, and prints readable summaries to the terminal.

All examples assume commands are executed from the project root folder.

---

## 1. Script location

```txt
scripts/analysis/scalability_analysis.py
```

---

## 2. Input folder

The script takes a root results folder:

```bash
--root-results-folder <root-results-folder>
```

If the folder name is exactly:

```txt
scalability_results
```

then the script uses it directly. Otherwise, it looks for:

```txt
<root-results-folder>/scalability_results
```

The script recursively searches for files matching:

```txt
*_scalability_results.csv
```

Example:

```txt
results/static_client_availability/scalability_results/
├── FedAvg/
│   └── fedavg_scalability_results.csv
├── MEC/
│   └── mec_scalability_results.csv
├── ECMTC/
│   └── ecmtc_scalability_results.csv
└── MetaCS-FL_015/
    ├── metacsfl_015_scalability_results.csv
    └── metacsfl_015/
        └── lns_summary/
            ├── round_1.csv
            └── round_2.csv
```

---

## 3. Required scalability CSV columns

Each `*_scalability_results.csv` file should contain at least:

```txt
approach_name
num_clients
num_tasks
num_rounds
total_client_selection_overhead
avg_overhead_per_round
avg_selection_cpu_time_seconds
avg_selection_rss_delta_mb
max_selection_rss_delta_mb
avg_selected_clients_per_round
```

Additional columns are allowed.

---

## 4. Optional LNS summary files

For each scalability result CSV, the script derives an experiment folder by removing the suffix:

```txt
_scalability_results.csv
```

Then it looks for:

```txt
<experiment-folder>/lns_summary/*.csv
```

If found, it aggregates optional LNS columns such as:

```txt
time_lapsed
cpu_time_seconds
rss_delta_mb
rss_peak_mb
num_iterations
iterations_per_second
num_accepted_moves
num_improving_moves
```

and appends averaged or total LNS metrics to the scalability dataframe.

---

## 5. Command-line usage

```bash
python3 scripts/analysis/scalability_analysis.py \
  --root-results-folder <root-results-folder> \
  --root-analysis-folder <analysis-output-root>
```

---

## 6. Parameters

| Parameter | Description | Required |
|---|---|---|
| `--root-results-folder` | Root results folder. If not named `scalability_results`, the script reads `<root-results-folder>/scalability_results`. | Yes |
| `--root-analysis-folder` | Root output folder for plots and tables. | Yes |

---

## 7. Output structure

The script creates:

```txt
<root-analysis-folder>/scalability_analysis/
├── fixed_clients/
│   ├── total_overhead/
│   ├── avg_cpu_overhead/
│   ├── avg_memory_delta/
│   ├── max_memory_delta/
│   ├── avg_lns_cpu_overhead/
│   ├── avg_lns_memory_delta/
│   └── avg_lns_iterations/
├── fixed_tasks/
│   ├── total_overhead/
│   ├── avg_cpu_overhead/
│   ├── avg_memory_delta/
│   ├── max_memory_delta/
│   ├── avg_lns_cpu_overhead/
│   ├── avg_lns_memory_delta/
│   └── avg_lns_iterations/
├── summary_tables/
└── representative_tables/
```

---

## 8. Generated plots

For each metric, the script generates two families of plots:

1. fixed number of clients, varying number of tasks;
2. fixed number of tasks, varying number of clients.

Metric prefixes include:

```txt
total_overhead
avg_cpu_overhead
avg_memory_delta
max_memory_delta
avg_lns_cpu_overhead
avg_lns_memory_delta
avg_lns_iterations
```

Example output files:

```txt
<root-analysis-folder>/scalability_analysis/fixed_clients/total_overhead/total_overhead_fixed_100_clients.pdf
<root-analysis-folder>/scalability_analysis/fixed_tasks/avg_cpu_overhead/avg_cpu_overhead_fixed_10000_tasks.pdf
```

If a metric column is not available, the script skips that metric and prints a message.

---

## 9. Generated tables

### 9.1 Full summary tables

For every `(num_clients, num_tasks)` pair, the script exports:

```txt
summary_tables/summary_c<num_clients>_t<num_tasks>.csv
summary_tables/summary_c<num_clients>_t<num_tasks>.tex
```

### 9.2 Representative overhead tables

For selected representative values:

```txt
num_clients = 10, 50, 100, 500, 1000
num_tasks   = 100, 1000, 10000, 30000, 40000
```

it exports:

```txt
representative_tables/fixed_<num_clients>_clients_overhead_summary.csv
representative_tables/fixed_<num_clients>_clients_overhead_summary.tex
representative_tables/fixed_<num_tasks>_tasks_overhead_summary.csv
representative_tables/fixed_<num_tasks>_tasks_overhead_summary.tex
```

### 9.3 Paper tables

For the representative large-scale case:

```txt
num_clients = 1000
num_tasks   = 40000
```

it exports:

```txt
representative_tables/paper_resource_overhead_table.csv
representative_tables/paper_resource_overhead_table.tex
representative_tables/paper_lns_overhead_table.csv
representative_tables/paper_lns_overhead_table.tex
```

The LNS table is generated only if the required LNS columns are available.

---

## 10. Approach order and supported labels

The script uses this approach order:

```txt
FedAvg
MEC
ECMTC
Oort
DivFL
ECSM
MetaCS-FL_015
MetaCS-FL_025
MetaCS-FL_050
```

The plotting style map supports normalized keys for those names. If you add new approaches, update both:

- `approach_order`;
- `color_marker_map`.

---

## 11. Example calls

### Direct scalability results folder

```bash
python3 scripts/analysis/scalability_analysis.py \
  --root-results-folder results/static_client_availability/scalability_results \
  --root-analysis-folder analysis_results/static_client_availability
```

### Parent folder containing `scalability_results/`

```bash
python3 scripts/analysis/scalability_analysis.py \
  --root-results-folder results/static_client_availability \
  --root-analysis-folder analysis_results/static_client_availability
```

---

## 12. Common issues

### 12.1 No scalability results found

If no files matching `*_scalability_results.csv` are found, the script raises an error.

Check with:

```bash
find <root-results-folder> -name '*_scalability_results.csv'
```

### 12.2 Missing metric column

The script skips plots for metrics whose columns are not present.

For example, LNS plots are skipped if LNS summary data is not available.

### 12.3 Approach does not appear in plots

Check that `approach_name` in the input CSV matches one of the names in `approach_order`, or update the script to include the new approach.

### 12.4 Empty paper tables

The paper tables use a fixed representative scenario:

```txt
num_clients = 1000
num_tasks = 40000
```

If your results do not include this pair, the corresponding table is skipped.
