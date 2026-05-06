# Performance Result Summarizer

This document explains how to use:

```txt
scripts/analysis/summarize_performance_results.py
```

to print numerical summaries for MetaCS-FL performance experiments and compare approaches against a baseline.

Unlike `plot_performance_results.py`, this script does not generate plots or files. It prints metrics and percentage changes to the terminal.

All examples assume commands are executed from the project root folder.

---

## 1. Script location

```txt
scripts/analysis/summarize_performance_results.py
```

---

## 2. Expected result layout

The script expects this structure:

```txt
<performance-results-folder>/
└── <num_available_clients>_clients/
    └── <dataset_name>/
        └── <dataset_distribution>/
            ├── <approach>_exec_1/
            │   ├── clients_resources.csv
            │   └── output/
            │       ├── individual_fit_metrics_history.csv
            │       ├── individual_evaluate_metrics_history.csv
            │       └── selected_fit_clients_history.csv
            ├── <approach>_exec_2/
            └── ...
```

For example:

```txt
results/static_client_availability/performance_results/
└── 100_clients/
    └── cifar_10/
        └── iid/
            ├── fedavg_exec_1/
            ├── mec_exec_1/
            ├── ecmtc_exec_1/
            └── metacsfl_exec_1/
```

---

## 3. Required files per trial

Each trial folder must contain:

```txt
output/individual_fit_metrics_history.csv
output/individual_evaluate_metrics_history.csv
output/selected_fit_clients_history.csv
clients_resources.csv
```

If any of these files is missing for a trial, that trial is skipped.

---

## 4. Required columns

### 4.1 `individual_fit_metrics_history.csv`

The script searches columns by substring, so exact names may vary, but it expects columns containing:

```txt
comm_round
client_id
loss
accuracy
examples
training_time
training_energy
```

### 4.2 `individual_evaluate_metrics_history.csv`

The script expects columns containing:

```txt
comm_round
loss
accuracy
examples
```

### 4.3 `selected_fit_clients_history.csv`

The script expects:

```txt
comm_round
selection_duration
num_selected_clients
available_clients
selected_clients
```

The `available_clients` and `selected_clients` columns are expected to use pipe-separated client names such as:

```txt
client_0|client_5|client_12
```

### 4.4 `clients_resources.csv`

The script expects:

```txt
client_id
mean_power_consumption_idle_in_watts
```

---

## 5. Command-line usage

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder <performance-results-folder> \
  --dataset-name <dataset> \
  --dataset-distribution <iid|non_iid> \
  --num-available-clients <N> \
  --num-trials <T> \
  --phase <train|test> \
  --all-approaches <comma-separated-approaches> \
  --baseline <baseline-approach>
```

---

## 6. Parameters

| Parameter | Description | Required |
|---|---|---|
| `--performance-results-folder` | Root performance results folder. | Yes |
| `--dataset-name` | Dataset name. Supported target-accuracy keys in the script: `cifar_10`, `fashion_mnist`, `emotion`. | Yes |
| `--dataset-distribution` | Dataset distribution. Supported values in the target map: `iid`, `non_iid`. | Yes |
| `--num-available-clients` | Number of available clients. The script reads `<N>_clients/`. | Yes |
| `--num-trials` | Number of independent executions. | Yes |
| `--phase` | Phase used to determine the target round. Usually `test`. | Yes |
| `--all-approaches` | Comma-separated approach keys. | Yes |
| `--baseline` | Baseline approach key used for percentage-change comparison. | Yes |

---

## 7. Built-in target accuracies

The script uses the following target accuracy map:

```txt
cifar_10:
  iid:     0.75
  non_iid: 0.45

fashion_mnist:
  iid:     0.85
  non_iid: 0.70

emotion:
  iid:     0.90
  non_iid: 0.80
```

For each trial, the script determines the first FL round at which the weighted mean evaluation accuracy reaches the target accuracy. Metrics are then summarized up to that target round.

---

## 8. Metrics printed

For each approach, the script prints mean and standard deviation across trials for metrics such as:

```txt
fl_round_@_target_accuracy
target_accuracy
max_training_time_across_all_rounds_in_seconds
total_training_time_in_seconds
total_training_energy_in_joules
min_number_selected_clients_training
max_number_selected_clients_training
mean_number_selected_clients_training
jain_fairness_index_training
mean_examples_per_selected_client_training
std_examples_per_selected_client_training
cv_tasks_per_client_training
total_selection_duration_in_seconds
total_idle_energy_during_selection_in_joules
total_idle_energy_during_training_in_joules
total_training+idle_energy_during_training_in_joules
```

It then prints percentage changes relative to the baseline approach.

---

## 9. Example calls

### Compare all approaches against FedAvg

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder results/static_client_availability/performance_results \
  --dataset-name cifar_10 \
  --dataset-distribution iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,mec,ecmtc,oort,divfl,ecsm,metacsfl \
  --baseline fedavg
```

### Compare MetaCS-FL variants against `metacsfl_no_privacy`

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder results/static_client_availability/dp_impact_results \
  --dataset-name cifar_10 \
  --dataset-distribution non_iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches metacsfl_no_privacy,metacsfl \
  --baseline metacsfl_no_privacy
```

---

## 10. Common issues

### 10.1 Missing files for a trial

The script prints warnings and skips trials if any required file is missing.

### 10.2 Division by zero or empty metrics

If all trials for an approach are skipped, the script can fail when computing means. Check that all expected trial folders and CSV files exist.

### 10.3 Unsupported dataset name

The script uses a fixed target-accuracy dictionary. If you pass a dataset not present in that dictionary, the script will fail. Add the dataset and target accuracy before running.

### 10.4 Baseline not included in `--all-approaches`

The baseline must be one of the approaches listed in `--all-approaches`; otherwise, the baseline comparison will fail.

### 10.5 Client list parsing fails

The script expects `available_clients` and `selected_clients` values like:

```txt
client_0|client_1|client_2
```

If your files use a different format, update the parser before running.
