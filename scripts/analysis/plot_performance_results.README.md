# Performance Accuracy Plotter

This document explains how to use:

```txt
scripts/analysis/plot_performance_results.py
```

to generate accuracy-over-round plots for MetaCS-FL performance experiments.

The script reads per-round individual metrics from multiple independent executions, aggregates weighted mean accuracy across trials, plots mean accuracy with standard deviation shading, and prints a small completion/failure/task summary per approach.

All examples assume commands are executed from the project root folder.

---

## 1. Script location

```txt
scripts/analysis/plot_performance_results.py
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
            │   └── output/
            │       ├── individual_fit_metrics_history.csv
            │       ├── individual_evaluate_metrics_history.csv
            │       ├── selected_fit_clients_history.csv
            │       └── selected_evaluate_clients_history.csv
            ├── <approach>_exec_2/
            └── ...
```

For example:

```txt
results/static_client_availability/performance_results/
└── 100_clients/
    └── cifar_10/
        └── iid/
            ├── fedavg_exec_1/output/individual_evaluate_metrics_history.csv
            ├── fedavg_exec_2/output/individual_evaluate_metrics_history.csv
            ├── metacsfl_exec_1/output/individual_evaluate_metrics_history.csv
            └── ...
```

---

## 3. Required CSV files

Depending on `--phase`, the script reads:

| Phase | Metrics file | Selection file |
|---|---|---|
| `train` | `individual_fit_metrics_history.csv` | `selected_fit_clients_history.csv` |
| `test` | `individual_evaluate_metrics_history.csv` | `selected_evaluate_clients_history.csv` |

The metrics file must contain columns matching these substrings:

```txt
comm_round
loss
accuracy
examples
client_id
```

The selection file must contain:

```txt
comm_round
num_selected_clients
num_tasks
```

For task completion summaries, the script uses:

```txt
ds_train_i  # train phase
ds_test_i   # test phase
```

---

## 4. Command-line usage

```bash
python3 scripts/analysis/plot_performance_results.py \
  --performance-results-folder <performance-results-folder> \
  --root-analysis-folder <analysis-output-root> \
  --experiment-tag <tag> \
  --dataset-name <dataset> \
  --dataset-distribution <iid|non_iid> \
  --num-available-clients <N> \
  --num-trials <T> \
  --phase <train|test> \
  --all-approaches <comma-separated-approaches>
```

---

## 5. Parameters

| Parameter | Description | Required |
|---|---|---|
| `--performance-results-folder` | Root performance results folder. | Yes |
| `--root-analysis-folder` | Root output folder for generated figures. | Yes |
| `--experiment-tag` | Output subfolder name, such as `performance` or `dp_impact`. | Yes |
| `--dataset-name` | Dataset name. Supported target-accuracy keys in the script: `cifar_10`, `fashion_mnist`, `emotion`. | Yes |
| `--dataset-distribution` | Dataset distribution. Supported values in the target map: `iid`, `non_iid`. | Yes |
| `--num-available-clients` | Number of available clients. The script reads `<N>_clients/`. | Yes |
| `--num-trials` | Number of independent executions to aggregate. | Yes |
| `--phase` | `train` or `test`. | Yes |
| `--all-approaches` | Comma-separated approach keys, such as `fedavg,mec,ecmtc,oort,divfl,ecsm,metacsfl`. | Yes |

---

## 6. Built-in target accuracies

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

These values are used to compute the first round reaching target accuracy.

---

## 7. Output files

The script creates:

```txt
<root-analysis-folder>/<experiment-tag>/
└── <phase>ing_accuracy_<dataset_name>_<dataset_distribution>_<num_available_clients>_clients.pdf
```

Examples:

```txt
analysis_results/static_client_availability/performance/testing_accuracy_cifar_10_iid_100_clients.pdf
analysis_results/static_client_availability/performance/training_accuracy_fashion_mnist_non_iid_50_clients.pdf
```

The plot includes:

- mean weighted accuracy across trials;
- standard deviation shading;
- final accuracy annotations;
- one curve per approach.

The script also prints a summary table with:

- selected clients;
- failed clients;
- allocated tasks;
- completed tasks;
- percentage of failed clients;
- percentage of completed tasks.

---

## 8. Approach keys

The script has built-in style entries for:

```txt
fedavg
mec
ecmtc
oort
divfl
ecsm
metacsfl
metacsfl_no_privacy
```

If `--all-approaches` contains a key not listed above, the script will fail when it tries to look up color/marker settings.

---

## 9. Example calls

### CIFAR-10 IID, 100 clients, testing phase

```bash
python3 scripts/analysis/plot_performance_results.py \
  --performance-results-folder results/static_client_availability/performance_results \
  --root-analysis-folder analysis_results/static_client_availability \
  --experiment-tag performance \
  --dataset-name cifar_10 \
  --dataset-distribution iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,mec,ecmtc,oort,divfl,ecsm,metacsfl
```

### Fashion-MNIST non-IID, 50 clients, training phase

```bash
python3 scripts/analysis/plot_performance_results.py \
  --performance-results-folder results/static_client_availability/performance_results \
  --root-analysis-folder analysis_results/static_client_availability \
  --experiment-tag performance \
  --dataset-name fashion_mnist \
  --dataset-distribution non_iid \
  --num-available-clients 50 \
  --num-trials 3 \
  --phase train \
  --all-approaches fedavg,mec,ecmtc,oort,divfl,ecsm,metacsfl
```

---

## 10. Common issues

### 10.1 Missing trial warnings

The script prints warnings when an expected trial folder or metrics file is missing. It skips missing trials and aggregates the available ones.

### 10.2 No plot generated

If no approach produces usable data, the script prints:

```txt
[ERROR] No experiment produced usable data. Skipping plot.
```

Check the folder layout and whether `--num-trials`, `--dataset-name`, `--dataset-distribution`, and `--num-available-clients` match your results.

### 10.3 Unsupported dataset name

The script uses a fixed target-accuracy dictionary. If you pass a dataset not present in that dictionary, the script will fail with a key error. Add the dataset and target accuracy to the script before running.

### 10.4 Unsupported approach key

If you add a new approach to `--all-approaches`, also add it to the script's style map.
