# Performance Accuracy Plotter

This document explains how to use:

```txt
scripts/analysis/plot_performance_results.py
```

to generate accuracy-over-round plots for MetaCS-FL performance experiments.

The script reads per-round individual metrics from multiple executions, aggregates weighted mean accuracy across trials, plots mean accuracy with standard-deviation shading, and prints a completion/failure/task summary per approach.

It supports three experiment-folder layouts:

- static/default experiments;
- late-joining-client experiments;
- intermittent-availability experiments.

It can generate one figure per configuration or a combined side-by-side figure for multiple late-joining tuples or availability scenarios.

All examples assume commands are executed from the project root folder.

---

## 1. Script location

```txt
scripts/analysis/plot_performance_results.py
```

---

## 2. Expected result layouts

The common parent structure is:

```txt
<performance-results-folder>/
└── <num_available_clients>_clients/
    └── <dataset_name>/
        └── <dataset_distribution>/
            └── <experiment-folder>/
                └── output/
                    ├── individual_fit_metrics_history.csv
                    ├── individual_evaluate_metrics_history.csv
                    ├── selected_fit_clients_history.csv
                    └── selected_evaluate_clients_history.csv
```

### 2.1 Static/default experiments

```txt
<approach>_exec_<id>/
```

Example:

```txt
results/static_client_availability/performance_results/
└── 100_clients/
    └── cifar_10/
        └── iid/
            ├── fedavg_exec_1/
            ├── fedavg_exec_2/
            ├── metacsfl_exec_1/
            └── ...
```

### 2.2 Late-joining-client experiments

```txt
<approach>_<num_late_clients>_late_clients_entry_round_<entry_round>_<performance_type>_performance_exec_<id>/
```

Example:

```txt
metacsfl_10_late_clients_entry_round_25_worst_performance_exec_5000/
```

A late-joining tuple has this logical form:

```txt
(num_late_clients, entry_round, performance_type)
```

### 2.3 Intermittent-availability experiments

```txt
<approach>_<scenario>_exec_<id>/
```

Examples:

```txt
fedavg_moderate_exec_5000/
metacsfl_severe_exec_5001/
```

Late-joining options and `--availability-scenarios` are mutually exclusive in one invocation.

### 2.4 Trial discovery

The script first checks the traditional sequence:

```txt
exec_1, exec_2, ..., exec_<num-trials>
```

If none of those folders exists for an approach/configuration, it discovers numeric execution IDs and uses the first `--num-trials` folders in numeric order. This supports campaign-generated IDs such as:

```txt
exec_5000, exec_5001, exec_5002
```

Missing or unreadable trials are skipped with warnings.

---

## 3. Required CSV files and columns

Depending on `--phase`, the script reads:

| Phase | Metrics file | Selection file |
|---|---|---|
| `train` | `individual_fit_metrics_history.csv` | `selected_fit_clients_history.csv` |
| `test` | `individual_evaluate_metrics_history.csv` | `selected_evaluate_clients_history.csv` |

The metrics file must contain:

```txt
comm_round
client_id
```

and columns whose names contain:

```txt
loss
accuracy
examples
```

The selection file must contain:

```txt
comm_round
num_selected_clients
num_tasks
```

For the completion summary, the metrics file must also contain:

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
  --all-approaches <comma-separated-approaches> \
  [late-joining options] \
  [intermittent-availability options] \
  [combined-figure options]
```

---

## 5. Parameters

### 5.1 Required parameters

| Parameter | Description |
|---|---|
| `--performance-results-folder` | Root performance-results folder. The script appends `<N>_clients/`. |
| `--root-analysis-folder` | Root output folder for generated figures. |
| `--experiment-tag` | Output subfolder, such as `performance`, `late_join_clients`, or `intermittent_availability`. |
| `--dataset-name` | Dataset key. Built-in target-accuracy keys are `cifar_10`, `fashion_mnist`, and `emotion`. |
| `--dataset-distribution` | Distribution key. Built-in values are `iid` and `non_iid`. |
| `--num-available-clients` | Number used to select `<N>_clients/`. |
| `--num-trials` | Maximum number of executions to aggregate for each approach/configuration. |
| `--phase` | `train` or `test`. |
| `--all-approaches` | Comma-separated approach keys. |

### 5.2 Late-joining options

Use either an explicit tuple list or the three Cartesian-product options.

| Parameter | Description |
|---|---|
| `--desired-latejoin-tuples` | Semicolon-separated tuples in `num_late_clients:entry_round:performance_type` form, for example `10:10:worst;25:50:best`. `/`, `|`, or `,` may also separate fields inside a tuple. |
| `--latejoin-num-clients` | Comma-separated late-client counts. Must be used with both options below. |
| `--latejoin-entry-rounds` | Comma-separated entry rounds. Must be used with the other two Cartesian-product options. |
| `--latejoin-performance-types` | Comma-separated performance types, for example `worst,best`. Must be used with the other two Cartesian-product options. |

When the three Cartesian-product options are used, every combination is analyzed. For example, two client counts, two entry rounds, and two performance types produce eight tuples.

### 5.3 Intermittent-availability options

| Parameter | Description |
|---|---|
| `--availability-scenarios` | Comma-separated folder suffixes, for example `moderate,severe`. The script reads `<approach>_<scenario>_exec_<id>`. |

Do not combine `--availability-scenarios` with late-joining tuple options.

### 5.4 Combined-figure options

| Parameter | Description |
|---|---|
| `--combine-availability-scenarios` | Create one side-by-side figure containing the listed availability scenarios and a shared legend. |
| `--combine-latejoin-tuples` | Create one side-by-side figure containing the selected late-joining tuples and a shared legend. Requires late-joining options. |
| `--also-save-separate-scenarios` | After a combined figure is created, also save each scenario/tuple as an individual figure. Without this flag, the script returns after saving the combined figure. |

`--combine-availability-scenarios` is only valid for intermittent-availability experiments. `--combine-latejoin-tuples` is only valid for late-joining experiments.

---

## 6. Built-in target accuracies

The script uses:

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

The target is used when calculating the first round that reaches the configured weighted mean accuracy. The accuracy curves themselves are plotted for all available rounds.

---

## 7. Output files

All figures are saved under:

```txt
<root-analysis-folder>/<experiment-tag>/
```

### 7.1 Static/default figure

```txt
<phase>ing_accuracy_<dataset_name>_<dataset_distribution>_<num_available_clients>_clients.pdf
```

Example:

```txt
testing_accuracy_cifar_10_iid_100_clients.pdf
```

### 7.2 Separate intermittent-availability figure

```txt
<phase>ing_accuracy_<dataset_name>_<dataset_distribution>_<num_available_clients>_clients_<scenario>.pdf
```

Example:

```txt
testing_accuracy_cifar_10_iid_100_clients_moderate.pdf
```

### 7.3 Separate late-joining figure

```txt
<phase>ing_accuracy_<dataset_name>_<dataset_distribution>_<num_available_clients>_clients_<num_late_clients>_late_entry_<entry_round>_<performance_type>.pdf
```

Example:

```txt
testing_accuracy_cifar_10_iid_100_clients_10_late_entry_25_worst.pdf
```

### 7.4 Combined intermittent-availability figure

```txt
<phase>ing_accuracy_<dataset_name>_<dataset_distribution>_<num_available_clients>_clients_<scenario_1>_<scenario_2>_combined.pdf
```

### 7.5 Combined late-joining figure

```txt
<phase>ing_accuracy_<dataset_name>_<dataset_distribution>_<num_available_clients>_clients_<tuple-labels>_combined_latejoin.pdf
```

Each plot includes:

- mean weighted accuracy across usable trials;
- standard-deviation shading;
- final-accuracy annotations;
- one curve per approach;
- automatically adjusted axes.

The script also prints a per-approach summary containing:

- selected clients;
- failed clients;
- allocated tasks;
- completed tasks;
- percentage of failed clients;
- percentage of completed tasks.

---

## 8. Approach styles

Built-in styles are defined for:

```txt
fedavg
mec
ecmtc
oort
divfl
ecsm
fedcab
rifles
rifles_gh
metacsfl
metacsfl_no_privacy
```

Unknown approach keys no longer cause a style lookup failure. The script assigns them a fallback color/marker combination and uses the uppercased key as the label.

---

## 9. Example calls

### 9.1 Static CIFAR-10 IID results

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
  --all-approaches fedavg,mec,ecmtc,oort,divfl,ecsm,fedcab,rifles,metacsfl
```

### 9.2 Combined intermittent-availability scenarios

```bash
python3 scripts/analysis/plot_performance_results.py \
  --performance-results-folder results/dynamic_client_availability/intermittent_availability_results \
  --root-analysis-folder analysis_results/dynamic_client_availability \
  --experiment-tag intermittent_availability \
  --dataset-name cifar_10 \
  --dataset-distribution non_iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,oort,rifles,metacsfl \
  --availability-scenarios moderate,severe \
  --combine-availability-scenarios \
  --also-save-separate-scenarios
```

### 9.3 Combined explicit late-joining tuples

```bash
python3 scripts/analysis/plot_performance_results.py \
  --performance-results-folder results/dynamic_client_availability/late_joining_clients_results \
  --root-analysis-folder analysis_results/dynamic_client_availability \
  --experiment-tag late_join_clients \
  --dataset-name fashion_mnist \
  --dataset-distribution iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,oort,rifles,metacsfl \
  --desired-latejoin-tuples '10:10:worst;25:25:best' \
  --combine-latejoin-tuples
```

### 9.4 Late-joining Cartesian product

```bash
python3 scripts/analysis/plot_performance_results.py \
  --performance-results-folder results/dynamic_client_availability/late_joining_clients_results \
  --root-analysis-folder analysis_results/dynamic_client_availability \
  --experiment-tag late_join_clients \
  --dataset-name emotion \
  --dataset-distribution non_iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,metacsfl \
  --latejoin-num-clients 10,25 \
  --latejoin-entry-rounds 10,25 \
  --latejoin-performance-types worst,best
```

---

## 10. Common issues

### 10.1 No trial folders found

The script prints:

```txt
[WARNING] No trial folders found for template: ...
```

Check the selected results root, dataset, distribution, scenario/tuple, and approach names. Also verify that execution-folder names contain a numeric value after `_exec_`.

### 10.2 Missing metrics or selection files

A missing metrics file is skipped while loading curves. A missing or unreadable selection/metrics pair is skipped in the completion summary.

### 10.3 No plot generated

If no approach produces usable curve data, the script prints an error and skips the figure. Check the folder layout, `--num-trials`, `--dataset-name`, `--dataset-distribution`, `--num-available-clients`, and scenario/tuple options.

### 10.4 Incomplete Cartesian-product options

When any of these is supplied:

```txt
--latejoin-num-clients
--latejoin-entry-rounds
--latejoin-performance-types
```

all three must be supplied.

### 10.5 Mixing dynamic experiment types

Do not combine late-joining options with `--availability-scenarios` in the same command.

### 10.6 Unsupported dataset or distribution

The target-accuracy map is fixed. Add a target for a new dataset/distribution before using it.
