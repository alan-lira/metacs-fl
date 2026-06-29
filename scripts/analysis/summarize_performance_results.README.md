# Performance Result Summarizer

This document explains how to use:

```txt
scripts/analysis/summarize_performance_results.py
```

to calculate numerical summaries for MetaCS-FL performance experiments and compare approaches against a baseline.

The script supports static/default, late-joining-client, and intermittent-availability result layouts. It prints detailed per-metric summaries and baseline-relative percentage changes. By default, it also prints:

- a compact terminal table;
- a copy-ready LaTeX table;
- an extra dropout-focused table for intermittent-availability runs;
- an extra engagement table for late-joining runs.

Optional arguments can save the LaTeX tables, export late-join engagement metrics to CSV, and generate late-join composition plots.

All examples assume commands are executed from the project root folder.

---

## 1. Script location

```txt
scripts/analysis/summarize_performance_results.py
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
                ├── clients_resources.csv
                └── output/
                    ├── individual_fit_metrics_history.csv
                    ├── individual_evaluate_metrics_history.csv
                    └── selected_fit_clients_history.csv
```

### 2.1 Static/default experiments

```txt
<approach>_exec_<id>/
```

### 2.2 Late-joining-client experiments

```txt
<approach>_<num_late_clients>_late_clients_entry_round_<entry_round>_<performance_type>_performance_exec_<id>/
```

Example:

```txt
metacsfl_25_late_clients_entry_round_50_best_performance_exec_5000/
```

Late-join engagement analysis also looks for:

```txt
late_joining_clients_ids.csv
```

The file may be located in the trial folder, its `output/` folder, the trial folder's parent, or another descendant of the trial folder. It must contain `client_id`. If `round_of_first_appearance` is present, it is used as each late client's entry round; otherwise, entry round `1` is assumed.

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

The script first checks:

```txt
exec_1, exec_2, ..., exec_<num-trials>
```

If none of those exists for an approach/configuration, it discovers numeric execution IDs and uses the first `--num-trials` folders in numeric order. This supports campaign IDs such as `exec_5000`, `exec_5001`, and `exec_5002`.

A trial is skipped if any of its four required files is missing.

---

## 3. Required files per trial

Each usable trial requires:

```txt
output/individual_fit_metrics_history.csv
output/individual_evaluate_metrics_history.csv
output/selected_fit_clients_history.csv
clients_resources.csv
```

For late-join engagement metrics, `late_joining_clients_ids.csv` is additionally required. Missing late-join metadata produces zero engagement metrics for that trial and a warning; it does not invalidate the base performance summary.

---

## 4. Required columns

### 4.1 `individual_fit_metrics_history.csv`

Required direct columns:

```txt
comm_round
client_id
```

Required columns found by substring:

```txt
training_time
training_energy
examples
```

For late-join sample accounting, the script prefers `ds_train_i` and falls back to the first column containing `examples`.

### 4.2 `individual_evaluate_metrics_history.csv`

The script expects:

```txt
comm_round
```

and columns whose names contain:

```txt
loss
accuracy
examples
```

The weighted mean evaluation accuracy determines the target round.

### 4.3 `selected_fit_clients_history.csv`

The script expects:

```txt
comm_round
selection_duration
num_selected_clients
num_tasks
available_clients
selected_clients
```

`available_clients` and `selected_clients` must be pipe-separated client IDs, for example:

```txt
client_0|client_5|client_12
```

The script accepts numeric client IDs and `client_<id>` values when normalizing late-join metadata.

### 4.4 `clients_resources.csv`

The script expects:

```txt
client_id
mean_power_consumption_idle_in_watts
```

### 4.5 `late_joining_clients_ids.csv`

For late-join engagement analysis:

```txt
client_id                    # required
round_of_first_appearance    # optional
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
  --baseline <baseline-approach> \
  [analysis-group options] \
  [table/output options]
```

---

## 6. Parameters

### 6.1 Required parameters

| Parameter | Description |
|---|---|
| `--performance-results-folder` | Root performance-results folder. The script appends `<N>_clients/`. |
| `--dataset-name` | Dataset key. Built-in target-accuracy keys are `cifar_10`, `fashion_mnist`, and `emotion`. |
| `--dataset-distribution` | Distribution key. Built-in values are `iid` and `non_iid`. |
| `--num-available-clients` | Number used to select `<N>_clients/`. |
| `--num-trials` | Maximum number of executions summarized for each approach/configuration. |
| `--phase` | Phase label used when naming weighted accuracy fields. The target round is calculated from the evaluation metrics file; `test` is the normal value. |
| `--all-approaches` | Comma-separated approach keys. |
| `--baseline` | Approach used for percentage-change comparisons. It should be included in `--all-approaches`. |

### 6.2 Late-joining options

Use either an explicit tuple list or all three Cartesian-product options.

| Parameter | Description |
|---|---|
| `--desired-latejoin-tuples` | Semicolon-separated tuples in `num_late_clients:entry_round:performance_type` form, for example `10:10:worst;25:50:best`. `/`, `|`, or `,` may also separate fields inside a tuple. |
| `--latejoin-num-clients` | Comma-separated late-client counts. Must be used with both options below. |
| `--latejoin-entry-rounds` | Comma-separated entry rounds. Must be used with the other two Cartesian-product options. |
| `--latejoin-performance-types` | Comma-separated performance types, for example `worst,best`. Must be used with the other two Cartesian-product options. |
| `--latejoin-samples-mode` | `completed` (default) uses completed late-client samples from `ds_train_i`/`examples`; `proportional_scheduled` estimates late-client scheduled samples from total round workload and the late-client share of selected clients. |

### 6.3 Intermittent-availability options

| Parameter | Description |
|---|---|
| `--availability-scenarios` | Comma-separated scenarios, for example `moderate,severe`. The script reads `<approach>_<scenario>_exec_<id>`. Values such as `moderate_availability` are normalized to `moderate`. |
| `--samples-per-task` | Overrides the number of samples represented by one scheduled task when calculating failed scheduled samples. Defaults: `cifar_10=1`, `fashion_mnist=1`, `emotion=10`; unknown datasets default to `1`. |

Do not combine `--availability-scenarios` with late-joining options.

### 6.4 Main terminal and LaTeX table options

| Parameter | Description |
|---|---|
| `--no-terminal-table` | Disable the compact final terminal table. Detailed metrics and percentage changes are still printed. |
| `--no-latex-table` | Disable the copy-ready LaTeX tables. This also prevents the optional LaTeX output files from being written. |
| `--latex-output-file` | Save the main generated LaTeX table to a `.tex` file. |
| `--latex-resize-width` | Width passed to `\\resizebox`. Default: `0.8\\columnwidth`. |
| `--latex-caption` | Custom caption for the main LaTeX table. |
| `--latex-label` | Custom label for the main LaTeX table. |
| `--no-mark-baseline-in-latex` | Remove the `(Baseline)` marker from generated terminal and LaTeX tables. |
| `--energy-metric` | Energy shown as `Sigma_total`: `training` (default) or `training_idle`, which includes idle energy during training. |

### 6.5 Dropout-focused output options

These outputs are generated only when `--availability-scenarios` is used.

| Parameter | Description |
|---|---|
| `--no-dropout-table` | Disable the extra dropout-focused terminal and LaTeX tables. |
| `--dropout-latex-output-file` | Save the dropout-focused LaTeX table to a `.tex` file. Requires LaTeX output to remain enabled. |

### 6.6 Late-join engagement output options

These outputs are generated only for late-joining groups.

| Parameter | Description |
|---|---|
| `--no-latejoin-engagement-table` | Disable the extra late-join engagement terminal and LaTeX tables. |
| `--latejoin-engagement-latex-output-file` | Save the late-join engagement LaTeX table to a `.tex` file. Requires LaTeX output to remain enabled. |
| `--latejoin-engagement-output-folder` | Save the engagement CSV and two stacked-composition PDF plots to this folder. This export is independent of `--no-latejoin-engagement-table`. |

---

## 7. Analysis groups

### 7.1 Static/default mode

When no dynamic options are provided, the script summarizes folders named:

```txt
<approach>_exec_<id>
```

### 7.2 Late-joining mode

Each tuple is summarized as a separate group. The main terminal/LaTeX table includes a `Tuple` column, and the engagement table measures post-entry workload redistribution between initial and late clients.

The engagement window begins when late clients become available and ends at the approach's target-accuracy round.

### 7.3 Intermittent-availability mode

Each scenario is summarized as a separate group. The main table adds failure columns, and the dropout-focused table provides normalized failure-impact metrics.

---

## 8. Built-in target accuracies

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

For each usable trial, the script finds the first communication round at which weighted mean evaluation accuracy reaches the target. If the target is not reached, the final available round is used. Most time, energy, fairness, failure, and engagement metrics are summarized only through that round.

---

## 9. Metrics and generated tables

### 9.1 Detailed metrics

For every approach/group, the script prints mean and standard deviation across trials for metrics including:

```txt
fl_round_@_target_accuracy
target_accuracy
max_training_time_across_all_rounds_in_seconds
total_training_time_in_seconds
total_training_energy_in_joules
min_number_selected_clients_training
max_number_selected_clients_training
mean_number_selected_clients_training
mean_scheduled_samples_training
mean_number_failing_clients_training
mean_number_failing_samples_training
failing_clients_per_selected_client_training_pct
failing_samples_per_selected_client_training
failing_samples_per_failing_client_training
failing_samples_share_scheduled_training_pct
jain_fairness_index_training
mean_examples_per_selected_client_training
std_examples_per_selected_client_training
cv_tasks_per_client_training
total_selection_duration_in_seconds
total_idle_energy_during_selection_in_joules
total_idle_energy_during_training_in_joules
total_training+idle_energy_during_training_in_joules
```

It then prints percentage changes relative to the baseline. A zero baseline value is reported as `N/A`.

### 9.2 Main compact table

The main terminal and LaTeX tables contain:

| Column | Meaning |
|---|---|
| `M_total` | Total training makespan through the target round, in seconds. |
| `Sigma_total` | Selected energy metric, in kilojoules. |
| `T_cs` | Total client-selection duration, in seconds. |
| `S_fair` | Jain fairness index for selection rates. |
| `n_fail` | Mean selected clients that did not complete training per round; intermittent mode only. |
| `s_fail` | Mean scheduled training samples not completed per round; intermittent mode only. |
| `RoA@<target>` | Mean ± standard deviation of the round of target accuracy. |

Non-baseline values include percentage changes versus the selected baseline.

### 9.3 Dropout-focused table

For intermittent-availability groups, the extra table contains:

```txt
mean selected clients
n_fail
n_fail/selected (%)
s_fail
s_fail/n_fail
s_fail/scheduled (%)
```

Here, `s_fail` is calculated as:

```txt
scheduled samples - completed samples
```

where scheduled samples are `num_tasks * samples_per_task`.

### 9.4 Late-join engagement table

For late-joining groups, the extra engagement output compares initial and late clients after entry and through the target round. It reports mean selected clients, mean samples, within-approach shares, and late-versus-initial ratios.

The table emphasizes composition shares. The CSV also includes the corresponding absolute values and ratios.

---

## 10. Output files

Without output-file options, all detailed summaries and generated tables are printed to standard output.

### 10.1 Optional LaTeX files

```txt
--latex-output-file <main-table.tex>
--dropout-latex-output-file <dropout-table.tex>
--latejoin-engagement-latex-output-file <engagement-table.tex>
```

Parent directories are created automatically.

### 10.2 Late-join engagement folder

When `--latejoin-engagement-output-folder <folder>` is supplied, the script writes:

```txt
<folder>/latejoin_engagement_summary.csv
<folder>/latejoin_selected_clients_composition_<dataset_name>_<dataset_distribution>.pdf
<folder>/latejoin_samples_composition_<dataset_name>_<dataset_distribution>.pdf
```

The PDF files are stacked composition plots showing initial-client and late-client contributions after the entry round.

---

## 11. Approach labels and styles

Built-in labels/styles are defined for:

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

Unknown approach keys receive fallback plot styles where plots are generated and use an uppercased key as their display label.

---

## 12. Example calls

### 12.1 Static comparison with a saved LaTeX table

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder results/static_client_availability/performance_results \
  --dataset-name cifar_10 \
  --dataset-distribution iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,mec,ecmtc,oort,divfl,ecsm,fedcab,rifles,metacsfl \
  --baseline fedavg \
  --latex-output-file analysis_results/tables/cifar10_iid_performance.tex
```

### 12.2 Intermittent availability with dropout outputs

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder results/dynamic_client_availability/intermittent_availability_results \
  --dataset-name cifar_10 \
  --dataset-distribution non_iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,oort,rifles,metacsfl \
  --baseline fedavg \
  --availability-scenarios moderate,severe \
  --samples-per-task 1 \
  --latex-output-file analysis_results/tables/intermittent_main.tex \
  --dropout-latex-output-file analysis_results/tables/intermittent_dropout.tex
```

### 12.3 Late-joining tuples with tables, CSV, and plots

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder results/dynamic_client_availability/late_joining_clients_results \
  --dataset-name fashion_mnist \
  --dataset-distribution iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,oort,rifles,metacsfl \
  --baseline fedavg \
  --desired-latejoin-tuples '10:10:worst;25:25:best' \
  --latejoin-samples-mode completed \
  --latex-output-file analysis_results/tables/latejoin_main.tex \
  --latejoin-engagement-latex-output-file analysis_results/tables/latejoin_engagement.tex \
  --latejoin-engagement-output-folder analysis_results/latejoin_engagement
```

### 12.4 Late-joining Cartesian product

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder results/dynamic_client_availability/late_joining_clients_results \
  --dataset-name emotion \
  --dataset-distribution non_iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,metacsfl \
  --baseline fedavg \
  --latejoin-num-clients 10,25 \
  --latejoin-entry-rounds 10,25 \
  --latejoin-performance-types worst,best \
  --latejoin-samples-mode proportional_scheduled
```

---

## 13. Common issues

### 13.1 No usable trials

The script warns and skips an approach/group when all matching trials are missing required files or cannot be found. Verify the result root, folder naming, dataset, distribution, scenario/tuple, and numeric execution IDs.

### 13.2 Incomplete late-join Cartesian options

If any of these options is used, all three are required:

```txt
--latejoin-num-clients
--latejoin-entry-rounds
--latejoin-performance-types
```

### 13.3 Mixing late-joining and availability scenarios

Use either late-joining options or `--availability-scenarios`, not both.

### 13.4 Missing baseline data

If the baseline has no usable metrics for a group, the script prints a warning and skips percentage changes for that group. Include the baseline in `--all-approaches` and verify its files.

### 13.5 Incorrect client-list format

`available_clients` and `selected_clients` should be pipe-separated, for example:

```txt
client_0|client_1|client_2
```

### 13.6 Incorrect failed-sample scale

Set `--samples-per-task` to match the experiment's task definition. The built-in defaults are `1` for CIFAR-10/Fashion-MNIST and `10` for Emotion.

### 13.7 Missing late-join metadata

Late-join engagement metrics require `late_joining_clients_ids.csv`. Confirm the file contains `client_id`; add `round_of_first_appearance` when clients have different entry rounds.

### 13.8 LaTeX file not written

Do not combine a `*-latex-output-file` option with `--no-latex-table`. The output directory is created automatically when LaTeX generation is enabled.

### 13.9 Unsupported dataset or distribution

The target-accuracy map is fixed. Add a target for a new dataset/distribution before using it.
