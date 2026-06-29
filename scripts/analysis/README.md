# Analysis Scripts

This folder contains post-processing scripts for MetaCS-FL experiment results.

The scripts are intended to run after experiment execution and, for distributed runs, after output gathering and merging. They do not launch experiments. They read result CSV files and produce figures, numerical summaries, CSV exports, LaTeX tables, and terminal reports.

All examples assume commands are executed from the project root.

## Scripts

| Script | Purpose | Main output |
|---|---|---|
| `plot_dp_impact_distributions.py` | Compare non-private and differentially private per-client class distributions. | PDF plots under `<root-analysis-folder>/dp_impact_analysis/` plus terminal summaries. |
| `plot_performance_results.py` | Plot weighted training/testing accuracy curves across approaches and trials for static, intermittent-availability, and late-joining experiments. | Separate or combined PDF figures under `<root-analysis-folder>/<experiment-tag>/`, plus completion/failure summaries. |
| `scalability_analysis.py` | Analyze scalability overhead results and generate plots/tables. | PDF plots, CSV tables, and LaTeX tables under `<root-analysis-folder>/scalability_analysis/`. |
| `summarize_performance_results.py` | Calculate numerical performance summaries and baseline-relative changes for static and dynamic experiments. | Terminal tables; optional LaTeX files; intermittent-availability dropout tables; late-join engagement CSV/PDF exports. |

## Performance result layouts

The two performance-oriented scripts share the following parent structure:

```txt
<performance-results-folder>/
└── <num_available_clients>_clients/
    └── <dataset_name>/
        └── <dataset_distribution>/
            └── <execution-folder>/
                ├── clients_resources.csv
                └── output/
                    ├── individual_fit_metrics_history.csv
                    ├── individual_evaluate_metrics_history.csv
                    ├── selected_fit_clients_history.csv
                    └── selected_evaluate_clients_history.csv
```

Not every script requires every file. See the corresponding script-specific README for the exact files and columns.

### Static/default execution folders

```txt
<approach>_exec_<id>/
```

Example:

```txt
fedavg_exec_1/
metacsfl_exec_5000/
```

### Intermittent-availability execution folders

```txt
<approach>_<scenario>_exec_<id>/
```

Examples:

```txt
fedavg_moderate_exec_5000/
metacsfl_severe_exec_5001/
```

Scenarios are passed through `--availability-scenarios`, for example:

```bash
--availability-scenarios moderate,severe
```

### Late-joining execution folders

```txt
<approach>_<num_late_clients>_late_clients_entry_round_<entry_round>_<performance_type>_performance_exec_<id>/
```

Example:

```txt
metacsfl_25_late_clients_entry_round_50_best_performance_exec_5000/
```

A late-joining configuration is represented by:

```txt
(num_late_clients, entry_round, performance_type)
```

Late-join engagement analysis additionally uses `late_joining_clients_ids.csv` when available.

### Trial discovery

The performance scripts first check the traditional sequence:

```txt
exec_1, exec_2, ..., exec_<num_trials>
```

If none of those folders exists for a configuration, they discover numeric execution IDs and select the first `--num-trials` folders in numeric order. This supports campaign-generated identifiers such as `exec_5000`, `exec_5001`, and `exec_5002`.

Late-joining options and `--availability-scenarios` must not be used in the same invocation.

## Scalability result layout

The scalability script searches recursively below the selected scalability results directory for files matching:

```txt
*_scalability_results.csv
```

## Dependencies

The analysis scripts use packages already expected in the MetaCS-FL environment, primarily:

```txt
numpy
pandas
matplotlib
```

Activate the project virtual environment before running analysis commands:

```bash
source .venv/bin/activate
```

## Recommended output location

Use a dedicated generated folder, for example:

```txt
analysis_results/
```

This folder should normally be ignored by Git:

```gitignore
analysis_results/
```

## Typical workflow

1. Run the desired experiments.
2. Gather and merge distributed outputs, when applicable.
3. Arrange or copy the merged execution folders under the result layout expected by the selected analysis script.
4. Run the plotting or summarization command.
5. Inspect the generated PDF figures, CSV files, LaTeX tables, and terminal summaries.

## Examples

### Static performance plot

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

### Combined intermittent-availability plot

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

### Late-joining summary with engagement exports

```bash
python3 scripts/analysis/summarize_performance_results.py \
  --performance-results-folder results/dynamic_client_availability/late_joining_clients_results \
  --dataset-name fashion_mnist \
  --dataset-distribution iid \
  --num-available-clients 100 \
  --num-trials 3 \
  --phase test \
  --all-approaches fedavg,oort,rifles,metacsfl \
  --baseline-approach fedavg \
  --desired-latejoin-tuples '10:10:worst;25:25:best' \
  --latejoin-engagement-latex-output-file analysis_results/tables/latejoin_engagement.tex \
  --latejoin-engagement-output-folder analysis_results/latejoin_engagement
```

## Script-specific documentation

See:

```txt
plot_dp_impact_distributions.README.md
plot_performance_results.README.md
scalability_analysis.README.md
summarize_performance_results.README.md
```
