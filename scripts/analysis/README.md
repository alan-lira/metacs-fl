# Analysis Scripts

This folder contains post-processing scripts for MetaCS-FL experiment results.

These scripts are intended to be run after experiment execution and output gathering/merging. They do not launch experiments. They read result CSV files and produce figures, tables, or terminal summaries for the paper analysis.

All examples below assume commands are executed from the project root folder.

## Scripts

| Script | Purpose | Main output |
|---|---|---|
| `plot_dp_impact_distributions.py` | Compare non-private and differentially private per-client class distributions. | PDF plots under `<root-analysis-folder>/dp_impact_analysis/` plus terminal summaries. |
| `plot_performance_results.py` | Plot training/testing accuracy curves across approaches and trials. | PDF accuracy plot under `<root-analysis-folder>/<experiment-tag>/`. |
| `scalability_analysis.py` | Analyze scalability overhead results and generate plots/tables. | PDF plots, CSV tables, and LaTeX tables under `<root-analysis-folder>/scalability_analysis/`. |
| `summarize_performance_results.py` | Print numerical performance summaries and percentage changes vs a baseline. | Terminal output only. |

## Common assumptions

The performance-oriented scripts assume the standard MetaCS-FL performance result layout:

```txt
<performance-results-folder>/
└── <num_available_clients>_clients/
    └── <dataset_name>/
        └── <dataset_distribution>/
            ├── fedavg_exec_1/
            │   ├── clients_resources.csv
            │   └── output/
            │       ├── individual_fit_metrics_history.csv
            │       ├── individual_evaluate_metrics_history.csv
            │       └── selected_fit_clients_history.csv
            ├── fedavg_exec_2/
            ├── mec_exec_1/
            └── ...
```

The scalability script assumes a scalability results folder containing one or more files matching:

```txt
*_scalability_results.csv
```

It searches recursively below the selected scalability results directory.

## Dependencies

The scripts use standard Python packages already expected in the MetaCS-FL environment:

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

Use a dedicated generated folder for analysis outputs, for example:

```txt
analysis_results/
```

This folder should usually be ignored by Git.

Example `.gitignore` entry:

```gitignore
analysis_results/
```

## Typical workflow

1. Run experiments.
2. Gather and merge distributed outputs, if applicable.
3. Run the relevant analysis script.
4. Inspect generated PDF figures, CSV tables, LaTeX tables, and terminal summaries.

Example:

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

## Script-specific documentation

See the script-specific README files:

```txt
plot_dp_impact_distributions.README.md
plot_performance_results.README.md
scalability_analysis.README.md
summarize_performance_results.README.md
```
