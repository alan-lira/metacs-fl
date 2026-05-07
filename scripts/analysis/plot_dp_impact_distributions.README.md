# DP Impact Distribution Plotter

This document explains how to use:

```txt
scripts/analysis/plot_dp_impact_distributions.py
```

to compare non-private and differentially private per-client class distributions.

The script reads one folder of non-private client distribution CSV files and one folder of differentially private client distribution CSV files. It then produces aggregate and per-client comparison plots, prints class-level summaries, computes L1 divergences, and reports the clients with the maximum and minimum DP impact.

All examples assume commands are executed from the project root folder.

---

## 1. Script location

```txt
scripts/analysis/plot_dp_impact_distributions.py
```

---

## 2. Input folders

The script expects two folders:

```txt
<non-private-data-distribution-folder>/
<differentially-private-data-distribution-folder>/
```

Each folder should contain files named:

```txt
client_<id>.csv
```

Example:

```txt
data_distributions/no_privacy/
├── client_0.csv
├── client_1.csv
└── client_2.csv

data_distributions/differentially_private/
├── client_0.csv
├── client_1.csv
└── client_2.csv
```

The client IDs should match between the two folders.

---

## 3. Expected CSV columns

### 3.1 Non-private files

Each non-private `client_<id>.csv` file must contain:

```csv
split,class,count
```

Example:

```csv
split,class,count
train,0,125
train,1,90
test,0,20
test,1,18
```

The script currently analyzes only:

```txt
split = train
```

### 3.2 Differentially private files

Each differentially private `client_<id>.csv` file must contain:

```csv
split,class,noisy_count
```

Example:

```csv
split,class,noisy_count
train,0,127
train,1,87
test,0,19
test,1,20
```

The `noisy_count` values are converted to integers using:

```python
int(float(noisy_count))
```

---

## 4. Command-line usage

```bash
python3 scripts/analysis/plot_dp_impact_distributions.py \
  --non-private-data-distribution-folder <non-private-folder> \
  --differentially-private-data-distribution-folder <dp-folder> \
  --root-analysis-folder <analysis-output-root>
```

---

## 5. Parameters

| Parameter | Description | Required |
|---|---|---|
| `--non-private-data-distribution-folder` | Folder containing non-private `client_<id>.csv` distribution files. | Yes |
| `--differentially-private-data-distribution-folder` | Folder containing DP `client_<id>.csv` distribution files. | Yes |
| `--root-analysis-folder` | Root folder where analysis outputs are written. | Yes |

---

## 6. Output files

The script creates:

```txt
<root-analysis-folder>/dp_impact_analysis/
├── overall_class_distribution.pdf
├── max_divergence_client_comparison.pdf
└── min_divergence_client_comparison.pdf
```

Where:

| File | Description |
|---|---|
| `overall_class_distribution.pdf` | Aggregate class distribution comparison across all clients. |
| `max_divergence_client_comparison.pdf` | Per-class comparison for the client with the maximum L1 divergence. |
| `min_divergence_client_comparison.pdf` | Per-class comparison for the client with the minimum L1 divergence. |

The script also prints terminal summaries for:

- aggregate class distribution;
- maximum and minimum L1 divergence clients;
- mean L1 divergence across clients;
- per-client DP impact ratio statistics.

---

## 7. Example call

```bash
python3 scripts/analysis/plot_dp_impact_distributions.py \
  --non-private-data-distribution-folder results/static_client_availability/dp_impact_results/no_privacy/data_distribution \
  --differentially-private-data-distribution-folder results/static_client_availability/dp_impact_results/differentially_private/data_distribution \
  --root-analysis-folder analysis_results/static_client_availability
```

---

## 8. What the script computes

For each client and class, the script compares:

```txt
non_private_count
noisy_count
```

It computes the per-client L1 divergence:

```txt
sum over classes |non_private_count - noisy_count|
```

It then reports:

- mean L1 divergence across clients;
- client with maximum L1 divergence;
- client with minimum L1 divergence;
- per-client DP impact ratio, defined as L1 divergence divided by the non-private total samples for that client.

---

## 9. Common issues

### 9.1 `KeyError: 'split'`, `KeyError: 'class'`, `KeyError: 'count'`, or `KeyError: 'noisy_count'`

One or more input CSV files are missing required columns.

Check headers with:

```bash
head -n 1 <folder>/client_0.csv
```

### 9.2 Empty or missing plots

Make sure the CSV files contain rows with:

```txt
split = train
```

The script ignores other splits.

### 9.3 Mismatched clients

The script assumes the DP and non-private folders contain the same client IDs. If a client appears in the non-private folder but not the DP folder, the divergence computation can fail.

### 9.4 Unexpected number of classes

The number of classes is detected from the maximum class index in the non-private train split. Classes are assumed to be zero-indexed.
