from argparse import ArgumentParser
from csv import DictReader
from numpy import arange, zeros
from os import listdir, path
from pathlib import Path
import matplotlib.pyplot as plt


def detect_num_classes(non_private_data_distribution_folder: Path,
                       split="train"):
    max_class = -1
    for filename in listdir(non_private_data_distribution_folder):
        if not filename.startswith("client_") or not filename.endswith(".csv"):
            continue
        with open(path.join(non_private_data_distribution_folder, filename), encoding="utf-8-sig") as f:
            reader = DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                cls = int(row["class"])
                if cls > max_class:
                    max_class = cls
    return max_class + 1  # classes are 0-indexed


def parse_non_private_csv_files(non_private_data_distribution_folder: Path,
                                num_classes: int,
                                split: str = "train") -> dict:
    non_private_distributions = {}
    for filename in sorted(listdir(non_private_data_distribution_folder)):
        if not filename.startswith("client_") or not filename.endswith(".csv"):
            continue
        client_id = int(filename.replace("client_", "").replace(".csv", ""))
        counts = {str(c): 0 for c in range(num_classes)}
        with open(path.join(non_private_data_distribution_folder, filename), encoding="utf-8-sig") as f:
            reader = DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                if row["split"] != split:
                    continue
                cls = str(int(row["class"]))
                counts[cls] = int(row["count"])
        non_private_distributions[client_id] = counts
    return non_private_distributions


def parse_differentially_private_csv_files(differentially_private_data_distribution_folder: Path,
                                           num_classes: int,
                                           split: str = "train") -> dict:
    differentially_private_distributions = {}
    for filename in sorted(listdir(differentially_private_data_distribution_folder)):
        if not filename.startswith("client_") or not filename.endswith(".csv"):
            continue
        client_id = int(filename.replace("client_", "").replace(".csv", ""))
        counts = {str(c): 0 for c in range(num_classes)}
        with open(path.join(differentially_private_data_distribution_folder, filename), encoding="utf-8-sig") as f:
            reader = DictReader(f)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                if row["split"] != split:
                    continue
                cls = str(int(row["class"]))
                noisy = int(float(row["noisy_count"]))
                counts[cls] = noisy
        differentially_private_distributions[client_id] = counts
    return differentially_private_distributions


def compute_l1_divergences(non_private_distributions: dict,
                           differentially_private_distributions: dict,
                           num_classes: int) -> dict:
    l1_divergences_dict = {cid: sum(abs(non_private_distributions[cid][str(c)] - differentially_private_distributions[cid][str(c)])
                                   for c in range(num_classes))
                           for cid in non_private_distributions}
    return l1_divergences_dict


def draw_difference_lines(ax,
                          x_center,
                          y_np,
                          y_dp,
                          width) -> None:
    if y_np == y_dp:
        return
    diff = int(y_dp - y_np)
    y_high = max(y_np, y_dp)
    for y in (y_np, y_dp):
        ax.plot([x_center - width / 2, x_center + width / 2],
                [y, y],
                linestyle="--",
                color="red",
                linewidth=1,
                zorder=3)
    text = "+{0}".format(diff) if diff > 0 else "{0}".format(diff)
    color = "green" if diff > 0 else "red"
    ax.text(x_center,
            y_high * 1.02,
            text,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color,
            zorder=4)


def plot_aggregate_distributions(non_private_distributions: dict,
                                 differentially_private_distributions: dict,
                                 num_classes: int,
                                 overall_class_distribution_file: Path) -> None:
    np_totals = zeros(num_classes)
    dp_totals = zeros(num_classes)
    for cid in non_private_distributions:
        for c in range(num_classes):
            cls = str(c)
            np_totals[c] += non_private_distributions[cid][cls]
            dp_totals[c] += differentially_private_distributions[cid][cls]
    x = arange(num_classes)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, np_totals, width, label="No Privacy")
    ax.bar(x + width / 2, dp_totals, width, label="Differentially Private")
    y_max = max(np_totals.max(), dp_totals.max())
    ax.set_ylim(0, y_max * 1.3)
    for i in range(num_classes):
        draw_difference_lines(ax, x[i], np_totals[i], dp_totals[i], width)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(overall_class_distribution_file, dpi=300)
    plt.close()


def plot_client_comparison(client_id: int,
                           non_private_distributions: dict,
                           differentially_private_distributions: dict,
                           num_classes: int,
                           output_file: Path) -> None:
    np_vals = [non_private_distributions[client_id][str(c)] for c in range(num_classes)]
    dp_vals = [differentially_private_distributions[client_id][str(c)] for c in range(num_classes)]
    x = arange(num_classes)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, np_vals, width, label="No Privacy")
    ax.bar(x + width / 2, dp_vals, width, label="Differentially Private")
    y_max = max(max(np_vals), max(dp_vals))
    ax.set_ylim(0, y_max * 1.15)
    for i in range(num_classes):
        draw_difference_lines(ax, x[i], np_vals[i], dp_vals[i], width)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(range(num_classes))
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def print_aggregate_summary(non_private_distributions: dict,
                            differentially_private_distributions: dict,
                            num_classes: int) -> None:
    print("\n=== Aggregate Class Distribution Summary ===")
    np_totals = zeros(num_classes, dtype=int)
    dp_totals = zeros(num_classes, dtype=int)
    for cid in non_private_distributions:
        for c in range(num_classes):
            cls = str(c)
            np_totals[c] += non_private_distributions[cid][cls]
            dp_totals[c] += differentially_private_distributions[cid][cls]
    print("Class | No Privacy | DP | Difference")
    print("-------------------------------------")
    for c in range(num_classes):
        diff = int(dp_totals[c] - np_totals[c])
        sign = "+" if diff > 0 else ""
        print("{0:5d} | {1:10d} | {2:6d} | {3}{4}".format(c, np_totals[c], dp_totals[c], sign, diff))
    total_np = np_totals.sum()
    total_dp = dp_totals.sum()
    total_diff = int(total_dp - total_np)
    rel_change = 100 * total_diff / total_np
    print("-------------------------------------")
    print("Total samples (No Privacy): {0}".format(total_np))
    print("Total samples (DP): {0}".format(total_dp))
    print("Total difference: {0}".format(total_diff))
    print("Relative change: {0:.2f}%".format(rel_change))


def print_client_summary(client_id: int,
                         non_private_distributions: dict,
                         differentially_private_distributions: dict,
                         num_classes: int) -> None:
    print("\n=== Client {0} Distribution Summary ===".format(client_id))
    print("Class | No Privacy | DP | Difference")
    print("-------------------------------------")
    l1 = 0
    np_total = 0
    dp_total = 0
    for c in range(num_classes):
        cls = str(c)
        np_c = non_private_distributions[client_id][cls]
        dp_c = differentially_private_distributions[client_id][cls]
        diff = int(dp_c - np_c)
        sign = "+" if diff > 0 else ""
        l1 += abs(diff)
        np_total += np_c
        dp_total += dp_c
        print("{0:5d} | {1:10d} | {2:6d} | {3}{4}".format(c, np_c, dp_c, sign, diff))
    print("-------------------------------------")
    print("Client total (No Privacy): {0}".format(np_total))
    print("Client total (DP): {0}".format(dp_total))
    print("L1 divergence: {0}".format(l1))


def main(non_private_data_distribution_folder: Path,
         differentially_private_data_distribution_folder: Path,
         root_analysis_folder: Path) -> None:

    # Output settings.
    dp_impact_analysis_folder = root_analysis_folder.joinpath("dp_impact_analysis")
    dp_impact_analysis_folder.mkdir(parents=True, exist_ok=True)
    overall_class_distribution_file = dp_impact_analysis_folder.joinpath("overall_class_distribution.pdf")
    max_divergence_client_comparison_file = dp_impact_analysis_folder.joinpath("max_divergence_client_comparison.pdf")
    min_divergence_client_comparison_file = dp_impact_analysis_folder.joinpath("min_divergence_client_comparison.pdf")

    # Detect the number of classes (using the non-private data distribution files).
    num_classes = detect_num_classes(non_private_data_distribution_folder)

    # Parse CSV files (Non-Private and Differentially Private).
    non_private_distributions = parse_non_private_csv_files(non_private_data_distribution_folder,
                                                            num_classes,
                                                            split="train")
    differentially_private_distributions = parse_differentially_private_csv_files(differentially_private_data_distribution_folder,
                                                                                  num_classes,
                                                                                  split="train")

    # Plot the aggregated distributions.
    plot_aggregate_distributions(non_private_distributions, differentially_private_distributions, num_classes, overall_class_distribution_file)
    print_aggregate_summary(non_private_distributions, differentially_private_distributions, num_classes)

    # Compute L1 divergences.
    l1_divergences_dict = compute_l1_divergences(non_private_distributions,
                                                 differentially_private_distributions,
                                                 num_classes)

    # Compute mean (average) L1 divergence across all clients.
    mean_l1_divergence = sum(l1_divergences_dict.values()) / len(l1_divergences_dict)
    print("Mean L1 divergence across clients: {0:.2f}".format(mean_l1_divergence))

    # Get the clients with maximum and minimum L1 divergences.
    max_client = max(l1_divergences_dict, key=l1_divergences_dict.get)
    min_client = min(l1_divergences_dict, key=l1_divergences_dict.get)

    # Plot client with maximum L1 divergence.
    plot_client_comparison(max_client,
                           non_private_distributions,
                           differentially_private_distributions,
                           num_classes,
                           max_divergence_client_comparison_file)
    print_client_summary(max_client, non_private_distributions, differentially_private_distributions, num_classes)

    # Plot client with minimum L1 divergence.
    plot_client_comparison(min_client, non_private_distributions, differentially_private_distributions, num_classes, min_divergence_client_comparison_file)
    print_client_summary(min_client, non_private_distributions, differentially_private_distributions, num_classes)

    # Print the list of saved plots.
    print("\nSaved plots:")
    print(" - {0}".format(overall_class_distribution_file))
    print(" - {0}".format(max_divergence_client_comparison_file))
    print(" - {0}".format(min_divergence_client_comparison_file))

    # Print client with maximum L1 divergence.
    print("Client with maximum L1 divergence: {0} (L1 = {1})".format(max_client, l1_divergences_dict[max_client]))

    # Print client with minimum L1 divergence.
    print("Client with minimum L1 divergence: {0} (L1 = {1})".format(min_client, l1_divergences_dict[min_client]))

    # Compute per-client DP impact ratio.
    dp_impact_ratios = {}
    for cid in non_private_distributions:
        total_samples = sum(non_private_distributions[cid][str(c)] for c in range(num_classes))
        # Avoid division by zero for empty clients.
        if total_samples > 0:
            dp_impact_ratios[cid] = l1_divergences_dict[cid] / total_samples
        else:
            dp_impact_ratios[cid] = 0.0

    # Summary statistics.
    max_ratio_client = max(dp_impact_ratios, key=dp_impact_ratios.get)
    min_ratio_client = min(dp_impact_ratios, key=dp_impact_ratios.get)
    mean_ratio = sum(dp_impact_ratios.values()) / len(dp_impact_ratios)
    print("\n=== DP Noise Impact Ratio per Client ===")
    print("Client with maximum DP impact ratio: {0} ({1:.2f})".format(max_ratio_client, dp_impact_ratios[max_ratio_client]))
    print("Client with minimum DP impact ratio: {0} ({1:.2f})".format(min_ratio_client, dp_impact_ratios[min_ratio_client]))
    print("Mean DP impact ratio across clients: {0:.2f}".format(mean_ratio))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--non-private-data-distribution-folder",
                        type=Path,
                        required=True,
                        help="Relative path to the non private data distribution folder (input)")
    parser.add_argument("--differentially-private-data-distribution-folder",
                        type=Path,
                        required=True,
                        help="Relative path to the differentially private data distribution folder (input)")
    parser.add_argument("--root-analysis-folder",
                        type=Path,
                        required=True,
                        help="Relative path to the root analysis folder (output)")
    args = parser.parse_args()
    non_private_data_distribution_folder = args.non_private_data_distribution_folder
    differentially_private_data_distribution_folder = args.differentially_private_data_distribution_folder
    root_analysis_folder = args.root_analysis_folder
    main(non_private_data_distribution_folder,
         differentially_private_data_distribution_folder,
         root_analysis_folder)
