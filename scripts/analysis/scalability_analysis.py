from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from pandas import DataFrame, concat, read_csv


def _aggregate_lns_summary_files(experiment_folder: Path) -> dict:
    lns_summary_folder = experiment_folder.joinpath("lns_summary")
    if not lns_summary_folder.exists():
        return {}
    lns_summary_files = sorted(lns_summary_folder.glob("*.csv"))
    if not lns_summary_files:
        return {}
    rows = []
    for lns_summary_file in lns_summary_files:
        try:
            df_tmp = read_csv(lns_summary_file)
            if not df_tmp.empty:
                rows.append(df_tmp.iloc[0])
        except Exception as e:
            print("Could not read LNS summary file '{0}': {1}".format(lns_summary_file, repr(e)))
    if not rows:
        return {}
    lns_df = DataFrame(rows)
    lns_metrics = {"num_lns_executions": len(lns_df)}
    if "time_lapsed" in lns_df.columns:
        lns_metrics["avg_lns_wall_time_seconds"] = lns_df["time_lapsed"].mean()
        lns_metrics["total_lns_wall_time_seconds"] = lns_df["time_lapsed"].sum()
    if "cpu_time_seconds" in lns_df.columns:
        lns_metrics["avg_lns_cpu_time_seconds"] = lns_df["cpu_time_seconds"].mean()
        lns_metrics["total_lns_cpu_time_seconds"] = lns_df["cpu_time_seconds"].sum()
    if "rss_delta_mb" in lns_df.columns:
        lns_metrics["avg_lns_memory_delta_mb"] = lns_df["rss_delta_mb"].mean()
        lns_metrics["max_lns_memory_delta_mb"] = lns_df["rss_delta_mb"].max()
    if "rss_peak_mb" in lns_df.columns:
        lns_metrics["avg_lns_rss_peak_mb"] = lns_df["rss_peak_mb"].mean()
        lns_metrics["max_lns_rss_peak_mb"] = lns_df["rss_peak_mb"].max()
    if "num_iterations" in lns_df.columns:
        lns_metrics["avg_lns_num_iterations"] = lns_df["num_iterations"].mean()
        lns_metrics["total_lns_num_iterations"] = lns_df["num_iterations"].sum()
    if "iterations_per_second" in lns_df.columns:
        lns_metrics["avg_lns_iterations_per_second"] = lns_df["iterations_per_second"].mean()
    if "num_accepted_moves" in lns_df.columns:
        lns_metrics["avg_lns_accepted_moves"] = lns_df["num_accepted_moves"].mean()
        lns_metrics["total_lns_accepted_moves"] = lns_df["num_accepted_moves"].sum()
    if "num_improving_moves" in lns_df.columns:
        lns_metrics["avg_lns_improving_moves"] = lns_df["num_improving_moves"].mean()
        lns_metrics["total_lns_improving_moves"] = lns_df["num_improving_moves"].sum()
    return lns_metrics


def load_scalability_results_dataframe(scalability_results_folder: Path) -> DataFrame:
    rows = []
    csv_files = list(scalability_results_folder.rglob("*_scalability_results.csv"))
    if not csv_files:
        error_message = "No '*_scalability_results.csv' files found under '{0}'!".format(scalability_results_folder)
        raise RuntimeError(error_message)
    print("Found {0} '*_scalability_results.csv' files under '{1}'!".format(len(csv_files), scalability_results_folder))
    for csv_path in csv_files:
        df_tmp = read_csv(csv_path)
        parent_dir = csv_path.parent.name
        if parent_dir.startswith("MetaCS-FL"):
            df_tmp["approach_name"] = parent_dir
        else:
            df_tmp["approach_name"] = df_tmp["approach_name"].iloc[0]

        experiment_folder_name = csv_path.name.replace("_scalability_results.csv", "")
        experiment_folder = csv_path.parent.joinpath(experiment_folder_name)
        lns_metrics = _aggregate_lns_summary_files(experiment_folder)
        for metric_name, metric_value in lns_metrics.items():
            df_tmp[metric_name] = metric_value
        if lns_metrics:
            print("Aggregated {0} LNS summary files for '{1}'.".format(
                lns_metrics.get("num_lns_executions", 0), experiment_folder
            ))
        rows.append(df_tmp)

    scalability_results_df = concat(rows, ignore_index=True)
    return scalability_results_df


def normalize_approach_key(name: str) -> str:
    return name.lower().replace("-", "").replace(".", "_")


def _reindex_subset_by_approach(subset: DataFrame,
                                approach_order: list | None) -> DataFrame:
    if approach_order is None:
        return subset
    ordered_approaches = [a for a in approach_order if a in subset["approach_name"].values]
    if not ordered_approaches:
        return subset
    subset = (subset.set_index("approach_name")
              .loc[ordered_approaches]
              .reset_index())
    return subset


def plot_metric_fixed_num_clients_varying_num_tasks(scalability_results_df: DataFrame,
                                                    approach_order: list,
                                                    color_marker_map: dict,
                                                    fixed_clients_folder: Path,
                                                    metric_column: str,
                                                    metric_ylabel: str,
                                                    file_prefix: str) -> None:
    if metric_column not in scalability_results_df.columns:
        print("Skipping plots for '{0}': column not found.".format(metric_column))
        return

    metric_folder = fixed_clients_folder.joinpath(file_prefix)
    metric_folder.mkdir(parents=True, exist_ok=True)

    for num_clients in sorted(scalability_results_df["num_clients"].unique()):
        subset = scalability_results_df[scalability_results_df["num_clients"] == num_clients]
#       plt.figure(figsize=(8, 6))
        plt.figure(figsize=(8, 3.2))
        for approach in approach_order:
            if approach not in subset["approach_name"].values:
                continue
            approach_df = subset[subset["approach_name"] == approach].sort_values("num_tasks")
            key = normalize_approach_key(approach)
            color, marker, label = color_marker_map[key]
            plt.plot(approach_df["num_tasks"],
                     approach_df[metric_column],
                     color=color,
                     marker=marker,
                     linewidth=2,
                     markersize=7,
                     label=label)
        plt.xlabel("Number of Tasks")
        plt.ylabel(metric_ylabel)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        output_file = metric_folder.joinpath("{0}_fixed_{1}_clients.pdf".format(file_prefix, num_clients))
        plt.savefig(output_file, dpi=300)
        plt.close()
        print("Generated the '{0}' file!".format(output_file))


def plot_metric_fixed_num_tasks_varying_num_clients(scalability_results_df: DataFrame,
                                                    approach_order: list,
                                                    color_marker_map: dict,
                                                    fixed_tasks_folder: Path,
                                                    metric_column: str,
                                                    metric_ylabel: str,
                                                    file_prefix: str) -> None:
    if metric_column not in scalability_results_df.columns:
        print("Skipping plots for '{0}': column not found.".format(metric_column))
        return

    metric_folder = fixed_tasks_folder.joinpath(file_prefix)
    metric_folder.mkdir(parents=True, exist_ok=True)

    for num_tasks in sorted(scalability_results_df["num_tasks"].unique()):
        subset = scalability_results_df[scalability_results_df["num_tasks"] == num_tasks]
#       plt.figure(figsize=(8, 6))
        plt.figure(figsize=(8, 3.2))
        for approach in approach_order:
            if approach not in subset["approach_name"].values:
                continue
            approach_df = subset[subset["approach_name"] == approach].sort_values("num_clients")
            key = normalize_approach_key(approach)
            color, marker, label = color_marker_map[key]
            plt.plot(approach_df["num_clients"],
                     approach_df[metric_column],
                     color=color,
                     marker=marker,
                     linewidth=2,
                     markersize=7,
                     label=label)
        plt.xlabel("Number of Clients")
        plt.ylabel(metric_ylabel)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        output_file = metric_folder.joinpath("{0}_fixed_{1}_tasks.pdf".format(file_prefix, num_tasks))
        plt.savefig(output_file, dpi=300)
        plt.close()
        print("Generated the '{0}' file!".format(output_file))


def _build_summary_dataframe(subset: DataFrame) -> DataFrame:
    summary_data = {"Approach": subset["approach_name"],
                    "Rounds": subset["num_rounds"].astype(int),
                    "Total Overhead (s)": subset["total_client_selection_overhead"].round(2),
                    "Avg Overhead / Round (s)": subset["avg_overhead_per_round"].round(4),
                    "Avg CPU / Round (s)": subset["avg_selection_cpu_time_seconds"].round(4),
                    "Avg RSS Delta / Round (MB)": subset["avg_selection_rss_delta_mb"].round(4),
                    "Max RSS Delta (MB)": subset["max_selection_rss_delta_mb"].round(4),
                    "Avg Selected Clients / Round": subset["avg_selected_clients_per_round"].round(2)}

    if "avg_lns_cpu_time_seconds" in subset.columns:
        summary_data["Avg LNS CPU / Execution (s)"] = subset["avg_lns_cpu_time_seconds"].round(4)
    if "avg_lns_memory_delta_mb" in subset.columns:
        summary_data["Avg LNS RSS Delta / Execution (MB)"] = subset["avg_lns_memory_delta_mb"].round(4)
    if "avg_lns_num_iterations" in subset.columns:
        summary_data["Avg LNS Iterations / Execution"] = subset["avg_lns_num_iterations"].round(2)

    summary_df = DataFrame(summary_data)
    return summary_df


def print_scalability_summaries_fixed_clients(scalability_results_df: DataFrame,
                                              approach_order: list | None = None) -> None:
    print("\n\n==============================")
    print("Fixed number of clients, varying number of tasks")
    print("==============================")

    tuples = (scalability_results_df[["num_clients", "num_tasks"]]
              .drop_duplicates()
              .sort_values(["num_clients", "num_tasks"])
              .itertuples(index=False))

    for num_clients, num_tasks in tuples:
        subset = scalability_results_df[(scalability_results_df["num_clients"] == num_clients) &
                                        (scalability_results_df["num_tasks"] == num_tasks)]
        subset = _reindex_subset_by_approach(subset, approach_order)
        summary_df = _build_summary_dataframe(subset)
        print("\nNum_Clients = {0}, Num_Tasks = {1}".format(num_clients, num_tasks))
        print(summary_df.to_string(index=False))


def print_scalability_summaries_fixed_tasks(scalability_results_df: DataFrame,
                                            approach_order: list | None = None) -> None:
    print("\n\n==============================")
    print("Fixed number of tasks, varying number of clients")
    print("==============================")

    tuples = (scalability_results_df[["num_tasks", "num_clients"]]
              .drop_duplicates()
              .sort_values(["num_tasks", "num_clients"])
              .itertuples(index=False))

    for num_tasks, num_clients in tuples:
        subset = scalability_results_df[(scalability_results_df["num_clients"] == num_clients) &
                                        (scalability_results_df["num_tasks"] == num_tasks)]
        subset = _reindex_subset_by_approach(subset, approach_order)
        summary_df = _build_summary_dataframe(subset)
        print("\nNum_Tasks = {0}, Num_Clients = {1}".format(num_tasks, num_clients))
        print(summary_df.to_string(index=False))


def export_full_summary_tables(scalability_results_df: DataFrame,
                               approach_order: list,
                               tables_folder: Path) -> None:
    tables_folder.mkdir(parents=True, exist_ok=True)

    tuples = (scalability_results_df[["num_clients", "num_tasks"]]
              .drop_duplicates()
              .sort_values(["num_clients", "num_tasks"])
              .itertuples(index=False))

    for num_clients, num_tasks in tuples:
        subset = scalability_results_df[(scalability_results_df["num_clients"] == num_clients) &
                                        (scalability_results_df["num_tasks"] == num_tasks)]
        subset = _reindex_subset_by_approach(subset, approach_order)

        summary_data = {"Approach": subset["approach_name"],
                        "Rounds": subset["num_rounds"].astype(int),
                        "Total_Overhead_s": subset["total_client_selection_overhead"].round(4),
                        "Avg_Overhead_per_Round_s": subset["avg_overhead_per_round"].round(6),
                        "Avg_CPU_per_Round_s": subset["avg_selection_cpu_time_seconds"].round(6),
                        "Avg_RSS_Delta_per_Round_MB": subset["avg_selection_rss_delta_mb"].round(6),
                        "Max_RSS_Delta_MB": subset["max_selection_rss_delta_mb"].round(6),
                        "Avg_Selected_Clients_per_Round": subset["avg_selected_clients_per_round"].round(4)}

        if "num_lns_executions" in subset.columns:
            summary_data["Num_LNS_Executions"] = subset["num_lns_executions"]
        if "avg_lns_wall_time_seconds" in subset.columns:
            summary_data["Avg_LNS_Wall_Time_s"] = subset["avg_lns_wall_time_seconds"].round(6)
        if "avg_lns_cpu_time_seconds" in subset.columns:
            summary_data["Avg_LNS_CPU_s"] = subset["avg_lns_cpu_time_seconds"].round(6)
        if "avg_lns_memory_delta_mb" in subset.columns:
            summary_data["Avg_LNS_RSS_Delta_MB"] = subset["avg_lns_memory_delta_mb"].round(6)
        if "avg_lns_num_iterations" in subset.columns:
            summary_data["Avg_LNS_Iterations"] = subset["avg_lns_num_iterations"].round(4)

        summary_df = DataFrame(summary_data)
        csv_output_file = tables_folder.joinpath("summary_c{0}_t{1}.csv".format(num_clients, num_tasks))
        tex_output_file = tables_folder.joinpath("summary_c{0}_t{1}.tex".format(num_clients, num_tasks))
        summary_df.to_csv(csv_output_file, index=False)
        with tex_output_file.open("w", encoding="utf-8") as f:
            f.write(summary_df.to_latex(index=False, escape=False, float_format="%.4f"))
        print("Generated the '{0}' file!".format(csv_output_file))
        print("Generated the '{0}' file!".format(tex_output_file))


def export_representative_overhead_tables(scalability_results_df: DataFrame,
                                          approach_order: list,
                                          representative_tables_folder: Path,
                                          representative_num_clients: list,
                                          representative_num_tasks: list) -> None:
    representative_tables_folder.mkdir(parents=True, exist_ok=True)

    # Fixed number of clients, varying representative task counts.
    for num_clients in representative_num_clients:
        subset = scalability_results_df[(scalability_results_df["num_clients"] == num_clients) &
                                        (scalability_results_df["num_tasks"].isin(representative_num_tasks))]
        if subset.empty:
            continue

        rows = []
        for approach in approach_order:
            approach_subset = subset[subset["approach_name"] == approach]
            if approach_subset.empty:
                continue
            row = {"Approach": approach}
            for num_tasks in representative_num_tasks:
                metric_subset = approach_subset[approach_subset["num_tasks"] == num_tasks]
                if metric_subset.empty:
                    continue
                row["T{0}_TotalOverhead_s".format(num_tasks)] = round(metric_subset["total_client_selection_overhead"].iloc[0], 2)
                row["T{0}_AvgOverhead_s".format(num_tasks)] = round(metric_subset["avg_overhead_per_round"].iloc[0], 4)
                row["T{0}_AvgCPU_s".format(num_tasks)] = round(metric_subset["avg_selection_cpu_time_seconds"].iloc[0], 4)
                row["T{0}_AvgRSSDelta_MB".format(num_tasks)] = round(metric_subset["avg_selection_rss_delta_mb"].iloc[0], 4)
                row["T{0}_MaxRSSDelta_MB".format(num_tasks)] = round(metric_subset["max_selection_rss_delta_mb"].iloc[0], 4)
            rows.append(row)

        table_df = DataFrame(rows)
        csv_output_file = representative_tables_folder.joinpath("fixed_{0}_clients_overhead_summary.csv".format(num_clients))
        tex_output_file = representative_tables_folder.joinpath("fixed_{0}_clients_overhead_summary.tex".format(num_clients))
        table_df.to_csv(csv_output_file, index=False)
        with tex_output_file.open("w", encoding="utf-8") as f:
            f.write(table_df.to_latex(index=False, escape=False))
        print("Generated the '{0}' file!".format(csv_output_file))
        print("Generated the '{0}' file!".format(tex_output_file))

    # Fixed number of tasks, varying representative client counts.
    for num_tasks in representative_num_tasks:
        subset = scalability_results_df[(scalability_results_df["num_tasks"] == num_tasks) &
                                        (scalability_results_df["num_clients"].isin(representative_num_clients))]
        if subset.empty:
            continue

        rows = []
        for approach in approach_order:
            approach_subset = subset[subset["approach_name"] == approach]
            if approach_subset.empty:
                continue
            row = {"Approach": approach}
            for num_clients in representative_num_clients:
                metric_subset = approach_subset[approach_subset["num_clients"] == num_clients]
                if metric_subset.empty:
                    continue
                row["C{0}_TotalOverhead_s".format(num_clients)] = round(metric_subset["total_client_selection_overhead"].iloc[0], 2)
                row["C{0}_AvgOverhead_s".format(num_clients)] = round(metric_subset["avg_overhead_per_round"].iloc[0], 4)
                row["C{0}_AvgCPU_s".format(num_clients)] = round(metric_subset["avg_selection_cpu_time_seconds"].iloc[0], 4)
                row["C{0}_AvgRSSDelta_MB".format(num_clients)] = round(metric_subset["avg_selection_rss_delta_mb"].iloc[0], 4)
                row["C{0}_MaxRSSDelta_MB".format(num_clients)] = round(metric_subset["max_selection_rss_delta_mb"].iloc[0], 4)
            rows.append(row)

        table_df = DataFrame(rows)
        csv_output_file = representative_tables_folder.joinpath("fixed_{0}_tasks_overhead_summary.csv".format(num_tasks))
        tex_output_file = representative_tables_folder.joinpath("fixed_{0}_tasks_overhead_summary.tex".format(num_tasks))
        table_df.to_csv(csv_output_file, index=False)
        with tex_output_file.open("w", encoding="utf-8") as f:
            f.write(table_df.to_latex(index=False, escape=False))
        print("Generated the '{0}' file!".format(csv_output_file))
        print("Generated the '{0}' file!".format(tex_output_file))


def export_resource_overhead_table_for_paper(scalability_results_df: DataFrame,
                                             approach_order: list,
                                             output_file_csv: Path,
                                             output_file_tex: Path,
                                             selected_num_clients: int,
                                             selected_num_tasks: int) -> None:
    subset = scalability_results_df[(scalability_results_df["num_clients"] == selected_num_clients) &
                                    (scalability_results_df["num_tasks"] == selected_num_tasks)]
    subset = _reindex_subset_by_approach(subset, approach_order)

    if subset.empty:
        print("Skipping representative paper table: no data for num_clients={0}, num_tasks={1}".format(
            selected_num_clients, selected_num_tasks
        ))
        return

    table_df = DataFrame({"Approach": subset["approach_name"],
                          "Avg Time / Round (s)": subset["avg_overhead_per_round"].round(4),
                          "Avg CPU / Round (s)": subset["avg_selection_cpu_time_seconds"].round(4),
                          "Avg Memory Delta / Round (MB)": subset["avg_selection_rss_delta_mb"].round(4),
                          "Max Memory Delta (MB)": subset["max_selection_rss_delta_mb"].round(4)})
    table_df.to_csv(output_file_csv, index=False)
    with output_file_tex.open("w", encoding="utf-8") as f:
        f.write(table_df.to_latex(index=False, escape=False))
    print("Generated the '{0}' file!".format(output_file_csv))
    print("Generated the '{0}' file!".format(output_file_tex))


def export_lns_overhead_table_for_paper(scalability_results_df: DataFrame,
                                        approach_order: list,
                                        output_file_csv: Path,
                                        output_file_tex: Path,
                                        selected_num_clients: int,
                                        selected_num_tasks: int) -> None:
    lns_required_columns = ["avg_lns_cpu_time_seconds", "avg_lns_memory_delta_mb", "avg_lns_num_iterations"]
    if not all(column in scalability_results_df.columns for column in lns_required_columns):
        print("Skipping LNS overhead export: required LNS-specific columns were not found in the scalability results.")
        return

    subset = scalability_results_df[(scalability_results_df["num_clients"] == selected_num_clients) &
                                    (scalability_results_df["num_tasks"] == selected_num_tasks) &
                                    (scalability_results_df["approach_name"].astype(str).str.startswith("MetaCS-FL"))]
    subset = _reindex_subset_by_approach(subset, approach_order)

    if subset.empty:
        print("Skipping LNS paper table: no MetaCS-FL data for num_clients={0}, num_tasks={1}".format(
            selected_num_clients, selected_num_tasks
        ))
        return

    table_df = DataFrame({"Variant": subset["approach_name"],
                          "LNS Executions": subset["num_lns_executions"].astype(int) if "num_lns_executions" in subset.columns else 0,
                          "Avg LNS Wall Time / Execution (s)": subset["avg_lns_wall_time_seconds"].round(4) if "avg_lns_wall_time_seconds" in subset.columns else 0,
                          "Avg LNS CPU / Execution (s)": subset["avg_lns_cpu_time_seconds"].round(4),
                          "Avg LNS Memory Delta / Execution (MB)": subset["avg_lns_memory_delta_mb"].round(4),
                          "Avg LNS Iterations / Execution": subset["avg_lns_num_iterations"].round(2)})
    table_df.to_csv(output_file_csv, index=False)
    with output_file_tex.open("w", encoding="utf-8") as f:
        f.write(table_df.to_latex(index=False, escape=False))
    print("Generated the '{0}' file!".format(output_file_csv))
    print("Generated the '{0}' file!".format(output_file_tex))


def main(root_results_folder: Path,
         root_analysis_folder: Path) -> None:
    # Input settings.
    if root_results_folder.name == "scalability_results":
        scalability_results_folder = root_results_folder
    else:
        scalability_results_folder = root_results_folder.joinpath("scalability_results")

    # Output settings.
    scalability_analysis_folder = root_analysis_folder.joinpath("scalability_analysis")
    fixed_clients_folder = scalability_analysis_folder.joinpath("fixed_clients")
    fixed_clients_folder.mkdir(parents=True, exist_ok=True)
    fixed_tasks_folder = scalability_analysis_folder.joinpath("fixed_tasks")
    fixed_tasks_folder.mkdir(parents=True, exist_ok=True)
    tables_folder = scalability_analysis_folder.joinpath("summary_tables")
    tables_folder.mkdir(parents=True, exist_ok=True)
    representative_tables_folder = scalability_analysis_folder.joinpath("representative_tables")
    representative_tables_folder.mkdir(parents=True, exist_ok=True)

    # Set the order of approaches.
    approach_order = ["FedAvg",
                      "MEC",
                      "ECMTC",
                      "Oort",
                      "DivFL",
                      "ECSM",
                      "MetaCS-FL_015",
                      "MetaCS-FL_025",
                      "MetaCS-FL_050"]

    # Set the plot styling of approaches.
    color_marker_map = {"fedavg": ("blue", "o", "FedAvg"),
                        "mec": ("gray", "h", "MEC"),
                        "ecmtc": ("green", "^", "ECMTC"),
                        "oort": ("purple", "D", "Oort"),
                        "divfl": ("brown", "v", "DivFL"),
                        "ecsm": ("black", "P", "ECSM"),
                        "metacsfl_015": ("red", "s", "MetaCS-FL_0.15"),
                        "metacsfl_025": ("orange", "*", "MetaCS-FL_0.25"),
                        "metacsfl_050": ("teal", "X", "MetaCS-FL_0.5")}

    metrics_to_plot = [{"column": "total_client_selection_overhead",
                        "ylabel": "Client Selection Overhead (s)",
                        "prefix": "total_overhead"},
                       {"column": "avg_selection_cpu_time_seconds",
                        "ylabel": "Average CPU Time / Round (s)",
                        "prefix": "avg_cpu_overhead"},
                       {"column": "avg_selection_rss_delta_mb",
                        "ylabel": "Average Memory Delta / Round (MB)",
                        "prefix": "avg_memory_delta"},
                       {"column": "max_selection_rss_delta_mb",
                        "ylabel": "Maximum Memory Delta (MB)",
                        "prefix": "max_memory_delta"},
                       {"column": "avg_lns_cpu_time_seconds",
                        "ylabel": "Average LNS CPU Time / Execution (s)",
                        "prefix": "avg_lns_cpu_overhead"},
                       {"column": "avg_lns_memory_delta_mb",
                        "ylabel": "Average LNS Memory Delta / Execution (MB)",
                        "prefix": "avg_lns_memory_delta"},
                       {"column": "avg_lns_num_iterations",
                        "ylabel": "Average LNS Iterations / Execution",
                        "prefix": "avg_lns_iterations"}]

    # Representative subsets used for paper tables.
    representative_num_clients = [10, 50, 100, 500, 1000]
    representative_num_tasks = [100, 1000, 10000, 30000, 40000]

    # Load data.
    scalability_results_df = load_scalability_results_dataframe(scalability_results_folder)

    # Generate plots for all overhead metrics.
    for metric_info in metrics_to_plot:
        plot_metric_fixed_num_clients_varying_num_tasks(scalability_results_df,
                                                        approach_order,
                                                        color_marker_map,
                                                        fixed_clients_folder,
                                                        metric_info["column"],
                                                        metric_info["ylabel"],
                                                        metric_info["prefix"])
        plot_metric_fixed_num_tasks_varying_num_clients(scalability_results_df,
                                                        approach_order,
                                                        color_marker_map,
                                                        fixed_tasks_folder,
                                                        metric_info["column"],
                                                        metric_info["ylabel"],
                                                        metric_info["prefix"])

    # Print both analysis orientations.
    print_scalability_summaries_fixed_clients(scalability_results_df, approach_order)
    print_scalability_summaries_fixed_tasks(scalability_results_df, approach_order)

    # Export full summary tables for every (num_clients, num_tasks) pair.
    export_full_summary_tables(scalability_results_df,
                               approach_order,
                               tables_folder)

    # Export representative multi-metric tables for the paper.
    export_representative_overhead_tables(scalability_results_df,
                                          approach_order,
                                          representative_tables_folder,
                                          representative_num_clients,
                                          representative_num_tasks)

    # Export one compact resource-overhead table for a representative large-scale scenario.
    export_resource_overhead_table_for_paper(scalability_results_df,
                                             approach_order,
                                             representative_tables_folder.joinpath("paper_resource_overhead_table.csv"),
                                             representative_tables_folder.joinpath("paper_resource_overhead_table.tex"),
                                             selected_num_clients=1000,
                                             selected_num_tasks=40000)

    # Export one LNS-specific table for MetaCS-FL if the necessary columns are available.
    export_lns_overhead_table_for_paper(scalability_results_df,
                                        approach_order,
                                        representative_tables_folder.joinpath("paper_lns_overhead_table.csv"),
                                        representative_tables_folder.joinpath("paper_lns_overhead_table.tex"),
                                        selected_num_clients=1000,
                                        selected_num_tasks=40000)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root-results-folder",
                        type=Path,
                        required=True,
                        help="Relative path to the root results folder (input)")
    parser.add_argument("--root-analysis-folder",
                        type=Path,
                        required=True,
                        help="Relative path to the root analysis folder (output)")
    args = parser.parse_args()
    main(args.root_results_folder, args.root_analysis_folder)
