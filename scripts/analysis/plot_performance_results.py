from argparse import ArgumentParser
from matplotlib.pyplot import figure, fill_between, gca, grid, legend, plot, savefig, show, text, tight_layout, xlabel, ylabel
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter, FixedLocator, MaxNLocator, MultipleLocator
from math import floor, ceil
from numpy import arange
from pandas import concat, DataFrame, read_csv
from pathlib import Path


def construct_base_path(results_folder: Path,
                        dataset_name: str,
                        dataset_distribution: str,
                        approach: str,
                        exec_id: int,
                        r_suffix: str) -> Path:
    base_name = "{0}_exec_{1}{2}".format(approach, exec_id, r_suffix)
    base_path = results_folder / dataset_name / dataset_distribution / base_name
    return base_path


def load_dataframe(csv_file: Path,
                   columns_to_remove: list = None) -> DataFrame:
    df = read_csv(csv_file)
    if columns_to_remove:
        existing_columns = [col for col in columns_to_remove if col in df.columns]
        df = df.drop(columns=existing_columns)
    return df


def get_fl_round_to_target_accuracy(individual_metrics_history_df: DataFrame,
                                    target_accuracy: float,
                                    phase: str) -> tuple:
    fl_round_column = "comm_round"
    loss_column = individual_metrics_history_df.columns[individual_metrics_history_df.columns.str.contains("loss")][0]
    accuracy_column = individual_metrics_history_df.columns[individual_metrics_history_df.columns.str.contains("accuracy")][0]
    num_examples_column = individual_metrics_history_df.columns[individual_metrics_history_df.columns.str.contains("examples")][0]
    individual_metrics_history_df["weighted_loss"] = individual_metrics_history_df[loss_column] * individual_metrics_history_df[num_examples_column]
    individual_metrics_history_df["weighted_accuracy"] = individual_metrics_history_df[accuracy_column] * individual_metrics_history_df[num_examples_column]
    round_metrics = (individual_metrics_history_df.groupby(fl_round_column)
                     .agg(total_weighted_loss=("weighted_loss", "sum"),
                          total_weighted_accuracy=("weighted_accuracy", "sum"),
                          total_examples=(num_examples_column, "sum"))
                     .reset_index())
    round_metrics["weighted_mean_{0}ing_loss".format(phase)] = round_metrics["total_weighted_loss"] / round_metrics["total_examples"]
    round_metrics["weighted_mean_{0}ing_accuracy".format(phase)] = round_metrics["total_weighted_accuracy"] / round_metrics["total_examples"]
    round_metrics = round_metrics.sort_values(by=fl_round_column)
    weighted_mean_accuracy_per_round_df = round_metrics[["comm_round", "weighted_mean_{0}ing_accuracy".format(phase)]]
    round_of_target_accuracy = round_metrics[round_metrics["weighted_mean_{0}ing_accuracy".format(phase)] >= target_accuracy]
    if round_of_target_accuracy.empty:
        final_fl_round = individual_metrics_history_df[fl_round_column].iloc[-1]
        return final_fl_round, weighted_mean_accuracy_per_round_df
    fl_round_to_target_accuracy = int(round_of_target_accuracy.iloc[0][fl_round_column])
    return fl_round_to_target_accuracy, weighted_mean_accuracy_per_round_df


def load_multiple_trials(base_path_template: Path,
                         num_trials: int,
                         phase: str) -> list:
    dfs = []
    metrics_file = "individual_fit_metrics_history.csv" if phase == "train" else "individual_evaluate_metrics_history.csv"
    parent = base_path_template.parent
    for i in range(1, num_trials + 1):
        name = base_path_template.name
        prefix = name.split("_exec_", 1)[0]
        suffix = name[len("{0}_exec_1".format(prefix)):]
        trial_path = parent / "{0}_exec_{1}{2}".format(prefix, i, suffix)
        metrics_path = trial_path / "output" / metrics_file
        if not metrics_path.exists():
            print("[WARNING] Missing metrics for trial {0}: {1}".format(i, metrics_path))
            continue
        dfs.append(load_dataframe(metrics_path))
    return dfs


def aggregate_trials(dfs: list,
                     target_accuracy: float,
                     phase: str) -> DataFrame:
    if not dfs:
        return None
    all_dfs = []
    for df in dfs:
        _, accuracy_df = get_fl_round_to_target_accuracy(df, target_accuracy, phase)
        accuracy_df = accuracy_df.set_index("comm_round")
        all_dfs.append(accuracy_df["weighted_mean_{0}ing_accuracy".format(phase)])
    if not all_dfs:
        return None
    combined = concat(all_dfs, axis=1)
    combined.columns = range(len(all_dfs))
    mean = combined.mean(axis=1)
    std = combined.std(axis=1)
    return DataFrame({"comm_round": combined.index,
                      "mean_accuracy": mean,
                      "std_accuracy": std}).reset_index(drop=True)


def smart_format(value: float) -> str:
    return "{0:.2f}".format(value) if float("{0:.4f}".format(value)) == float("{0:.2f}".format(value)) else "{0:.4f}".format(value)


def annotate(x, y, annotation_text, color):
    text(x, y, annotation_text,
         ha="center",
         va="bottom",
         fontsize=9,
         fontweight="bold",
         color=color,
         path_effects=[withStroke(linewidth=2.5, foreground="white")])


def plot_with_shaded_area(mean_std_df: DataFrame,
                          label: str,
                          color: str,
                          seen_x: dict,
                          marker: str = "o",
                          markersize: int = 3):
    rounds = mean_std_df["comm_round"]
    mean = mean_std_df["mean_accuracy"]
    std = mean_std_df["std_accuracy"]
    plot(rounds, mean, label=label, color=color, marker=marker, markersize=markersize)
    fill_between(rounds, mean - std, mean + std, alpha=0.2, color=color)
    y0, y1 = gca().get_ylim()
    offset = (y1 - y0) * 0.02
    last_x = rounds.iloc[-1]
    last_y = mean.iloc[-1]
    text_value = smart_format(last_y)
    key = int(round(last_x))
    seen_x.setdefault(key, [])
    seen_x[key].append((last_y, color, text_value))
    entries = sorted(seen_x[key], key=lambda t: t[0])
    n = len(entries)
    mid = (n - 1) / 2
    current_entry = seen_x[key][-1]
    current_index = entries.index(current_entry)
    dy = (current_index - mid) * offset
    annotate(last_x, last_y + dy, text_value, color)


def compute_summary_for_approach(base_path_template: Path,
                                 num_trials: int,
                                 phase: str) -> dict:
    sel_file = "selected_fit_clients_history.csv" if phase == "train" else "selected_evaluate_clients_history.csv"
    met_file = "individual_fit_metrics_history.csv" if phase == "train" else "individual_evaluate_metrics_history.csv"
    ds_col = "ds_train_i" if phase == "train" else "ds_test_i"
    trial_results = []
    parent = base_path_template.parent
    for i in range(1, num_trials + 1):
        name = base_path_template.name
        prefix = name.split("_exec_", 1)[0]
        suffix = name[len("{0}_exec_1".format(prefix)):]
        trial_path = parent / "{0}_exec_{1}{2}".format(prefix, i, suffix)
        sel_path = trial_path / "output" / sel_file
        met_path = trial_path / "output" / met_file
        try:
            df_sel = read_csv(sel_path)
            df_met = read_csv(met_path)
        except:
            print("[WARNING] Missing trial {0}".format(i))
            continue
        df_sel_rounds = df_sel.groupby("comm_round").agg(selected=("num_selected_clients", "sum"),
                                                         allocated_tasks=("num_tasks", "sum"))
        df_completed = df_met.groupby("comm_round").agg(completed_clients=("client_id", "count"),
                                                        completed_tasks=(ds_col, "sum"))
        merged = df_sel_rounds.merge(df_completed, left_index=True, right_index=True, how="left").fillna(0)
        merged["failed_clients"] = merged["selected"] - merged["completed_clients"]
        trial_avg = {"selected": merged["selected"].mean(),
                     "failed": merged["failed_clients"].mean(),
                     "allocated_tasks": merged["allocated_tasks"].mean(),
                     "completed_tasks": merged["completed_tasks"].mean()}
        trial_results.append(trial_avg)
    if not trial_results:
        return None
    df = DataFrame(trial_results)
    avg = df.mean().to_dict()
    avg["pct_failed_clients"] = 100 * avg["failed"] / avg["selected"] if avg["selected"] > 0 else 0
    avg["pct_completed_tasks"] = 100 * avg["completed_tasks"] / avg["allocated_tasks"] if avg["allocated_tasks"] > 0 else 0
    return avg


def process_experiments_for_r(results_folder: Path,
                              root_analysis_folder: Path,
                              experiment_tag: str,
                              dataset_name: str,
                              dataset_distribution: str,
                              r_suffix: str,
                              num_trials: int,
                              phase: str,
                              target_weighted_mean_testing_accuracy: float,
                              num_available_clients: int,
                              all_approaches: list):
    base_paths = {}
    for key in all_approaches:
        base_paths[key] = construct_base_path(results_folder,
                                              dataset_name,
                                              dataset_distribution,
                                              key,
                                              1,
                                              r_suffix)
    mean_std = {}
    for key, path in base_paths.items():
        dfs = load_multiple_trials(path, num_trials, phase)
        ms = aggregate_trials(dfs, target_weighted_mean_testing_accuracy, phase)
        if ms is not None:
            mean_std[key] = ms
    if not mean_std:
        print("[ERROR] No experiment produced usable data. Skipping plot.")
        return
    figure(figsize=(7.2, 4.0))
    seen_x = {}
    color_marker_map = {"fedavg": ("blue", "o", "FedAvg"),
                        "ecmtc": ("green", "^", "ECMTC"),
                        "mec": ("gray", "h", "MEC"),
                        "oort": ("purple", "D", "Oort"),
                        "divfl": ("brown", "v", "DivFL"),
                        "ecsm": ("black", "P", "ECSM"),
                        "metacsfl": ("red", "s", "MetaCS-FL"),
                        "metacsfl_no_privacy": ("orange", "*", "MetaCS-FL No Privacy")}
    for key, df in mean_std.items():
        color, marker, label = color_marker_map[key]
        plot_with_shaded_area(df, label, color, seen_x, marker, 3)
    ax = gca()
    # Smart axes.
    all_x = []
    all_y_low = []
    all_y_high = []
    for df in mean_std.values():
        all_x.extend(df["comm_round"].values)
        all_y_low.extend((df["mean_accuracy"] - df["std_accuracy"]).values)
        all_y_high.extend((df["mean_accuracy"] + df["std_accuracy"]).values)
    xmin = 0
    xmax = max(all_x)
    ymin = min(all_y_low)
    ymax = max(all_y_high)
    ax.set_xlim(xmin, xmax)
    if xmax <= xmin:
        x_step = 1
    else:
        raw_step = (xmax - xmin) / 10
        x_step = int(round(raw_step))
        if x_step < 1:
            x_step = 1
        if x_step > 1 and x_step % 5 != 0:
            x_step = int(round(x_step / 5) * 5)
        if (xmax - xmin) <= 10:
            x_step = 1
    if x_step == 0:
        x_step = 1
    x_ticks = list(range(xmin, xmax + 1, x_step))
    for required in (1, xmax):
        if required not in x_ticks:
            x_ticks.append(required)
    if 0 in x_ticks:
        x_ticks.remove(0)
    x_ticks = sorted(set(x_ticks))
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.set_xticks(x_ticks)
    y_range = ymax - ymin
    ymin -= 0.05 * y_range
    ymax += 0.05 * y_range
    ymin = max(0.0, floor(ymin * 10) / 10)
    ymax = min(1.0, ceil(ymax * 10) / 10)
    ax.set_ylim(ymin, ymax)
    if ymax - ymin <= 0.3:
        y_step = 0.05
    else:
        y_step = 0.1
    y_ticks = list(arange(ymin, ymax + 1e-9, y_step))
    y_ticks = [t for t in y_ticks if t != 0]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{0:.2f}".format(y).rstrip("0").rstrip(".")))

    xlabel("FL Round")
    ylabel("{0} Accuracy".format(phase.capitalize() if phase == "test" else phase.capitalize() + "ing"))
    legend(loc="lower right", frameon=True, framealpha=0.9)
    grid(True)
    tight_layout()

    figure_file = root_analysis_folder.joinpath("{0}/{1}ing_accuracy_{2}_{3}{4}_{5}_clients.pdf" \
                                                .format(experiment_tag,
                                                        phase,
                                                        dataset_name,
                                                        dataset_distribution,
                                                        r_suffix,
                                                        num_available_clients))
    savefig(figure_file, dpi=300, bbox_inches="tight")

    summary_rows = []
    for key in all_approaches:
        path = base_paths.get(key)
        if path is None:
            continue
        summary = compute_summary_for_approach(path, num_trials, phase)
        if summary is None:
            continue
        label = color_marker_map[key][2]
        summary_rows.append([label,
                             summary["selected"],
                             summary["failed"],
                             summary["allocated_tasks"],
                             summary["completed_tasks"],
                             summary["pct_failed_clients"],
                             summary["pct_completed_tasks"]])
    if summary_rows:
        summary_table = DataFrame(summary_rows, columns=["Approach",
                                                         "Selected Clients",
                                                         "Failed Clients",
                                                         "Allocated Tasks",
                                                         "Completed Tasks",
                                                         "% Failed Clients",
                                                         "% Completed Tasks"])
        print(summary_table.to_string(index=False))


def main(performance_results_folder: Path,
         root_analysis_folder: Path,
         experiment_tag: str,
         dataset_name: str,
         dataset_distribution: str,
         num_available_clients: int,
         num_trials: int,
         phase: str,
         all_approaches: str) -> None:

    # Create the experiment tag, if needed.
    experiment_tag_folder = root_analysis_folder.joinpath("{0}".format(experiment_tag))
    experiment_tag_folder.mkdir(parents=True, exist_ok=True)

    # Get the specific performance results folder.
    specific_performance_results_folder = performance_results_folder.joinpath("{0}_clients".format(num_available_clients))

    # Get the list of all approaches.
    all_approaches = [item.strip() for item in all_approaches.split(",")]

    # Set the target testing accuracies.
    target_testing_accuracies = {"cifar_10": {"iid": 0.75, "non_iid": 0.45},
                                 "fashion_mnist": {"iid": 0.85, "non_iid": 0.70},
                                 "emotion": {"iid": 0.9, "non_iid": 0.8}}

    # Get the corresponding target testing accuracy.
    target_weighted_mean_testing_accuracy = target_testing_accuracies[dataset_name][dataset_distribution]

    print("\n")
    print("- Dataset Name:", dataset_name)
    print("- Dataset Distribution:", dataset_distribution)
    print("- Target Testing Accuracy:", target_weighted_mean_testing_accuracy)
    print("- Phase:", phase)
    print("- Number of Trials:", num_trials)
    print("\n")

    # Calculate metrics summary for all approaches.
    # r_list = [1, 10, 25, 50, 75]
    r_list = [""]
    for r in r_list:
        if r == "":
            r_suffix = ""
        else:
            r_suffix = "_r_{0}".format(r)
        process_experiments_for_r(specific_performance_results_folder,
                                  root_analysis_folder,
                                  experiment_tag,
                                  dataset_name,
                                  dataset_distribution,
                                  r_suffix,
                                  num_trials,
                                  phase,
                                  target_weighted_mean_testing_accuracy,
                                  num_available_clients,
                                  all_approaches)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--performance-results-folder",
                        type=Path,
                        required=True,
                        help="Relative path to the performance results folder (input)")
    parser.add_argument("--root-analysis-folder",
                        type=Path,
                        required=True,
                        help="Relative path to the root analysis folder (output)")
    parser.add_argument("--experiment-tag",
                        type=str,
                        required=True,
                        help="Unique tag for the experiment (e.g., performance, dp_impact)")
    parser.add_argument("--dataset-name",
                        type=str,
                        required=True,
                        help="Name of the dataset")
    parser.add_argument("--dataset-distribution",
                        type=str,
                        required=True,
                        help="Distribution of the dataset")
    parser.add_argument("--num-available-clients",
                        type=int,
                        required=True,
                        help="Number of available clients")
    parser.add_argument("--num-trials",
                        type=int,
                        required=True,
                        help="Number of trials (independent executions)")
    parser.add_argument("--phase",
                        type=str,
                        required=True,
                        help="Phase of interest (train / test)")
    parser.add_argument("--all-approaches",
                        type=str,
                        required=True,
                        help="All approaches (comma-separated)")
    args = parser.parse_args()
    performance_results_folder = args.performance_results_folder
    root_analysis_folder = args.root_analysis_folder
    experiment_tag = args.experiment_tag
    dataset_name = args.dataset_name
    dataset_distribution = args.dataset_distribution
    num_available_clients = args.num_available_clients
    num_trials = args.num_trials
    phase = args.phase
    all_approaches = args.all_approaches
    main(performance_results_folder,
         root_analysis_folder,
         experiment_tag,
         dataset_name,
         dataset_distribution,
         num_available_clients,
         num_trials,
         phase,
         all_approaches)
