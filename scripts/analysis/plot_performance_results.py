from argparse import ArgumentParser
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, fill_between, gca, grid, legend, plot, savefig, text, tight_layout, xlabel, ylabel
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter, FixedLocator
from math import floor, ceil
from numpy import arange
from pandas import concat, DataFrame, read_csv
from pathlib import Path


APPROACH_STYLE_MAP = {"fedavg": ("blue", "o", "FedAvg"),
                      "ecmtc": ("green", "^", "ECMTC"),
                      "mec": ("gray", "h", "MEC"),
                      "oort": ("purple", "D", "Oort"),
                      "divfl": ("brown", "v", "DivFL"),
                      "ecsm": ("black", "P", "ECSM"),
                      "fedcab": ("olive", "X", "FedCAB"),
                      "rifles": ("teal", "<", "RIFLES"),
                      "rifles_gh": ("teal", "<", "RIFLES-GH"),
                      "metacsfl": ("red", "s", "MetaCS-FL"),
                      "metacsfl_no_privacy": ("orange", "*", "MetaCS-FL No Privacy")}


FALLBACK_STYLES = [("tab:blue", "o"), ("tab:orange", "s"), ("tab:green", "^"),
                   ("tab:red", "D"), ("tab:purple", "v"), ("tab:brown", "P"),
                   ("tab:pink", "X"), ("tab:gray", "h"), ("tab:olive", "<"),
                   ("tab:cyan", ">")]


def parse_csv_list(value: str) -> list:
    if value is None or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_csv_list(value: str) -> list:
    return [int(item) for item in parse_csv_list(value)]


def normalize_performance_type(value: str) -> str:
    return str(value).strip().lower().replace("_performance", "").replace("performance", "").strip("_- ")


def parse_desired_latejoin_tuples(value: str) -> list:
    """Parse late-join tuples as '10:10:worst;10:25:best'.

    Also accepts '/', '|', or ',' inside each tuple, for example
    '10,10,worst;25,50,best'. The semicolon separates tuples.
    """
    if value is None or str(value).strip() == "":
        return []
    tuples = []
    for raw_tuple in str(value).split(";"):
        raw_tuple = raw_tuple.strip()
        if not raw_tuple:
            continue
        normalized = raw_tuple.replace("|", ":").replace("/", ":")
        parts = [p.strip() for p in normalized.split(":")]
        if len(parts) == 1 and "," in normalized:
            parts = [p.strip() for p in normalized.split(",")]
        if len(parts) != 3:
            raise ValueError("Invalid late-join tuple '{0}'. Use 'num_late_clients:entry_round:performance_type', "
                             "for example '10:10:worst;10:25:best'.".format(raw_tuple))
        tuples.append((int(parts[0]), int(parts[1]), normalize_performance_type(parts[2])))
    return tuples


def build_latejoin_tuples(desired_latejoin_tuples: str,
                          latejoin_num_clients: str,
                          latejoin_entry_rounds: str,
                          latejoin_performance_types: str) -> list:
    explicit_tuples = parse_desired_latejoin_tuples(desired_latejoin_tuples)
    if explicit_tuples:
        return explicit_tuples
    nums = parse_int_csv_list(latejoin_num_clients)
    rounds = parse_int_csv_list(latejoin_entry_rounds)
    perf_types = [normalize_performance_type(x) for x in parse_csv_list(latejoin_performance_types)]
    if nums or rounds or perf_types:
        if not nums or not rounds or not perf_types:
            raise ValueError("When using --latejoin-num-clients, --latejoin-entry-rounds, or "
                             "--latejoin-performance-types, all three must be provided.")
        return list(product(nums, rounds, perf_types))
    return []


def latejoin_suffix(latejoin_tuple: tuple) -> str:
    if latejoin_tuple is None:
        return ""
    num_late_clients, entry_round, performance_type = latejoin_tuple
    return "_{0}_late_clients_entry_round_{1}_{2}_performance".format(num_late_clients, entry_round, performance_type)


def safe_label_for_tuple(latejoin_tuple: tuple) -> str:
    if latejoin_tuple is None:
        return ""
    num_late_clients, entry_round, performance_type = latejoin_tuple
    return "{0}_late_entry_{1}_{2}".format(num_late_clients, entry_round, performance_type)


def construct_base_path(results_folder: Path,
                        dataset_name: str,
                        dataset_distribution: str,
                        approach: str,
                        exec_id: int,
                        r_suffix: str = "",
                        latejoin_tuple: tuple = None,
                        availability_scenario: str = "") -> Path:
    """Build the path to one experiment execution.

    Supported layouts:
      - Standard:      <approach>_exec_<id>
      - Late joining:  <approach>_<late_tuple>_exec_<id>
      - Intermittent:  <approach>_<scenario>_exec_<id>
    """
    availability_scenario = str(availability_scenario or "").strip()
    if latejoin_tuple is not None and availability_scenario:
        raise ValueError("Use either latejoin_tuple or availability_scenario, not both.")
    if latejoin_tuple is not None:
        base_name = "{0}{1}_exec_{2}{3}".format(approach, latejoin_suffix(latejoin_tuple), exec_id, r_suffix)
    elif availability_scenario:
        base_name = "{0}_{1}_exec_{2}{3}".format(approach, availability_scenario, exec_id, r_suffix)
    else:
        base_name = "{0}_exec_{1}{2}".format(approach, exec_id, r_suffix)
    return results_folder / dataset_name / dataset_distribution / base_name


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
    individual_metrics_history_df = individual_metrics_history_df.copy()
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


def _extract_exec_number(path: Path,
                         prefix: str,
                         suffix: str) -> int:
    """Extract the numeric execution ID from an experiment-folder name."""
    name = path.name
    expected_prefix = prefix + "_exec_"
    if not name.startswith(expected_prefix):
        return 10 ** 18
    tail = name[len(expected_prefix):]
    if suffix and tail.endswith(suffix):
        tail = tail[:-len(suffix)]
    try:
        return int(tail)
    except ValueError:
        return 10 ** 18


def discover_trial_paths(base_path_template: Path,
                         num_trials: int) -> list:
    """Discover up to ``num_trials`` execution folders.

    The function preserves the traditional ``exec_1`` ... ``exec_N`` behavior
    when those folders exist. Otherwise, it discovers numeric IDs such as
    ``exec_5000``, ``exec_5001``, and ``exec_5002``.
    """
    parent = base_path_template.parent
    prefix = base_path_template.name.split("_exec_", 1)[0]
    suffix = base_path_template.name[len("{0}_exec_1".format(prefix)):]
    expected_paths = [parent / "{0}_exec_{1}{2}".format(prefix, i, suffix)
                      for i in range(1, num_trials + 1)]
    if any(path.exists() for path in expected_paths):
        return expected_paths
    discovered = sorted(parent.glob("{0}_exec_*{1}".format(prefix, suffix)),
                        key=lambda path: _extract_exec_number(path, prefix, suffix))
    return discovered[:num_trials]


def load_multiple_trials(base_path_template: Path,
                         num_trials: int,
                         phase: str) -> list:
    dfs = []
    metrics_file = ("individual_fit_metrics_history.csv"
                    if phase == "train" else "individual_evaluate_metrics_history.csv")
    trial_paths = discover_trial_paths(base_path_template, num_trials)
    if not trial_paths:
        print("[WARNING] No trial folders found for template: {0}".format(base_path_template))
        return dfs
    for trial_path in trial_paths:
        metrics_path = trial_path / "output" / metrics_file
        if not metrics_path.exists():
            print("[WARNING] Missing metrics for trial folder {0}: {1}".format(trial_path.name, metrics_path))
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
    std = combined.std(axis=1).fillna(0)
    return DataFrame({"comm_round": combined.index,
                      "mean_accuracy": mean,
                      "std_accuracy": std}).reset_index(drop=True)


def smart_format(value: float) -> str:
    return "{0:.2f}".format(value) if float("{0:.4f}".format(value)) == float("{0:.2f}".format(value)) else "{0:.4f}".format(value)


def annotate(x,
             y,
             annotation_text,
             color):
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


def style_for_approach(key: str,
                       index: int = 0) -> tuple:
    if key in APPROACH_STYLE_MAP:
        return APPROACH_STYLE_MAP[key]
    color, marker = FALLBACK_STYLES[index % len(FALLBACK_STYLES)]
    return color, marker, key.upper()


def compute_summary_for_approach(base_path_template: Path,
                                 num_trials: int,
                                 phase: str) -> dict | None:
    sel_file = "selected_fit_clients_history.csv" if phase == "train" else "selected_evaluate_clients_history.csv"
    met_file = "individual_fit_metrics_history.csv" if phase == "train" else "individual_evaluate_metrics_history.csv"
    ds_col = "ds_train_i" if phase == "train" else "ds_test_i"
    trial_results = []
    trial_paths = discover_trial_paths(base_path_template, num_trials)
    if not trial_paths:
        print("[WARNING] No trial folders found for template: {0}".format(base_path_template))
        return None
    for trial_path in trial_paths:
        sel_path = trial_path / "output" / sel_file
        met_path = trial_path / "output" / met_file
        try:
            df_sel = read_csv(sel_path)
            df_met = read_csv(met_path)
        except Exception:
            print("[WARNING] Missing or unreadable trial folder: {0}".format(trial_path))
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


def collect_mean_std_for_experiment(results_folder: Path,
                                    dataset_name: str,
                                    dataset_distribution: str,
                                    r_suffix: str,
                                    num_trials: int,
                                    phase: str,
                                    target_weighted_mean_testing_accuracy: float,
                                    all_approaches: list,
                                    latejoin_tuple: tuple = None,
                                    availability_scenario: str = "") -> tuple:
    base_paths = {}
    for key in all_approaches:
        base_paths[key] = construct_base_path(results_folder,
                                              dataset_name,
                                              dataset_distribution,
                                              key,
                                              1,
                                              r_suffix,
                                              latejoin_tuple,
                                              availability_scenario)
    mean_std = {}
    for key, path in base_paths.items():
        dfs = load_multiple_trials(path, num_trials, phase)
        ms = aggregate_trials(dfs, target_weighted_mean_testing_accuracy, phase)
        if ms is not None:
            mean_std[key] = ms
    return base_paths, mean_std


def annotate_on_axis(ax,
                     x,
                     y,
                     annotation_text,
                     color):
    ax.text(x, y, annotation_text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
            path_effects=[withStroke(linewidth=2.5, foreground="white")])


def plot_with_shaded_area_on_axis(ax,
                                  mean_std_df: DataFrame,
                                  label: str,
                                  color: str,
                                  seen_x: dict,
                                  marker: str = "o",
                                  markersize: int = 3):
    rounds = mean_std_df["comm_round"]
    mean = mean_std_df["mean_accuracy"]
    std = mean_std_df["std_accuracy"]
    line, = ax.plot(rounds, mean, label=label, color=color, marker=marker, markersize=markersize)
    ax.fill_between(rounds, mean - std, mean + std, alpha=0.2, color=color)
    y0, y1 = ax.get_ylim()
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
    annotate_on_axis(ax, last_x, last_y + dy, text_value, color)
    return line


def format_accuracy_axis(ax,
                         mean_std: dict,
                         phase: str,
                         show_ylabel: bool = True):
    all_x = []
    all_y_low = []
    all_y_high = []
    for df in mean_std.values():
        all_x.extend(df["comm_round"].values)
        all_y_low.extend((df["mean_accuracy"] - df["std_accuracy"]).values)
        all_y_high.extend((df["mean_accuracy"] + df["std_accuracy"]).values)
    if not all_x:
        return
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
    x_ticks = list(range(xmin, xmax + 1, max(x_step, 1)))
    for required in (1, xmax):
        if required not in x_ticks:
            x_ticks.append(required)
    if 0 in x_ticks:
        x_ticks.remove(0)
    x_ticks = sorted(set(x_ticks))
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.set_xticks(x_ticks)
    y_range = ymax - ymin
    if y_range == 0:
        y_range = 0.1
    ymin -= 0.05 * y_range
    ymax += 0.05 * y_range
    ymin = max(0.0, floor(ymin * 10) / 10)
    ymax = min(1.0, ceil(ymax * 10) / 10)
    ax.set_ylim(ymin, ymax)
    y_step = 0.05 if ymax - ymin <= 0.3 else 0.1
    y_ticks = list(arange(ymin, ymax + 1e-9, y_step))
    y_ticks = [t for t in y_ticks if t != 0]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{0:.2f}".format(y).rstrip("0").rstrip(".")))
    ax.set_xlabel("FL Round")
    if show_ylabel:
        ax.set_ylabel("{0} Accuracy".format(phase.capitalize() if phase == "test" else phase.capitalize() + "ing"))
    ax.grid(True)


def process_combined_availability_figure(results_folder: Path,
                                         root_analysis_folder: Path,
                                         experiment_tag: str,
                                         dataset_name: str,
                                         dataset_distribution: str,
                                         r_suffix: str,
                                         num_trials: int,
                                         phase: str,
                                         target_weighted_mean_testing_accuracy: float,
                                         num_available_clients: int,
                                         all_approaches: list,
                                         availability_scenarios: list):
    if not availability_scenarios:
        return
    scenario_data = []
    for scenario in availability_scenarios:
        _, mean_std = collect_mean_std_for_experiment(results_folder,
                                                      dataset_name,
                                                      dataset_distribution,
                                                      r_suffix,
                                                      num_trials,
                                                      phase,
                                                      target_weighted_mean_testing_accuracy,
                                                      all_approaches,
                                                      None,
                                                      scenario)
        if not mean_std:
            print("[WARNING] No usable data for combined scenario {0}.".format(scenario))
            continue
        scenario_data.append((scenario, mean_std))
    if not scenario_data:
        print("[ERROR] No scenario produced usable data. Skipping combined availability figure.")
        return
    ncols = len(scenario_data)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]
    legend_handles = []
    legend_labels = []
    for ax_idx, (ax, (scenario, mean_std)) in enumerate(zip(axes, scenario_data)):
        seen_x = {}
        for idx, key in enumerate(all_approaches):
            if key not in mean_std:
                continue
            color, marker, label = style_for_approach(key, idx)
            line = plot_with_shaded_area_on_axis(ax, mean_std[key], label, color, seen_x, marker, 3)
            if ax_idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)
        ax.set_title(str(scenario).capitalize())
        format_accuracy_axis(ax, mean_std, phase, show_ylabel=(ax_idx == 0))
    n_legend_cols = min(len(legend_labels), max(1, len(legend_labels)))
    fig.legend(legend_handles,
               legend_labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=n_legend_cols,
               frameon=True,
               framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    scenario_suffix = "_" + "_".join([str(s).strip() for s in availability_scenarios if str(s).strip()])
    figure_file = root_analysis_folder.joinpath("{0}/{1}ing_accuracy_{2}_{3}{4}_{5}_clients{6}_combined.pdf"
                                                .format(experiment_tag,
                                                        phase,
                                                        dataset_name,
                                                        dataset_distribution,
                                                        r_suffix,
                                                        num_available_clients,
                                                        scenario_suffix))
    fig.savefig(figure_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved combined figure: {0}".format(figure_file))


def format_latejoin_title(latejoin_tuple: tuple) -> str:
    if latejoin_tuple is None:
        return "Baseline"
    num_late_clients, entry_round, performance_type = latejoin_tuple
    return "{0} late clients, entry round {1}, {2}".format(num_late_clients,
                                                           entry_round,
                                                           str(performance_type).capitalize())


def process_combined_latejoin_figure(results_folder: Path,
                                     root_analysis_folder: Path,
                                     experiment_tag: str,
                                     dataset_name: str,
                                     dataset_distribution: str,
                                     r_suffix: str,
                                     num_trials: int,
                                     phase: str,
                                     target_weighted_mean_testing_accuracy: float,
                                     num_available_clients: int,
                                     all_approaches: list,
                                     latejoin_tuples: list):
    """Create one side-by-side figure comparing multiple late-joining tuples.

    Each subplot contains the same approaches for one tuple, where a tuple is:
      (num_late_clients, entry_round, performance_type)

    Example tuple CLI string:
      --desired-latejoin-tuples "10:10:best;25:25:worst"
    """
    if not latejoin_tuples or latejoin_tuples == [None]:
        return
    tuple_data = []
    for latejoin_tuple in latejoin_tuples:
        _, mean_std = collect_mean_std_for_experiment(results_folder,
                                                      dataset_name,
                                                      dataset_distribution,
                                                      r_suffix,
                                                      num_trials,
                                                      phase,
                                                      target_weighted_mean_testing_accuracy,
                                                      all_approaches,
                                                      latejoin_tuple,
                                                      "")
        if not mean_std:
            print("[WARNING] No usable data for late-join tuple {0}.".format(latejoin_tuple))
            continue
        tuple_data.append((latejoin_tuple, mean_std))
    if not tuple_data:
        print("[ERROR] No late-join tuple produced usable data. Skipping combined late-join figure.")
        return
    ncols = len(tuple_data)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]
    # Keep a single shared legend across subplots. A dict prevents duplicate labels.
    legend_by_label = {}
    for ax_idx, (ax, (latejoin_tuple, mean_std)) in enumerate(zip(axes, tuple_data)):
        seen_x = {}
        for idx, key in enumerate(all_approaches):
            if key not in mean_std:
                continue
            color, marker, label = style_for_approach(key, idx)
            line = plot_with_shaded_area_on_axis(ax,
                                                 mean_std[key],
                                                 label,
                                                 color,
                                                 seen_x,
                                                 marker,
                                                 3)
            if label not in legend_by_label:
                legend_by_label[label] = line
        ax.set_title(format_latejoin_title(latejoin_tuple))
        format_accuracy_axis(ax, mean_std, phase, show_ylabel=(ax_idx == 0))
    legend_labels = list(legend_by_label.keys())
    legend_handles = [legend_by_label[label] for label in legend_labels]
    n_legend_cols = min(len(legend_labels), max(1, len(legend_labels)))
    fig.legend(legend_handles,
               legend_labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=n_legend_cols,
               frameon=True,
               framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    tuple_suffix = "_".join([safe_label_for_tuple(t) for t in latejoin_tuples])
    figure_file = root_analysis_folder.joinpath("{0}/{1}ing_accuracy_{2}_{3}{4}_{5}_clients_{6}_combined_latejoin.pdf"
                                                .format(experiment_tag,
                                                        phase,
                                                        dataset_name,
                                                        dataset_distribution,
                                                        r_suffix,
                                                        num_available_clients,
                                                        tuple_suffix))
    fig.savefig(figure_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved combined late-join figure: {0}".format(figure_file))


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
                              all_approaches: list,
                              latejoin_tuple: tuple = None,
                              availability_scenario: str = ""):
    base_paths = {}
    for key in all_approaches:
        base_paths[key] = construct_base_path(results_folder,
                                              dataset_name,
                                              dataset_distribution,
                                              key,
                                              1,
                                              r_suffix,
                                              latejoin_tuple,
                                              availability_scenario)
    mean_std = {}
    for key, path in base_paths.items():
        dfs = load_multiple_trials(path, num_trials, phase)
        ms = aggregate_trials(dfs, target_weighted_mean_testing_accuracy, phase)
        if ms is not None:
            mean_std[key] = ms
    if not mean_std:
        print("[ERROR] No experiment produced usable data. Skipping plot for tuple {0}, scenario {1}.".format(latejoin_tuple, availability_scenario))
        return
    figure(figsize=(7.2, 4.0))
    seen_x = {}
    for idx, (key, df) in enumerate(mean_std.items()):
        color, marker, label = style_for_approach(key, idx)
        plot_with_shaded_area(df, label, color, seen_x, marker, 3)
    ax = gca()
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
    x_ticks = list(range(xmin, xmax + 1, max(x_step, 1)))
    for required in (1, xmax):
        if required not in x_ticks:
            x_ticks.append(required)
    if 0 in x_ticks:
        x_ticks.remove(0)
    x_ticks = sorted(set(x_ticks))
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.set_xticks(x_ticks)
    y_range = ymax - ymin
    if y_range == 0:
        y_range = 0.1
    ymin -= 0.05 * y_range
    ymax += 0.05 * y_range
    ymin = max(0.0, floor(ymin * 10) / 10)
    ymax = min(1.0, ceil(ymax * 10) / 10)
    ax.set_ylim(ymin, ymax)
    y_step = 0.05 if ymax - ymin <= 0.3 else 0.1
    y_ticks = list(arange(ymin, ymax + 1e-9, y_step))
    y_ticks = [t for t in y_ticks if t != 0]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{0:.2f}".format(y).rstrip("0").rstrip(".")))
    xlabel("FL Round")
    ylabel("{0} Accuracy".format(phase.capitalize() if phase == "test" else phase.capitalize() + "ing"))
    legend(loc="lower right", frameon=True, framealpha=0.9)
    grid(True)
    tight_layout()
    tuple_label = safe_label_for_tuple(latejoin_tuple)
    tuple_suffix = "" if tuple_label == "" else "_{0}".format(tuple_label)
    scenario_label = str(availability_scenario or "").strip()
    scenario_suffix = "" if scenario_label == "" else "_{0}".format(scenario_label)
    figure_file = root_analysis_folder.joinpath("{0}/{1}ing_accuracy_{2}_{3}{4}_{5}_clients{6}{7}.pdf"
                                                .format(experiment_tag,
                                                        phase,
                                                        dataset_name,
                                                        dataset_distribution,
                                                        r_suffix,
                                                        num_available_clients,
                                                        scenario_suffix,
                                                        tuple_suffix))
    savefig(figure_file, dpi=300, bbox_inches="tight")
    print("[INFO] Saved figure: {0}".format(figure_file))
    summary_rows = []
    for idx, key in enumerate(all_approaches):
        path = base_paths.get(key)
        if path is None:
            continue
        summary = compute_summary_for_approach(path, num_trials, phase)
        if summary is None:
            continue
        label = style_for_approach(key, idx)[2]
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
         all_approaches: str,
         desired_latejoin_tuples: str,
         latejoin_num_clients: str,
         latejoin_entry_rounds: str,
         latejoin_performance_types: str,
         availability_scenarios: str,
         combine_availability_scenarios: bool,
         also_save_separate_scenarios: bool,
         combine_latejoin_tuples: bool) -> None:
    experiment_tag_folder = root_analysis_folder.joinpath("{0}".format(experiment_tag))
    experiment_tag_folder.mkdir(parents=True, exist_ok=True)
    specific_performance_results_folder = performance_results_folder.joinpath("{0}_clients".format(num_available_clients))
    all_approaches = parse_csv_list(all_approaches)
    latejoin_tuples = build_latejoin_tuples(desired_latejoin_tuples,
                                            latejoin_num_clients,
                                            latejoin_entry_rounds,
                                            latejoin_performance_types)
    if not latejoin_tuples:
        latejoin_tuples = [None]
    availability_scenarios = parse_csv_list(availability_scenarios)
    if latejoin_tuples != [None] and availability_scenarios:
        raise ValueError("Do not combine late-joining tuple options with --availability-scenarios.")
    if not availability_scenarios:
        availability_scenarios = [""]
    target_testing_accuracies = {"cifar_10": {"iid": 0.75, "non_iid": 0.45},
                                 "fashion_mnist": {"iid": 0.85, "non_iid": 0.70},
                                 "emotion": {"iid": 0.9, "non_iid": 0.8}}
    target_weighted_mean_testing_accuracy = target_testing_accuracies[dataset_name][dataset_distribution]
    print("\n")
    print("- Dataset Name:", dataset_name)
    print("- Dataset Distribution:", dataset_distribution)
    print("- Target Testing Accuracy:", target_weighted_mean_testing_accuracy)
    print("- Phase:", phase)
    print("- Number of Trials:", num_trials)
    print("- Approaches:", ",".join(all_approaches))
    print("- Late-Join Tuples:", latejoin_tuples)
    print("- Availability Scenarios:", availability_scenarios)
    print("\n")
    r_list = [""]
    if combine_latejoin_tuples:
        if latejoin_tuples == [None]:
            raise ValueError("--combine-latejoin-tuples requires --desired-latejoin-tuples or latejoin tuple options.")
        if availability_scenarios != [""]:
            raise ValueError("Do not combine late-joining tuples with --availability-scenarios.")
        for r in r_list:
            r_suffix = "" if r == "" else "_r_{0}".format(r)
            process_combined_latejoin_figure(specific_performance_results_folder,
                                             root_analysis_folder,
                                             experiment_tag,
                                             dataset_name,
                                             dataset_distribution,
                                             r_suffix,
                                             num_trials,
                                             phase,
                                             target_weighted_mean_testing_accuracy,
                                             num_available_clients,
                                             all_approaches,
                                             latejoin_tuples)
        if not also_save_separate_scenarios:
            return
    if combine_availability_scenarios:
        if latejoin_tuples != [None]:
            raise ValueError("Combined availability figures are only supported for intermittent experiments, not late-joining tuples.")
        for r in r_list:
            r_suffix = "" if r == "" else "_r_{0}".format(r)
            process_combined_availability_figure(specific_performance_results_folder,
                                                 root_analysis_folder,
                                                 experiment_tag,
                                                 dataset_name,
                                                 dataset_distribution,
                                                 r_suffix,
                                                 num_trials,
                                                 phase,
                                                 target_weighted_mean_testing_accuracy,
                                                 num_available_clients,
                                                 all_approaches,
                                                 availability_scenarios)
        if not also_save_separate_scenarios:
            return
    for latejoin_tuple in latejoin_tuples:
        for availability_scenario in availability_scenarios:
            for r in r_list:
                r_suffix = "" if r == "" else "_r_{0}".format(r)
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
                                          all_approaches,
                                          latejoin_tuple,
                                          availability_scenario)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--performance-results-folder", type=Path, required=True,
                        help="Relative path to the performance results folder (input)")
    parser.add_argument("--root-analysis-folder", type=Path, required=True,
                        help="Relative path to the root analysis folder (output)")
    parser.add_argument("--experiment-tag", type=str, required=True,
                        help="Unique tag for the experiment (e.g., performance, late_join_clients)")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Name of the dataset")
    parser.add_argument("--dataset-distribution", type=str, required=True,
                        help="Distribution of the dataset")
    parser.add_argument("--num-available-clients", type=int, required=True,
                        help="Number of available clients")
    parser.add_argument("--num-trials", type=int, required=True,
                        help="Number of trials (independent executions)")
    parser.add_argument("--phase", type=str, required=True,
                        help="Phase of interest (train / test)")
    parser.add_argument("--all-approaches", type=str, required=True,
                        help="All approaches (comma-separated). Supports fedcab and rifles.")
    parser.add_argument("--desired-latejoin-tuples", type=str, default="",
                        help="Optional semicolon-separated tuples: num_late_clients:entry_round:performance_type. Example: '10:10:worst;10:25:best'.")
    parser.add_argument("--latejoin-num-clients", type=str, default="",
                        help="Optional comma-separated late-client counts. Used with --latejoin-entry-rounds and --latejoin-performance-types.")
    parser.add_argument("--latejoin-entry-rounds", type=str, default="",
                        help="Optional comma-separated late-client entry rounds. Used with --latejoin-num-clients and --latejoin-performance-types.")
    parser.add_argument("--latejoin-performance-types", type=str, default="",
                        help="Optional comma-separated performance types, e.g., worst,best. Used with --latejoin-num-clients and --latejoin-entry-rounds.")
    parser.add_argument("--availability-scenarios", type=str, default="",
                        help="Optional comma-separated intermittent-availability scenarios, e.g., moderate,severe. "
                             "Uses folders named <approach>_<scenario>_exec_<id>.")
    parser.add_argument("--combine-availability-scenarios", action="store_true",
                        help="Create one side-by-side figure for the listed availability scenarios with a shared legend.")
    parser.add_argument("--also-save-separate-scenarios", action="store_true",
                        help="When --combine-availability-scenarios is used, also save the per-scenario figures.")
    parser.add_argument("--combine-latejoin-tuples", action="store_true",
                        help="Create one side-by-side figure for the listed late-joining tuples with a shared legend.")
    args = parser.parse_args()
    main(args.performance_results_folder,
         args.root_analysis_folder,
         args.experiment_tag,
         args.dataset_name,
         args.dataset_distribution,
         args.num_available_clients,
         args.num_trials,
         args.phase,
         args.all_approaches,
         args.desired_latejoin_tuples,
         args.latejoin_num_clients,
         args.latejoin_entry_rounds,
         args.latejoin_performance_types,
         args.availability_scenarios,
         args.combine_availability_scenarios,
         args.also_save_separate_scenarios,
         args.combine_latejoin_tuples)
