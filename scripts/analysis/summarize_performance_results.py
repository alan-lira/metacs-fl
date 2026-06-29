from argparse import ArgumentParser
from itertools import product
from numpy import array, bincount, sum as np_sum, std, inf
import matplotlib.pyplot as plt
from math import isnan
from pandas import concat, DataFrame, read_csv
from pathlib import Path


APPROACH_LABELS = {"fedavg": "FedAvg",
                   "ecmtc": "ECMTC",
                   "mec": "MEC",
                   "oort": "Oort",
                   "divfl": "DivFL",
                   "ecsm": "ECSM",
                   "fedcab": "FedCAB",
                   "rifles": "RIFLES",
                   "rifles_gh": "RIFLES-GH",
                   "metacsfl": "MetaCS-FL",
                   "metacsfl_no_privacy": "MetaCS-FL No Privacy"}


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


def style_for_approach(key: str,
                       index: int = 0) -> tuple:
    if key in APPROACH_STYLE_MAP:
        return APPROACH_STYLE_MAP[key]
    color, marker = FALLBACK_STYLES[index % len(FALLBACK_STYLES)]
    return color, marker, APPROACH_LABELS.get(key, key.upper())


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


def normalize_availability_scenario(value: str) -> str:
    return str(value).strip().lower().replace("_availability", "").replace("availability", "").strip("_- ")


def build_availability_scenarios(value: str) -> list:
    return [normalize_availability_scenario(item) for item in parse_csv_list(value)]


def build_analysis_groups(latejoin_tuples: list,
                          availability_scenarios: list) -> list:
    """Create mutually exclusive analysis groups.

    Groups are dictionaries so the same summarization/table code can handle:
    - static/default folders: approach_exec_N
    - late-join folders: approach_X_late_clients_entry_round_Y_Z_performance_exec_N
    - intermittent folders: approach_scenario_exec_N, e.g., metacsfl_moderate_exec_5000
    """
    if latejoin_tuples and availability_scenarios:
        raise ValueError("Use either late-joining options or --availability-scenarios, not both in the same run.")
    if availability_scenarios:
        return [{"group_type": "availability", "group_value": scenario} for scenario in availability_scenarios]
    if latejoin_tuples:
        return [{"group_type": "latejoin", "group_value": latejoin_tuple} for latejoin_tuple in latejoin_tuples]
    return [{"group_type": "default", "group_value": None}]


def group_display_terminal(group: dict) -> str:
    if group is None or group.get("group_type") == "default":
        return ""
    if group.get("group_type") == "latejoin":
        return tuple_to_terminal(group.get("group_value"))
    if group.get("group_type") == "availability":
        return str(group.get("group_value")).capitalize()
    return str(group.get("group_value"))


def group_display_latex(group: dict) -> str:
    if group is None or group.get("group_type") == "default":
        return ""
    if group.get("group_type") == "latejoin":
        return tuple_to_latex(group.get("group_value"))
    if group.get("group_type") == "availability":
        return r"\textit{" + latex_escape(str(group.get("group_value")).capitalize()) + r"}"
    return latex_escape(str(group.get("group_value")))


def group_header(group_type: str) -> str:
    if group_type == "latejoin":
        return "Tuple"
    if group_type == "availability":
        return "Availability Scenario"
    return "Group"


def construct_base_path(results_folder: Path,
                        dataset_name: str,
                        dataset_distribution: str,
                        approach: str,
                        exec_id: int,
                        r_suffix: str = "",
                        latejoin_tuple: tuple = None,
                        availability_scenario: str = None) -> Path:
    if latejoin_tuple is not None:
        base_name = "{0}{1}_exec_{2}{3}".format(approach, latejoin_suffix(latejoin_tuple), exec_id, r_suffix)
    elif availability_scenario is not None:
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
    name = path.name
    if not name.startswith(prefix + "_exec_"):
        return 10 ** 18
    tail = name[len(prefix + "_exec_"):]
    if suffix and tail.endswith(suffix):
        tail = tail[:-len(suffix)]
    try:
        return int(tail)
    except ValueError:
        return 10 ** 18


def discover_trial_paths(base_path_template: Path,
                         num_trials: int) -> list:
    """Find trial folders for a template path.

    The old script assumed folders named exec_1, exec_2, ..., exec_N.  This
    version first tries the old convention, but if those folders do not exist
    it automatically discovers folders such as exec_5000, exec_5001, exec_5002.
    """
    parent = base_path_template.parent
    prefix = base_path_template.name.split("_exec_")[0]
    suffix = base_path_template.name[len("{0}_exec_1".format(prefix)):]
    expected_paths = [parent / "{0}_exec_{1}{2}".format(prefix, i, suffix)
                      for i in range(1, num_trials + 1)]
    if any(p.exists() for p in expected_paths):
        return expected_paths
    discovered = sorted(parent.glob("{0}_exec_*{1}".format(prefix, suffix)),
                        key=lambda p: _extract_exec_number(p, prefix, suffix))
    return discovered[:num_trials]


def load_multiple_trials(base_path_template: Path,
                         num_trials: int) -> list:
    trials_data = []
    trial_paths = discover_trial_paths(base_path_template, num_trials)
    if not trial_paths:
        print("[WARNING] No trial folders found for template: {0}".format(base_path_template))
        return trials_data
    for trial_index, trial_path in enumerate(trial_paths, start=1):
        output_path = trial_path / "output"
        fit_path = output_path / "individual_fit_metrics_history.csv"
        eval_path = output_path / "individual_evaluate_metrics_history.csv"
        selected_path = output_path / "selected_fit_clients_history.csv"
        resources_path = trial_path / "clients_resources.csv"
        missing_files = []
        for p in [fit_path, eval_path, selected_path, resources_path]:
            if not p.exists():
                missing_files.append(str(p))
        if missing_files:
            print("[WARNING] Missing files for trial folder {0}:".format(trial_path.name))
            for m in missing_files:
                print("  - {0}".format(m))
            continue
        trial_data = {"individual_fit_metrics_history": load_dataframe(fit_path),
                      "individual_evaluate_metrics_history": load_dataframe(eval_path),
                      "selected_fit_clients_history": load_dataframe(selected_path),
                      "clients_resources": load_dataframe(resources_path),
                      "trial_path": trial_path}
        trials_data.append(trial_data)
    return trials_data


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
    std_values = combined.std(axis=1).fillna(0)
    return DataFrame({"comm_round": combined.index,
                      "mean_accuracy": mean,
                      "std_accuracy": std_values}).reset_index(drop=True)


def calculate_jain_fairness_index(df: DataFrame) -> float:
    candidates_raw = df["available_clients"].tolist()
    selected_raw = df["selected_clients"].tolist()
    candidates_all = []
    for r in candidates_raw:
        candidates_all.extend([int(x.replace("client_", "")) for x in r.split("|")])
    unique_candidates = sorted(set(candidates_all))
    selected_all = []
    for r in selected_raw:
        selected_all.extend([int(x.replace("client_", "")) for x in r.split("|")])
    if not unique_candidates:
        return 0.0
    max_id = max(unique_candidates)
    counts = bincount(selected_all, minlength=max_id + 1)
    rates = []
    for cid in unique_candidates:
        total = candidates_all.count(cid)
        rate = counts[cid] / total if total > 0 else 0
        rates.append(rate)
    rates = array(rates)
    if np_sum(rates ** 2) == 0:
        return 0.0
    return (np_sum(rates) ** 2) / (len(rates) * np_sum(rates ** 2))


def calculate_metrics_summary(selected_df: DataFrame,
                              fit_df: DataFrame,
                              eval_df: DataFrame,
                              resources_df: DataFrame,
                              target_accuracy: float,
                              fl_round_target: int,
                              all_trials_selected_clients: dict,
                              samples_per_task: int = 1) -> dict:
    fl_col = "comm_round"
    time_col = fit_df.columns[fit_df.columns.str.contains("training_time")][0]
    energy_col = fit_df.columns[fit_df.columns.str.contains("training_energy")][0]
    sel_duration_col = selected_df.columns[selected_df.columns.str.contains("selection_duration")][0]
    num_sel_col = selected_df.columns[selected_df.columns.str.contains("num_selected_clients")][0]
    num_examples_col = fit_df.columns[fit_df.columns.str.contains("examples")][0]
    fit_df = fit_df[fit_df[fl_col] <= fl_round_target]
    selected_df = selected_df[selected_df[fl_col] <= fl_round_target]
    failing_clients_per_round = []
    failing_tasks_per_round = []
    scheduled_samples_per_round = []
    completed_examples_per_round = fit_df.groupby(fl_col)[num_examples_col].sum().to_dict()
    completed_clients_per_round = fit_df.groupby(fl_col)["client_id"].apply(lambda values: set(str(v) for v in values)).to_dict()
    for _, sel_row in selected_df.iterrows():
        round_id = sel_row[fl_col]
        selected_clients_raw = str(sel_row.get("selected_clients", ""))
        if selected_clients_raw.strip():
            selected_clients = set(x.strip() for x in selected_clients_raw.split("|") if x.strip())
        else:
            selected_clients = set()
        completed_clients = completed_clients_per_round.get(round_id, set())
        num_failing_clients = max(0, len(selected_clients - completed_clients))
        scheduled_tasks = float(sel_row.get("num_tasks", 0))
        scheduled_samples = scheduled_tasks * int(samples_per_task)
        completed_samples = float(completed_examples_per_round.get(round_id, 0))
        num_failing_samples = max(0.0, scheduled_samples - completed_samples)
        failing_clients_per_round.append(num_failing_clients)
        failing_tasks_per_round.append(num_failing_samples)
        scheduled_samples_per_round.append(scheduled_samples)
    mean_failing_clients = (sum(failing_clients_per_round) / len(failing_clients_per_round)
                            if failing_clients_per_round else 0.0)
    mean_failing_tasks = (sum(failing_tasks_per_round) / len(failing_tasks_per_round)
                          if failing_tasks_per_round else 0.0)
    mean_scheduled_samples = (sum(scheduled_samples_per_round) / len(scheduled_samples_per_round)
                              if scheduled_samples_per_round else 0.0)
    mean_selected_clients = selected_df[num_sel_col].mean()
    failing_clients_per_selected_client_pct = (100.0 * mean_failing_clients / mean_selected_clients
                                               if mean_selected_clients > 0 else 0.0)
    failing_samples_per_selected_client = (mean_failing_tasks / mean_selected_clients
                                           if mean_selected_clients > 0 else 0.0)
    failing_samples_per_failing_client = (mean_failing_tasks / mean_failing_clients
                                          if mean_failing_clients > 0 else 0.0)
    failing_samples_share_scheduled_pct = (100.0 * mean_failing_tasks / mean_scheduled_samples
                                           if mean_scheduled_samples > 0 else 0.0)
    mean_examples_per_round = fit_df.groupby(fl_col)[num_examples_col].mean()
    mean_examples_scheduled = mean_examples_per_round.mean()
    std_examples_scheduled = mean_examples_per_round.std()
    tasks_per_client = fit_df.groupby("client_id")[num_examples_col].sum()
    cv_tasks = tasks_per_client.std() / tasks_per_client.mean() if tasks_per_client.mean() != 0 else 0
    metrics = {"fl_round_@_target_accuracy": fl_round_target,
               "target_accuracy": target_accuracy,
               "max_training_time_across_all_rounds_in_seconds": fit_df.groupby(fl_col)[time_col].max().max(),
               "total_training_time_in_seconds": fit_df.groupby(fl_col)[time_col].max().sum(),
               "total_training_energy_in_joules": fit_df[energy_col].sum(),
               "min_number_selected_clients_training": selected_df[num_sel_col].min(),
               "max_number_selected_clients_training": selected_df[num_sel_col].max(),
               "mean_number_selected_clients_training": mean_selected_clients,
               "mean_scheduled_samples_training": mean_scheduled_samples,
               "mean_number_failing_clients_training": mean_failing_clients,
               "mean_number_failing_tasks_training": mean_failing_tasks,
               "mean_number_failing_samples_training": mean_failing_tasks,
               "failing_clients_per_selected_client_training_pct": failing_clients_per_selected_client_pct,
               "failing_samples_per_selected_client_training": failing_samples_per_selected_client,
               "failing_samples_per_failing_client_training": failing_samples_per_failing_client,
               "failing_samples_share_scheduled_training_pct": failing_samples_share_scheduled_pct,
               "jain_fairness_index_training": calculate_jain_fairness_index(selected_df),
               "mean_examples_per_selected_client_training": mean_examples_scheduled,
               "std_examples_per_selected_client_training": std_examples_scheduled,
               "cv_tasks_per_client_training": cv_tasks}
    all_trials_selected_clients["min"] = min(all_trials_selected_clients["min"],
                                             metrics["min_number_selected_clients_training"])
    all_trials_selected_clients["max"] = max(all_trials_selected_clients["max"],
                                             metrics["max_number_selected_clients_training"])
    idle_sel_energy = 0
    idle_train_energy = 0
    total_selection_duration = 0
    for r in range(1, fl_round_target + 1):
        selected_round_df = selected_df[selected_df[fl_col] == r]
        if selected_round_df.empty:
            continue
        sel_row = selected_round_df.iloc[0]
        duration = sel_row[sel_duration_col]
        total_selection_duration += duration
        candidates = list(map(int, sel_row["available_clients"].replace("client_", "").split("|")))
        selected = list(map(int, sel_row["selected_clients"].replace("client_", "").split("|")))
        for cid in candidates:
            idle_power_series = resources_df[resources_df["client_id"] == cid]["mean_power_consumption_idle_in_watts"]
            if idle_power_series.empty:
                continue
            idle_sel_energy += duration * idle_power_series.iloc[0]
        round_fit = fit_df[fit_df[fl_col] == r]
        if round_fit.empty:
            continue
        makespan = round_fit[time_col].max()
        for cid in set(candidates) - set(selected):
            idle_power_series = resources_df[resources_df["client_id"] == cid]["mean_power_consumption_idle_in_watts"]
            if idle_power_series.empty:
                continue
            idle_train_energy += makespan * idle_power_series.iloc[0]
    metrics.update({"total_selection_duration_in_seconds": total_selection_duration,
                    "total_idle_energy_during_selection_in_joules": idle_sel_energy,
                    "total_idle_energy_during_training_in_joules": idle_train_energy,
                    "total_training+idle_energy_during_training_in_joules": metrics["total_training_energy_in_joules"] + idle_train_energy})
    return metrics


def smart_format(value: float) -> str:
    return ("{0:.2f}".format(value) if float("{0:.4f}".format(value)) == float("{0:.2f}".format(value)) else "{0:.4f}".format(value))


def latex_escape(value: str) -> str:
    return (str(value)
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&"))


def dataset_display_name(dataset_name: str) -> str:
    mapping = {"cifar_10": "CIFAR-10",
               "fashion_mnist": "Fashion-MNIST",
               "emotion": "Emotion"}
    return mapping.get(dataset_name, dataset_name.replace("_", "-").title())


def infer_samples_per_task(dataset_name: str,
                           user_value: int = None) -> int:
    """Return how many training samples one scheduled task represents.

    Image datasets use one sample per task in the current experiments.
    Emotion uses 10 text samples per scheduled task. The user can override
    this with --samples-per-task if a future dataset/configuration differs.
    """
    if user_value is not None and int(user_value) > 0:
        return int(user_value)
    mapping = {"cifar_10": 1,
               "fashion_mnist": 1,
               "emotion": 10}
    return mapping.get(dataset_name, 1)


def normalize_client_id_value(value) -> int:
    """Normalize client identifiers such as 78, '78', or 'client_78' to int."""
    text = str(value).strip()
    if text.startswith("client_"):
        text = text.replace("client_", "", 1)
    return int(text)


def parse_client_id_set(raw_value) -> set:
    """Parse a pipe-separated client list from selected_fit_clients_history.csv."""
    raw_text = "" if raw_value is None else str(raw_value).strip()
    if not raw_text or raw_text.lower() == "nan":
        return set()
    result = set()
    for token in raw_text.split("|"):
        token = token.strip()
        if not token:
            continue
        result.add(normalize_client_id_value(token))
    return result


def find_late_joining_clients_file(trial_path: Path) -> Path:
    """Find late_joining_clients_ids.csv robustly across likely locations."""
    candidates = [trial_path / "late_joining_clients_ids.csv",
                  trial_path / "output" / "late_joining_clients_ids.csv",
                  trial_path.parent / "late_joining_clients_ids.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    discovered = list(trial_path.rglob("late_joining_clients_ids.csv"))
    if discovered:
        return discovered[0]
    return None


def load_late_joining_clients_df(trial_path: Path) -> DataFrame:
    late_join_file = find_late_joining_clients_file(trial_path)
    if late_join_file is None:
        print("[WARNING] Missing late_joining_clients_ids.csv for trial folder: {0}".format(trial_path))
        return None
    return read_csv(late_join_file)


def calculate_latejoin_engagement_trial_metrics(selected_df: DataFrame,
                                                fit_df: DataFrame,
                                                latejoin_df: DataFrame,
                                                fl_round_target: int,
                                                samples_per_task: int,
                                                latejoin_samples_mode: str) -> dict:
    """Compute post-entry workload redistribution between initial and late clients.

    The metrics are computed only for rounds after the late-joining clients have
    become available and only up to the round where the approach reaches the
    target accuracy.  This makes the table answer: after late clients can be
    selected, how much of the selected-client set and sample workload is assigned
    to late clients versus the clients that were already present initially?

    Returned metrics include absolute means, late-client shares of the post-entry
    selected/workload totals, and late-vs-initial ratios.  The latter are useful
    in CSVs; the table uses the shares because they are easier to read as a
    redistribution/composition measure.
    """
    zero_metrics = {"mean_initial_selected_clients_training": 0.0,
                    "mean_latejoin_selected_clients_training": 0.0,
                    "mean_total_postentry_selected_clients_training": 0.0,
                    "latejoin_selected_clients_share_training_pct": 0.0,
                    "initial_selected_clients_share_training_pct": 0.0,
                    "latejoin_selected_clients_pct_vs_initial_training": 0.0,
                    "mean_initial_samples_training": 0.0,
                    "mean_latejoin_samples_training": 0.0,
                    "mean_total_postentry_samples_training": 0.0,
                    "latejoin_samples_share_training_pct": 0.0,
                    "initial_samples_share_training_pct": 0.0,
                    "latejoin_samples_pct_vs_initial_training": 0.0}
    if latejoin_df is None or latejoin_df.empty:
        return zero_metrics.copy()
    if "client_id" not in latejoin_df.columns:
        print("[WARNING] late_joining_clients_ids.csv does not contain client_id.")
        return zero_metrics.copy()
    if "round_of_first_appearance" in latejoin_df.columns:
        entries = {normalize_client_id_value(row["client_id"]): int(row["round_of_first_appearance"])
                   for _, row in latejoin_df.iterrows()}
    else:
        entries = {normalize_client_id_value(cid): 1 for cid in latejoin_df["client_id"].tolist()}
    if not entries:
        return zero_metrics.copy()
    fl_col = "comm_round"
    selected_df = selected_df[selected_df[fl_col] <= fl_round_target].copy()
    fit_df = fit_df[fit_df[fl_col] <= fl_round_target].copy()
    # Prefer ds_train_i, because it explicitly corresponds to the training
    # sample count used by each client. Fall back to any examples column.
    if "ds_train_i" in fit_df.columns:
        sample_col = "ds_train_i"
    else:
        sample_col = fit_df.columns[fit_df.columns.str.contains("examples")][0]
    if "client_id" in fit_df.columns:
        fit_df["_client_id_int"] = fit_df["client_id"].apply(normalize_client_id_value)
    else:
        print("[WARNING] individual_fit_metrics_history.csv does not contain client_id.")
        return zero_metrics.copy()
    late_selected_counts = []
    initial_selected_counts = []
    late_sample_values = []
    initial_sample_values = []
    min_entry_round = min(entries.values())
    late_client_ids = set(entries.keys())
    for _, sel_row in selected_df.iterrows():
        round_id = int(sel_row[fl_col])
        if round_id < min_entry_round:
            continue
        eligible_late_ids = {cid for cid, entry_round in entries.items()
                             if round_id >= int(entry_round)}
        if not eligible_late_ids:
            continue
        selected_ids = parse_client_id_set(sel_row.get("selected_clients", ""))
        selected_late_ids = selected_ids.intersection(eligible_late_ids)
        selected_initial_ids = selected_ids.difference(late_client_ids)
        late_selected_counts.append(float(len(selected_late_ids)))
        initial_selected_counts.append(float(len(selected_initial_ids)))
        if latejoin_samples_mode == "proportional_scheduled":
            selected_count = len(selected_ids)
            scheduled_round_samples = float(sel_row.get("num_tasks", 0)) * int(samples_per_task)
            late_samples = (scheduled_round_samples * (len(selected_late_ids) / selected_count)
                            if selected_count > 0 else 0.0)
            initial_samples = (scheduled_round_samples * (len(selected_initial_ids) / selected_count)
                               if selected_count > 0 else 0.0)
        else:
            round_fit = fit_df[fit_df[fl_col] == round_id]
            late_samples = round_fit[round_fit["_client_id_int"].isin(selected_late_ids)][sample_col].sum()
            initial_samples = round_fit[round_fit["_client_id_int"].isin(selected_initial_ids)][sample_col].sum()
        late_sample_values.append(float(late_samples))
        initial_sample_values.append(float(initial_samples))
    if not late_selected_counts:
        return zero_metrics.copy()
    mean_late_selected = sum(late_selected_counts) / len(late_selected_counts)
    mean_initial_selected = sum(initial_selected_counts) / len(initial_selected_counts)
    mean_total_selected = mean_late_selected + mean_initial_selected
    mean_late_samples = sum(late_sample_values) / len(late_sample_values)
    mean_initial_samples = sum(initial_sample_values) / len(initial_sample_values)
    mean_total_samples = mean_late_samples + mean_initial_samples
    return {"mean_initial_selected_clients_training": mean_initial_selected,
            "mean_latejoin_selected_clients_training": mean_late_selected,
            "mean_total_postentry_selected_clients_training": mean_total_selected,
            "latejoin_selected_clients_share_training_pct": (100.0 * mean_late_selected / mean_total_selected
                                                             if mean_total_selected > 0 else 0.0),
            "initial_selected_clients_share_training_pct": (100.0 * mean_initial_selected / mean_total_selected
                                                            if mean_total_selected > 0 else 0.0),
            "latejoin_selected_clients_pct_vs_initial_training": (100.0 * mean_late_selected / mean_initial_selected
                                                                  if mean_initial_selected > 0 else 0.0),
            "mean_initial_samples_training": mean_initial_samples,
            "mean_latejoin_samples_training": mean_late_samples,
            "mean_total_postentry_samples_training": mean_total_samples,
            "latejoin_samples_share_training_pct": (100.0 * mean_late_samples / mean_total_samples
                                                    if mean_total_samples > 0 else 0.0),
            "initial_samples_share_training_pct": (100.0 * mean_initial_samples / mean_total_samples
                                                   if mean_total_samples > 0 else 0.0),
            "latejoin_samples_pct_vs_initial_training": (100.0 * mean_late_samples / mean_initial_samples
                                                         if mean_initial_samples > 0 else 0.0)}


def distribution_display_name(dataset_distribution: str) -> str:
    mapping = {"iid": "IID", "non_iid": "non-IID", "non-iid": "non-IID"}
    return mapping.get(dataset_distribution, dataset_distribution.replace("_", "-"))


def latex_label_fragment(value: str) -> str:
    return str(value).lower().replace("-", "_").replace(" ", "_")


def format_latex_number(value: float,
                        digits: int = 2,
                        comma: bool = True) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "--"
    if isnan(v):
        return "--"
    if comma:
        return ("{0:,." + str(digits) + "f}").format(v)
    return ("{0:." + str(digits) + "f}").format(v)


def format_latex_pct(value: float) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    sign = "+" if v >= 0 else r"$-$"
    return "{0}{1}\\%".format(sign, format_latex_number(abs(v), 2, comma=False))


def latex_metric_cell(value: float,
                      ref_value: float = None,
                      digits: int = 2,
                      comma: bool = True,
                      include_delta: bool = True) -> str:
    value_s = format_latex_number(value, digits=digits, comma=comma)
    if (not include_delta) or ref_value is None or float(ref_value) == 0:
        return value_s
    pct = ((float(value) - float(ref_value)) / abs(float(ref_value))) * 100
    return r"\begin{tabular}[c]{@{}c@{}}" + value_s + r"\\(" + format_latex_pct(pct) + r")\end{tabular}"


def latex_roa_cell(mean_value: float,
                   std_value: float) -> str:
    return "{0} $\\pm$ {1}".format(format_latex_number(mean_value, 2, comma=False),
                                     format_latex_number(std_value, 2, comma=False))


def format_terminal_number(value: float,
                           digits: int = 2,
                           comma: bool = True) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "--"
    if isnan(v):
        return "--"
    if comma:
        return ("{0:,." + str(digits) + "f}").format(v)
    return ("{0:." + str(digits) + "f}").format(v)


def format_terminal_pct(value: float) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    sign = "+" if v >= 0 else "-"
    return "{0}{1}%".format(sign, format_terminal_number(abs(v), 2, comma=False))


def terminal_metric_cell(value: float,
                         ref_value: float = None,
                         digits: int = 2,
                         comma: bool = True,
                         include_delta: bool = True) -> str:
    value_s = format_terminal_number(value, digits=digits, comma=comma)
    if (not include_delta) or ref_value is None or float(ref_value) == 0:
        return value_s
    pct = ((float(value) - float(ref_value)) / abs(float(ref_value))) * 100
    return "{0}\n({1})".format(value_s, format_terminal_pct(pct))


def terminal_roa_cell(mean_value: float,
                      std_value: float) -> str:
    return "{0} ± {1}".format(format_terminal_number(mean_value, 2, comma=False),
                              format_terminal_number(std_value, 2, comma=False))


def tuple_to_terminal(latejoin_tuple: tuple) -> str:
    if latejoin_tuple is None:
        return ""
    num_late_clients, entry_round, performance_type = latejoin_tuple
    return "({0},{1},{2})".format(num_late_clients, entry_round, performance_type)


def make_terminal_approach_label(approach: str,
                                 baseline: str,
                                 mark_baseline: bool) -> str:
    label = APPROACH_LABELS.get(approach, approach.upper())
    if mark_baseline and approach == baseline:
        return "{0}\n(Baseline)".format(label)
    return label


def _split_cell(value) -> list:
    text = "" if value is None else str(value)
    return text.split("\n") if text else [""]


def _table_border(widths: list,
                  left: str,
                  mid: str,
                  right: str,
                  fill: str) -> str:
    return left + mid.join(fill * (w + 2) for w in widths) + right


def _render_terminal_grid(headers: list,
                          rows: list,
                          group_breaks: set = None) -> str:
    """Render a compact Unicode grid table with support for multi-line cells."""
    group_breaks = group_breaks or set()
    all_rows = [headers] + rows
    widths = []
    for col_idx in range(len(headers)):
        max_width = 0
        for row in all_rows:
            for line in _split_cell(row[col_idx]):
                max_width = max(max_width, len(line))
        widths.append(max_width)
    top = _table_border(widths, "┌", "┬", "┐", "─")
    header_sep = _table_border(widths, "├", "┼", "┤", "─")
    row_sep = _table_border(widths, "├", "┼", "┤", "─")
    bottom = _table_border(widths, "└", "┴", "┘", "─")
    def render_row(row):
        cell_lines = [_split_cell(cell) for cell in row]
        height = max(len(lines) for lines in cell_lines)
        rendered = []
        for line_idx in range(height):
            parts = []
            for col_idx, lines in enumerate(cell_lines):
                line = lines[line_idx] if line_idx < len(lines) else ""
                parts.append(" " + line.ljust(widths[col_idx]) + " ")
            rendered.append("│" + "│".join(parts) + "│")
        return rendered
    lines = [top]
    lines.extend(render_row(headers))
    lines.append(header_sep)
    for row_idx, row in enumerate(rows):
        if row_idx in group_breaks and row_idx != 0:
            lines.append(row_sep)
        lines.extend(render_row(row))
    lines.append(bottom)
    return "\n".join(lines)


def build_terminal_table(summary_results: list,
                         all_approaches: list,
                         baseline: str,
                         target_accuracy: float,
                         mark_baseline: bool,
                         energy_metric: str) -> str:
    has_grouped_results = any(item.get("group_type") != "default" for item in summary_results)
    is_availability_table = bool(summary_results) and summary_results[0].get("group_type") == "availability"
    target_s = format_terminal_number(target_accuracy, 2, comma=False).rstrip("0").rstrip(".")
    energy_key = ("total_training+idle_energy_during_training_in_joules"
                  if energy_metric == "training_idle"
                  else "total_training_energy_in_joules")
    headers = []
    if has_grouped_results:
        headers.append(group_header(summary_results[0].get("group_type")))
    headers.extend(["CS Approach", "M_total (s)", "Sigma_total (kJ)", "T_cs (s)", "S_fair"])
    if is_availability_table:
        headers.extend(["n_fail", "s_fail"])
    headers.append("RoA@{0}".format(target_s))
    rows = []
    group_breaks = set()
    for tuple_index, item in enumerate(summary_results):
        means_all = item["means"]
        stds_all = item["stds"]
        group = {"group_type": item.get("group_type"), "group_value": item.get("group_value")}
        ref = means_all.get(baseline)
        available_approaches = [a for a in all_approaches if a in means_all]
        if not available_approaches:
            continue
        if rows:
            group_breaks.add(len(rows))
        for approach_index, approach in enumerate(available_approaches):
            means = means_all[approach]
            stds = stds_all[approach]
            ref_metrics = ref if ref is not None and approach != baseline else None
            row = []
            if has_grouped_results:
                row.append(group_display_terminal(group) if approach_index == 0 else "")
            row.extend([
                make_terminal_approach_label(approach, baseline, mark_baseline),
                terminal_metric_cell(means["total_training_time_in_seconds"],
                                     None if ref_metrics is None else ref_metrics["total_training_time_in_seconds"],
                                     digits=2, comma=True),
                terminal_metric_cell(means[energy_key] / 1000.0,
                                     None if ref_metrics is None else ref_metrics[energy_key] / 1000.0,
                                     digits=2, comma=True),
                terminal_metric_cell(means["total_selection_duration_in_seconds"],
                                     None if ref_metrics is None else ref_metrics["total_selection_duration_in_seconds"],
                                     digits=2, comma=True),
                terminal_metric_cell(means["jain_fairness_index_training"],
                                     None if ref_metrics is None else ref_metrics["jain_fairness_index_training"],
                                     digits=2, comma=False),
            ])
            if is_availability_table:
                row.extend([
                    terminal_metric_cell(means["mean_number_failing_clients_training"],
                                         None if ref_metrics is None else ref_metrics["mean_number_failing_clients_training"],
                                         digits=2, comma=False),
                    terminal_metric_cell(means["mean_number_failing_tasks_training"],
                                         None if ref_metrics is None else ref_metrics["mean_number_failing_tasks_training"],
                                         digits=2, comma=True),
                ])
            row.append(terminal_roa_cell(means["fl_round_@_target_accuracy"],
                                         stds["fl_round_@_target_accuracy"]))
            rows.append(row)
    if not rows:
        return "[WARNING] No usable rows found for the terminal table."
    table = _render_terminal_grid(headers, rows, group_breaks)
    units = "Units: M_total and T_cs in seconds; Sigma_total in kilojoules."
    if is_availability_table:
        units += " n_fail is the mean number of selected clients that failed per training round; s_fail is the mean number of scheduled training samples not completed per training round."
    units += " Values in parentheses are percentage changes vs {0}.".format(APPROACH_LABELS.get(baseline, baseline.upper()))
    return table + "\n" + units


def tuple_to_latex(latejoin_tuple: tuple) -> str:
    if latejoin_tuple is None:
        return ""
    num_late_clients, entry_round, performance_type = latejoin_tuple
    return r"$({0},{1},\textit{{{2}}})$".format(num_late_clients, entry_round, performance_type)


def make_latex_approach_label(approach: str,
                              baseline: str,
                              mark_baseline: bool) -> str:
    label = APPROACH_LABELS.get(approach, approach.upper())
    if mark_baseline and approach == baseline:
        return r"\begin{tabular}[c]{@{}c@{}}" + latex_escape(label) + r"\\(Baseline)\end{tabular}"
    return r"\begin{tabular}[c]{@{}c@{}}" + latex_escape(label) + r"\end{tabular}"


def has_availability_groups(summary_results: list) -> bool:
    return bool(summary_results) and any(item.get("group_type") == "availability" for item in summary_results)


def build_dropout_terminal_table(summary_results: list,
                                 all_approaches: list,
                                 baseline: str,
                                 mark_baseline: bool) -> str:
    """Build a compact table focused only on intermittent dropout behavior.

    Columns:
    - availability scenario
    - approach
    - mean number of selected clients per training round
    - mean number of failed selected clients per training round
    - failed selected clients normalized by selected clients
    - mean number of failed scheduled training samples per training round
    - failed scheduled samples normalized by failing selected clients
    - failed scheduled samples normalized by all scheduled samples
    """
    availability_results = [item for item in summary_results if item.get("group_type") == "availability"]
    if not availability_results:
        return "[INFO] Dropout table skipped because no availability scenarios were analyzed."
    headers = ["Availability Scenario",
               "CS Approach",
               "mean selected clients",
               "n_fail",
               "n_fail/selected (%)",
               "s_fail",
               "s_fail/n_fail",
               "s_fail/scheduled (%)"]
    rows = []
    group_breaks = set()
    for item in availability_results:
        means_all = item["means"]
        group = {"group_type": item.get("group_type"), "group_value": item.get("group_value")}
        ref = means_all.get(baseline)
        available_approaches = [a for a in all_approaches if a in means_all]
        if not available_approaches:
            continue
        if rows:
            group_breaks.add(len(rows))
        for approach_index, approach in enumerate(available_approaches):
            means = means_all[approach]
            ref_metrics = ref if ref is not None and approach != baseline else None
            rows.append([
                group_display_terminal(group) if approach_index == 0 else "",
                make_terminal_approach_label(approach, baseline, mark_baseline),
                terminal_metric_cell(means["mean_number_selected_clients_training"],
                                     None if ref_metrics is None else ref_metrics["mean_number_selected_clients_training"],
                                     digits=2,
                                     comma=False),
                terminal_metric_cell(means["mean_number_failing_clients_training"],
                                     None if ref_metrics is None else ref_metrics["mean_number_failing_clients_training"],
                                     digits=2,
                                     comma=False),
                terminal_metric_cell(means["failing_clients_per_selected_client_training_pct"],
                                     None if ref_metrics is None else ref_metrics["failing_clients_per_selected_client_training_pct"],
                                     digits=2,
                                     comma=False),
                terminal_metric_cell(means["mean_number_failing_samples_training"],
                                     None if ref_metrics is None else ref_metrics["mean_number_failing_samples_training"],
                                     digits=2,
                                     comma=True),
                terminal_metric_cell(means["failing_samples_per_failing_client_training"],
                                     None if ref_metrics is None else ref_metrics["failing_samples_per_failing_client_training"],
                                     digits=2,
                                     comma=True),
                terminal_metric_cell(means["failing_samples_share_scheduled_training_pct"],
                                     None if ref_metrics is None else ref_metrics["failing_samples_share_scheduled_training_pct"],
                                     digits=2,
                                     comma=False),
            ])
    if not rows:
        return "[WARNING] No usable rows found for the dropout terminal table."
    table = _render_terminal_grid(headers, rows, group_breaks)
    units = ("Failure-impact training table. mean selected clients is the mean number of selected clients per training round; "
             "n_fail is failing selected clients per round; n_fail/selected normalizes failures by selected-client exposure; "
             "s_fail is scheduled training samples not completed per round; s_fail/n_fail normalizes lost workload by the number of failing selected clients; "
             "s_fail/scheduled is the share of scheduled samples not completed. "
             "Values in parentheses are percentage changes vs {0}.").format(APPROACH_LABELS.get(baseline, baseline.upper()))
    return table + "\n" + units


def build_dropout_latex_table(summary_results: list,
                              all_approaches: list,
                              baseline: str,
                              dataset_name: str,
                              dataset_distribution: str,
                              num_available_clients: int,
                              latex_resize_width: str,
                              mark_baseline: bool,
                              latex_caption: str = "",
                              latex_label: str = "") -> str:
    """Build a copy-ready LaTeX table focused on intermittent dropout behavior."""
    availability_results = [item for item in summary_results if item.get("group_type") == "availability"]
    if not availability_results:
        return ""
    dataset_label = dataset_display_name(dataset_name)
    distribution_label = distribution_display_name(dataset_distribution)
    if not latex_caption:
        latex_caption = "Training failure-impact behavior for {0} {1} under intermittent availability.".format(dataset_label, distribution_label)
    if not latex_label:
        latex_label = "tab:intermittent_dropout_{0}_{1}".format(latex_label_fragment(dataset_name),
                                                                 latex_label_fragment(dataset_distribution))
    lines = []
    lines.append("% {0} {1} intermittent dropout results {2}-clients system.".format(dataset_label, distribution_label, num_available_clients))
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"    \setlength{\tabcolsep}{2pt}")
    lines.append(r"    \centering")
    lines.append(r"    \caption{" + latex_escape(latex_caption) + "}")
    lines.append(r"    \label{" + latex_label + "}")
    lines.append(r"    \resizebox{" + latex_resize_width + r"}{!}{")
    lines.append(r"        \begin{tabular}{cccccccc}")
    lines.append(r"            \toprule")
    lines.append(r"            \textbf{Availability Scenario} &")
    lines.append(r"            \textbf{\begin{tabular}[c]{@{}c@{}}CS\\Approach\end{tabular}} &")
    lines.append(r"            \textbf{\begin{tabular}[c]{@{}c@{}}Selected\\Clients\end{tabular}} &")
    lines.append(r"            \textbf{$\bar{n}_{\text{fail}}$} &")
    lines.append(r"            \textbf{\begin{tabular}[c]{@{}c@{}}$\bar{n}_{\text{fail}}/\bar{n}_{\text{sel}}$\\(\%)\end{tabular}} &")
    lines.append(r"            \textbf{$\bar{s}_{\text{fail}}$} &")
    lines.append(r"            \textbf{$\bar{s}_{\text{fail}}/\bar{n}_{\text{fail}}$} &")
    lines.append("            \\textbf{\\begin{tabular}[c]{@{}c@{}}$\\bar{s}_{\\text{fail}}/\\bar{s}_{\\text{sched}}$\\\\(\\%)\\end{tabular}} \\\\")
    lines.append(r"            \midrule")
    lines.append("")
    for group_index, item in enumerate(availability_results):
        means_all = item["means"]
        group = {"group_type": item.get("group_type"), "group_value": item.get("group_value")}
        ref = means_all.get(baseline)
        available_approaches = [a for a in all_approaches if a in means_all]
        if not available_approaches:
            continue
        for approach_index, approach in enumerate(available_approaches):
            means = means_all[approach]
            ref_metrics = ref if ref is not None and approach != baseline else None
            cells = []
            if approach_index == 0:
                cells.append(r"            \multirow{" + str(len(available_approaches)) + r"}{*}{" + group_display_latex(group) + r"}")
            else:
                cells.append(r"            ")
            cells.append(make_latex_approach_label(approach, baseline, mark_baseline))
            cells.append(latex_metric_cell(means["mean_number_selected_clients_training"],
                                           None if ref_metrics is None else ref_metrics["mean_number_selected_clients_training"],
                                           digits=2,
                                           comma=False))
            cells.append(latex_metric_cell(means["mean_number_failing_clients_training"],
                                           None if ref_metrics is None else ref_metrics["mean_number_failing_clients_training"],
                                           digits=2,
                                           comma=False))
            cells.append(latex_metric_cell(means["failing_clients_per_selected_client_training_pct"],
                                           None if ref_metrics is None else ref_metrics["failing_clients_per_selected_client_training_pct"],
                                           digits=2,
                                           comma=False))
            cells.append(latex_metric_cell(means["mean_number_failing_samples_training"],
                                           None if ref_metrics is None else ref_metrics["mean_number_failing_samples_training"],
                                           digits=2,
                                           comma=True))
            cells.append(latex_metric_cell(means["failing_samples_per_failing_client_training"],
                                           None if ref_metrics is None else ref_metrics["failing_samples_per_failing_client_training"],
                                           digits=2,
                                           comma=True))
            cells.append(latex_metric_cell(means["failing_samples_share_scheduled_training_pct"],
                                           None if ref_metrics is None else ref_metrics["failing_samples_share_scheduled_training_pct"],
                                           digits=2,
                                           comma=False))
            lines.append(" & ".join(cells) + r" \\")
        if group_index < len(availability_results) - 1:
            lines.append(r"            \arrayrulecolor{lightgray}\specialrule{0.5pt}{0pt}{0pt}")
            lines.append("")
    lines.append(r"            \bottomrule")
    lines.append(r"        \end{tabular}")
    lines.append(r"    }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def has_latejoin_groups(summary_results: list) -> bool:
    return bool(summary_results) and any(item.get("group_type") == "latejoin" for item in summary_results)


def terminal_value_share_cell(value: float,
                              share_pct: float,
                              digits: int = 2,
                              comma: bool = True) -> str:
    """Format a value with its within-approach post-entry share."""
    return "{0}\n({1}%)".format(format_terminal_number(value, digits=digits, comma=comma),
                                format_terminal_number(share_pct, digits=2, comma=False))


def latex_value_share_cell(value: float,
                           share_pct: float,
                           digits: int = 2,
                           comma: bool = True) -> str:
    """Format a value with its within-approach post-entry share."""
    value_s = format_latex_number(value, digits=digits, comma=comma)
    pct_s = format_latex_number(share_pct, digits=2, comma=False)
    return (r"\begin{tabular}[c]{@{}c@{}}" + value_s +
            r"\\(" + pct_s + r"\%)\end{tabular}")


def build_latejoin_engagement_terminal_table(summary_results: list,
                                             all_approaches: list,
                                             baseline: str,
                                             mark_baseline: bool) -> str:
    latejoin_results = [item for item in summary_results if item.get("group_type") == "latejoin"]
    if not latejoin_results:
        return "[INFO] Late-join engagement table skipped because no late-join tuples were analyzed."
    headers = ["Tuple",
               "CS Approach",
               "Number of Selected Clients\nInitial",
               "Number of Selected Clients\nLate-Joining",
               "Number of Scheduled Samples\nInitial",
               "Number of Scheduled Samples\nLate-Joining"]
    rows = []
    group_breaks = set()
    for item in latejoin_results:
        means_all = item["means"]
        group = {"group_type": item.get("group_type"), "group_value": item.get("group_value")}
        available_approaches = [a for a in all_approaches if a in means_all]
        if not available_approaches:
            continue
        if rows:
            group_breaks.add(len(rows))
        for approach_index, approach in enumerate(available_approaches):
            means = means_all[approach]
            if "mean_latejoin_selected_clients_training" not in means:
                continue
            rows.append([
                group_display_terminal(group) if approach_index == 0 else "",
                make_terminal_approach_label(approach, baseline, mark_baseline),
                terminal_value_share_cell(
                    means.get("mean_initial_selected_clients_training", 0.0),
                    means.get("initial_selected_clients_share_training_pct", 0.0),
                    digits=2,
                    comma=False,
                ),
                terminal_value_share_cell(
                    means.get("mean_latejoin_selected_clients_training", 0.0),
                    means.get("latejoin_selected_clients_share_training_pct", 0.0),
                    digits=2,
                    comma=False,
                ),
                terminal_value_share_cell(
                    means.get("mean_initial_samples_training", 0.0),
                    means.get("initial_samples_share_training_pct", 0.0),
                    digits=2,
                    comma=True,
                ),
                terminal_value_share_cell(
                    means.get("mean_latejoin_samples_training", 0.0),
                    means.get("latejoin_samples_share_training_pct", 0.0),
                    digits=2,
                    comma=True,
                ),
            ])
    if not rows:
        return "[WARNING] No usable rows found for the late-join engagement terminal table."
    table = _render_terminal_grid(headers, rows, group_breaks)
    units = ("Late-join engagement training table. Values are post-entry training-round means up to RoA@X. "
             "Initial clients/samples correspond to clients already present before the late-entry event; late clients/samples "
             "correspond to late-joining clients after they become available. Percentages are within-approach shares of the "
             "post-entry selected-client set or scheduled/processed sample workload, not comparisons against FedAvg.")
    return table + "\n" + units


def build_latejoin_engagement_latex_table(summary_results: list,
                                          all_approaches: list,
                                          baseline: str,
                                          dataset_name: str,
                                          dataset_distribution: str,
                                          num_available_clients: int,
                                          latex_resize_width: str,
                                          mark_baseline: bool,
                                          latex_caption: str = "",
                                          latex_label: str = "") -> str:
    latejoin_results = [item for item in summary_results if item.get("group_type") == "latejoin"]
    if not latejoin_results:
        return ""
    dataset_label = dataset_display_name(dataset_name)
    distribution_label = distribution_display_name(dataset_distribution)
    if not latex_caption:
        latex_caption = "Late-joining workload redistribution for {0} {1}.".format(dataset_label, distribution_label)
    if not latex_label:
        latex_label = "tab:late_joining_engagement_{0}_{1}".format(latex_label_fragment(dataset_name),
                                                                    latex_label_fragment(dataset_distribution))
    lines = []
    lines.append("% {0} {1} late-joining workload redistribution results {2}-clients system.".format(dataset_label, distribution_label, num_available_clients))
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"    \setlength{\tabcolsep}{2pt}")
    lines.append(r"    \centering")
    lines.append(r"    \caption{" + latex_escape(latex_caption) + "}")
    lines.append(r"    \label{" + latex_label + "}")
    lines.append(r"    \resizebox{" + latex_resize_width + r"}{!}{")
    lines.append(r"        \begin{tabular}{cccccc}")
    lines.append(r"            \toprule")
    lines.append(r"            \multirow{2}{*}{\textbf{Tuple}} &")
    lines.append(r"            \multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}CS\\Approach\end{tabular}}} &")
    lines.append(r"            \multicolumn{2}{c}{\textbf{Number of Selected Clients}} &")
    lines.append(r"            \multicolumn{2}{c}{\textbf{Number of Scheduled Samples}} \\")
    lines.append(r"            \cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    lines.append(r"             & & \textbf{Initial} & \textbf{Late-Joining} & \textbf{Initial} & \textbf{Late-Joining} \\")
    lines.append(r"            \midrule")
    lines.append("")
    for group_index, item in enumerate(latejoin_results):
        means_all = item["means"]
        group = {"group_type": item.get("group_type"), "group_value": item.get("group_value")}
        available_approaches = [a for a in all_approaches if a in means_all]
        if not available_approaches:
            continue
        available_with_metrics = [a for a in available_approaches
                                  if "mean_latejoin_selected_clients_training" in means_all[a]]
        if not available_with_metrics:
            continue
        for approach_index, approach in enumerate(available_with_metrics):
            means = means_all[approach]
            cells = []
            if approach_index == 0:
                cells.append(r"            \multirow{" + str(len(available_with_metrics)) + r"}{*}{" + group_display_latex(group) + r"}")
            else:
                cells.append(r"            ")
            cells.append(make_latex_approach_label(approach, baseline, mark_baseline))
            cells.append(latex_value_share_cell(means.get("mean_initial_selected_clients_training", 0.0),
                                                means.get("initial_selected_clients_share_training_pct", 0.0),
                                                digits=2,
                                                comma=False))
            cells.append(latex_value_share_cell(means.get("mean_latejoin_selected_clients_training", 0.0),
                                                means.get("latejoin_selected_clients_share_training_pct", 0.0),
                                                digits=2,
                                                comma=False))
            cells.append(latex_value_share_cell(means.get("mean_initial_samples_training", 0.0),
                                                means.get("initial_samples_share_training_pct", 0.0),
                                                digits=2,
                                                comma=True))
            cells.append(latex_value_share_cell(means.get("mean_latejoin_samples_training", 0.0),
                                                means.get("latejoin_samples_share_training_pct", 0.0),
                                                digits=2,
                                                comma=True))
            lines.append(" & ".join(cells) + r" \\")
        if group_index < len(latejoin_results) - 1:
            lines.append(r"            \arrayrulecolor{lightgray}\specialrule{0.5pt}{0pt}{0pt}")
            lines.append("")
    lines.append(r"            \bottomrule")
    lines.append(r"            \multicolumn{6}{l}{Training-round means after late-client entry up to RoA@X. Percentages are within-approach workload shares, not FedAvg deltas.}")
    lines.append(r"        \end{tabular}")
    lines.append(r"    }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def export_latejoin_engagement_csv(summary_results: list,
                                   all_approaches: list,
                                   baseline: str,
                                   output_file: Path) -> DataFrame:
    rows = []
    latejoin_results = [item for item in summary_results if item.get("group_type") == "latejoin"]
    for item in latejoin_results:
        means_all = item["means"]
        tuple_value = item.get("group_value")
        for approach in all_approaches:
            if approach not in means_all:
                continue
            means = means_all[approach]
            if "mean_latejoin_selected_clients_training" not in means:
                continue
            rows.append({
                "tuple": tuple_to_terminal(tuple_value),
                "num_late_clients": tuple_value[0],
                "entry_round": tuple_value[1],
                "performance_type": tuple_value[2],
                "approach": approach,
                "approach_label": APPROACH_LABELS.get(approach, approach.upper()),
                "mean_initial_selected_clients_training": means.get("mean_initial_selected_clients_training", 0.0),
                "mean_latejoin_selected_clients_training": means.get("mean_latejoin_selected_clients_training", 0.0),
                "mean_total_postentry_selected_clients_training": means.get("mean_total_postentry_selected_clients_training", 0.0),
                "initial_selected_clients_share_training_pct": means.get("initial_selected_clients_share_training_pct", 0.0),
                "latejoin_selected_clients_share_training_pct": means.get("latejoin_selected_clients_share_training_pct", 0.0),
                "latejoin_selected_clients_pct_vs_initial_training": means.get("latejoin_selected_clients_pct_vs_initial_training", 0.0),
                "mean_initial_samples_training": means.get("mean_initial_samples_training", 0.0),
                "mean_latejoin_samples_training": means.get("mean_latejoin_samples_training", 0.0),
                "mean_total_postentry_samples_training": means.get("mean_total_postentry_samples_training", 0.0),
                "initial_samples_share_training_pct": means.get("initial_samples_share_training_pct", 0.0),
                "latejoin_samples_share_training_pct": means.get("latejoin_samples_share_training_pct", 0.0),
                "latejoin_samples_pct_vs_initial_training": means.get("latejoin_samples_pct_vs_initial_training", 0.0),
            })
    df = DataFrame(rows)
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
    return df


def plot_latejoin_engagement_bars(summary_results: list,
                                  all_approaches: list,
                                  output_folder: Path,
                                  dataset_name: str,
                                  dataset_distribution: str) -> None:
    """Generate styled stacked bar plots for post-entry workload redistribution."""
    if output_folder is None:
        return
    output_folder.mkdir(parents=True, exist_ok=True)
    df = export_latejoin_engagement_csv(summary_results,
                                        all_approaches,
                                        baseline="",
                                        output_file=output_folder / "latejoin_engagement_summary.csv")
    if df.empty:
        print("[WARNING] Late-join engagement plot skipped because there are no rows.")
        return
    tuple_labels = list(dict.fromkeys(df["tuple"].tolist()))
    approach_order = [a for a in all_approaches if a in set(df["approach"].tolist())]
    def make_composition_plot(initial_col: str,
                              late_col: str,
                              share_col: str,
                              ylabel: str,
                              filename: str):
        n_approaches = len(approach_order)
        tuple_gap = 1.45
        bar_width = 0.72
        tuple_centers = []
        current_x = 0.0
        max_bar_total = 0.0
        fig_width = max(8.5, len(tuple_labels) * max(4.0, n_approaches * 0.55))
        fig, ax = plt.subplots(figsize=(fig_width, 4.9))
        # Use a single legend shared by all tuples.  The legend identifies the
        # approach colors; the stacked bars still show the initial share as a
        # lighter base and the late-joining share as the solid top segment.
        legend_handles = []
        legend_labels = []
        for approach_index, approach in enumerate(approach_order):
            color, marker, label = style_for_approach(approach, approach_index)
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=color,
                                                edgecolor="black", linewidth=0.4, alpha=0.95))
            legend_labels.append(label)
        for tuple_index, tuple_label in enumerate(tuple_labels):
            tuple_start = current_x
            bars_in_tuple = 0
            for approach_index, approach in enumerate(approach_order):
                subset = df[(df["tuple"] == tuple_label) & (df["approach"] == approach)]
                if subset.empty:
                    continue
                row = subset.iloc[0]
                initial_value = float(row.get(initial_col, 0.0))
                late_value = float(row.get(late_col, 0.0))
                late_share = float(row.get(share_col, 0.0))
                color, marker, label = style_for_approach(approach, approach_index)
                x = current_x
                # Initial workload is shown as a lighter transparent base; late workload uses
                # the exact approach color from the accuracy-curve script.
                ax.bar(x, initial_value, bar_width, color=color, alpha=0.22,
                       edgecolor=color, linewidth=0.8)
                ax.bar(x, late_value, bar_width, bottom=initial_value, color=color, alpha=0.95,
                       edgecolor="black", linewidth=0.4)
                total_value = initial_value + late_value
                max_bar_total = max(max_bar_total, total_value)
                if total_value > 0:
                    # Keep a small, fixed visual gap between the bar top and the
                    # percentage annotation.  Using an offset in points is more
                    # stable than adding a data-unit offset because the selected-client
                    # and sample plots have very different y-axis scales.
                    ax.annotate("{0:.1f}%".format(late_share),
                                xy=(x, total_value),
                                xytext=(0, 7),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                                fontweight="bold",
                                rotation=90)
                current_x += 1.0
                bars_in_tuple += 1
            tuple_end = current_x - 1.0
            if bars_in_tuple > 0:
                tuple_centers.append((tuple_start + tuple_end) / 2.0)
            else:
                tuple_centers.append(tuple_start)
            current_x += tuple_gap
        ax.set_ylabel(ylabel)
        ax.set_xticks(tuple_centers)
        ax.set_xticklabels(tuple_labels, fontweight="bold")
        ax.tick_params(axis="x", length=0, pad=8)
        ax.grid(axis="y", alpha=0.3)
        # Reserve vertical room so percentage labels do not collide with the
        # legend placed above the axes.
        ymin, ymax = ax.get_ylim()
        if max_bar_total > 0:
            ax.set_ylim(ymin, max(ymax, max_bar_total * 1.22))
        legend_ncol = min(len(legend_handles), 4)
        ax.legend(legend_handles, legend_labels,
                  loc="upper center",
                  bbox_to_anchor=(0.5, 1.22),
                  ncol=legend_ncol,
                  frameon=True,
                  framealpha=0.95,
                  columnspacing=1.2,
                  handlelength=1.4)
        # No x-axis title: tuple labels are sufficient and avoid redundant text.
        ax.set_xlabel("")
        fig.subplots_adjust(top=0.76, bottom=0.14)
        fig.savefig(output_folder / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    suffix = "{0}_{1}".format(dataset_name, dataset_distribution)
    make_composition_plot("mean_initial_selected_clients_training",
                          "mean_latejoin_selected_clients_training",
                          "latejoin_selected_clients_share_training_pct",
                          "Mean selected clients per post-entry round",
                          "latejoin_selected_clients_composition_{0}.pdf".format(suffix))
    make_composition_plot("mean_initial_samples_training",
                          "mean_latejoin_samples_training",
                          "latejoin_samples_share_training_pct",
                          "Mean samples per post-entry round",
                          "latejoin_samples_composition_{0}.pdf".format(suffix))


def summarize_one_group(specific_performance_results_folder: Path,
                        dataset_name: str,
                        dataset_distribution: str,
                        num_trials: int,
                        phase: str,
                        all_approaches: list,
                        baseline: str,
                        target_weighted_mean_testing_accuracy: float,
                        group: dict,
                        samples_per_task: int,
                        latejoin_samples_mode: str) -> dict:
    r_suffix = ""
    metrics_means_all = {}
    metrics_stds_all = {}
    group_type = group.get("group_type", "default")
    group_value = group.get("group_value")
    latejoin_tuple = group_value if group_type == "latejoin" else None
    availability_scenario = group_value if group_type == "availability" else None
    print("\n============================================================")
    if group_type == "default":
        print("Static/default result folders")
    elif group_type == "latejoin":
        print("Late-join tuple: num_late_clients={0}, entry_round={1}, performance_type={2}".format(*latejoin_tuple))
    elif group_type == "availability":
        print("Availability scenario: {0}".format(availability_scenario))
    print("============================================================")
    for name in all_approaches:
        all_trials_selected_clients = {"min": inf, "max": 0}
        base_path = construct_base_path(specific_performance_results_folder,
                                        dataset_name,
                                        dataset_distribution,
                                        name,
                                        exec_id=1,
                                        r_suffix=r_suffix,
                                        latejoin_tuple=latejoin_tuple,
                                        availability_scenario=availability_scenario)
        trials = load_multiple_trials(base_path, num_trials)
        trial_metrics = []
        for trial in trials:
            fl_round_target, _ = get_fl_round_to_target_accuracy(trial["individual_evaluate_metrics_history"],
                                                                 target_weighted_mean_testing_accuracy,
                                                                 phase)
            metrics_summary = calculate_metrics_summary(trial["selected_fit_clients_history"],
                                                        trial["individual_fit_metrics_history"],
                                                        trial["individual_evaluate_metrics_history"],
                                                        trial["clients_resources"],
                                                        target_weighted_mean_testing_accuracy,
                                                        fl_round_target,
                                                        all_trials_selected_clients,
                                                        samples_per_task)
            if group_type == "latejoin":
                latejoin_df = load_late_joining_clients_df(trial["trial_path"])
                latejoin_metrics = calculate_latejoin_engagement_trial_metrics(
                    trial["selected_fit_clients_history"],
                    trial["individual_fit_metrics_history"],
                    latejoin_df,
                    fl_round_target,
                    samples_per_task,
                    latejoin_samples_mode,
                )
                metrics_summary.update(latejoin_metrics)
            trial_metrics.append(metrics_summary)
        if not trial_metrics:
            print("\n------- {0} -------".format(APPROACH_LABELS.get(name, name.upper())))
            print("[WARNING] No usable trials found for {0}".format(base_path))
            continue
        means = {k: sum(d[k] for d in trial_metrics) / len(trial_metrics)
                 for k in trial_metrics[0]}
        stds = {k: std([d[k] for d in trial_metrics])
                for k in trial_metrics[0]}
        print("\n------- {0} -------".format(APPROACH_LABELS.get(name, name.upper())))
        for k in means:
            print("{0}: {1} ± {2}".format(k, smart_format(means[k]), smart_format(stds[k])))
        metrics_means_all[name] = means
        metrics_stds_all[name] = stds
    if baseline not in metrics_means_all:
        print("\n[WARNING] Baseline '{0}' has no usable metrics for this group; skipping percentage changes.".format(baseline))
        return {"group_type": group_type, "group_value": group_value, "means": metrics_means_all, "stds": metrics_stds_all}
    ref = metrics_means_all[baseline]
    print("\n------- Percentage Change vs {0} -------".format(APPROACH_LABELS.get(baseline, baseline.upper())))
    for name, metrics in metrics_means_all.items():
        if name == baseline:
            continue
        print("\n{0}:".format(APPROACH_LABELS.get(name, name.upper())))
        for k, v in metrics.items():
            if ref[k] == 0:
                print(" - {0}: N/A".format(k))
            else:
                pct = ((v - ref[k]) / abs(ref[k])) * 100
                print(" - {0}: {1}%".format(k, smart_format(pct)))
    return {"group_type": group_type, "group_value": group_value, "means": metrics_means_all, "stds": metrics_stds_all}


def build_latex_table(summary_results: list,
                      all_approaches: list,
                      baseline: str,
                      dataset_name: str,
                      dataset_distribution: str,
                      num_available_clients: int,
                      target_accuracy: float,
                      latex_resize_width: str,
                      latex_caption: str,
                      latex_label: str,
                      mark_baseline: bool,
                      energy_metric: str) -> str:
    has_grouped_results = any(item.get("group_type") != "default" for item in summary_results)
    is_availability_table = bool(summary_results) and summary_results[0].get("group_type") == "availability"
    dataset_label = dataset_display_name(dataset_name)
    distribution_label = distribution_display_name(dataset_distribution)
    target_s = format_latex_number(target_accuracy, 2, comma=False).rstrip("0").rstrip(".")
    if not latex_caption:
        if has_grouped_results and summary_results[0].get("group_type") == "latejoin":
            latex_caption = "Late-joining performance for {0} {1} under representative configurations.".format(dataset_label, distribution_label)
        elif has_grouped_results and summary_results[0].get("group_type") == "availability":
            latex_caption = "Intermittent-availability performance for {0} {1} under representative scenarios.".format(dataset_label, distribution_label)
        else:
            latex_caption = "Performances for {0} {1} ({2}-client system).".format(dataset_label, distribution_label, num_available_clients)
    if not latex_label:
        if has_grouped_results and summary_results[0].get("group_type") == "latejoin":
            latex_label = "tab:late_joining_{0}_{1}".format(latex_label_fragment(dataset_name), latex_label_fragment(dataset_distribution))
        elif has_grouped_results and summary_results[0].get("group_type") == "availability":
            latex_label = "tab:intermittent_availability_{0}_{1}".format(latex_label_fragment(dataset_name), latex_label_fragment(dataset_distribution))
        else:
            latex_label = "tab:cs_algorithms_performance_{0}_{1}".format(latex_label_fragment(dataset_name), latex_label_fragment(dataset_distribution))
    if energy_metric == "training_idle":
        energy_key = "total_training+idle_energy_during_training_in_joules"
    else:
        energy_key = "total_training_energy_in_joules"
    lines = []
    lines.append("% {0} {1} results {2}-clients system.".format(dataset_label, distribution_label, num_available_clients))
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"    \setlength{\tabcolsep}{2pt}")
    lines.append(r"    \centering")
    lines.append(r"    \caption{" + latex_escape(latex_caption) + "}")
    lines.append(r"    \label{" + latex_label + "}")
    lines.append(r"    \resizebox{" + latex_resize_width + r"}{!}{")
    if has_grouped_results:
        if is_availability_table:
            lines.append(r"        \begin{tabular}{ccccccccc}")
        else:
            lines.append(r"        \begin{tabular}{ccccccc}")
        lines.append(r"            \toprule")
        lines.append(r"            \textbf{" + latex_escape(group_header(summary_results[0].get("group_type"))) + r"} &")
        lines.append(r"            \textbf{\begin{tabular}[c]{@{}c@{}}CS\\Approach\end{tabular}} &")
    else:
        lines.append(r"        \begin{tabular}{cccccc}")
        lines.append(r"            \toprule")
        lines.append(r"            \textbf{\begin{tabular}[c]{@{}c@{}}CS\\Approach\end{tabular}} &")
    lines.append(r"            \textbf{$\mathrm{M}_{\text{total}}$} &")
    lines.append(r"            \textbf{$\Sigma_{\text{total}}$} &")
    lines.append(r"            \textbf{$\mathrm{T}_{\text{cs}}$} &")
    lines.append(r"            \textbf{$\mathrm{S}_{\text{fair}}$} &")
    if is_availability_table:
        lines.append(r"            \textbf{$\bar{n}_{\text{fail}}$} &")
        lines.append(r"            \textbf{$\bar{s}_{\text{fail}}$} &")
    lines.append(r"            \textbf{$\mathrm{RoA}@" + target_s + r"$} \\")
    lines.append(r"            \midrule")
    lines.append("")
    for tuple_index, item in enumerate(summary_results):
        means_all = item["means"]
        stds_all = item["stds"]
        group = {"group_type": item.get("group_type"), "group_value": item.get("group_value")}
        ref = means_all.get(baseline)
        available_approaches = [a for a in all_approaches if a in means_all]
        if not available_approaches:
            continue
        for approach_index, approach in enumerate(available_approaches):
            means = means_all[approach]
            stds = stds_all[approach]
            ref_metrics = ref if ref is not None and approach != baseline else None
            cells = []
            if has_grouped_results:
                if approach_index == 0:
                    cells.append(r"            \multirow{" + str(len(available_approaches)) + r"}{*}{" + group_display_latex(group) + r"}")
                else:
                    cells.append(r"            ")
            else:
                cells.append(r"            ")
            cells.append(make_latex_approach_label(approach, baseline, mark_baseline))
            cells.append(latex_metric_cell(means["total_training_time_in_seconds"],
                                           None if ref_metrics is None else ref_metrics["total_training_time_in_seconds"],
                                           digits=2, comma=True))
            cells.append(latex_metric_cell(means[energy_key] / 1000.0,
                                           None if ref_metrics is None else ref_metrics[energy_key] / 1000.0,
                                           digits=2, comma=True))
            cells.append(latex_metric_cell(means["total_selection_duration_in_seconds"],
                                           None if ref_metrics is None else ref_metrics["total_selection_duration_in_seconds"],
                                           digits=2, comma=True))
            cells.append(latex_metric_cell(means["jain_fairness_index_training"],
                                           None if ref_metrics is None else ref_metrics["jain_fairness_index_training"],
                                           digits=2, comma=False))
            if is_availability_table:
                cells.append(latex_metric_cell(means["mean_number_failing_clients_training"],
                                               None if ref_metrics is None else ref_metrics["mean_number_failing_clients_training"],
                                               digits=2, comma=False))
                cells.append(latex_metric_cell(means["mean_number_failing_tasks_training"],
                                               None if ref_metrics is None else ref_metrics["mean_number_failing_tasks_training"],
                                               digits=2, comma=True))
            cells.append(latex_roa_cell(means["fl_round_@_target_accuracy"],
                                        stds["fl_round_@_target_accuracy"]))
            lines.append(" & ".join(cells) + r" \\")
        if tuple_index < len(summary_results) - 1:
            lines.append(r"            \arrayrulecolor{lightgray}\specialrule{0.5pt}{0pt}{0pt}")
            lines.append("")
    lines.append(r"            \bottomrule")
    if has_grouped_results:
        if is_availability_table:
            lines.append(r"            \multicolumn{9}{l}{Units: $\mathrm{M}_{\text{total}}$ and $\mathrm{T}_{\text{cs}}$ in seconds; $\Sigma_{\text{total}}$ in kilojoules. $\bar{n}_{\text{fail}}$ and $\bar{s}_{\text{fail}}$ are per-training-round means; $\bar{s}_{\text{fail}}$ is reported in samples.}")
        else:
            lines.append(r"            \multicolumn{7}{l}{Units: $\mathrm{M}_{\text{total}}$ and $\mathrm{T}_{\text{cs}}$ in seconds; $\Sigma_{\text{total}}$ in kilojoules.}")
    else:
        lines.append(r"            \multicolumn{6}{l}{Units: $\mathrm{M}_{\text{total}}$ and $\mathrm{T}_{\text{cs}}$ in seconds; $\Sigma_{\text{total}}$ in kilojoules.}")
    lines.append(r"        \end{tabular}")
    lines.append(r"    }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main(performance_results_folder: Path,
         dataset_name: str,
         dataset_distribution: str,
         num_available_clients: int,
         num_trials: int,
         phase: str,
         all_approaches: str,
         baseline: str,
         desired_latejoin_tuples: str,
         latejoin_num_clients: str,
         latejoin_entry_rounds: str,
         latejoin_performance_types: str,
         availability_scenarios: str,
         print_terminal_table: bool,
         print_latex_table: bool,
         latex_output_file: Path,
         latex_resize_width: str,
         latex_caption: str,
         latex_label: str,
         mark_baseline_in_latex: bool,
         energy_metric: str,
         samples_per_task_arg: int,
         print_dropout_table: bool,
         dropout_latex_output_file: Path,
         print_latejoin_engagement_table: bool,
         latejoin_engagement_latex_output_file: Path,
         latejoin_engagement_output_folder: Path,
         latejoin_samples_mode: str) -> None:
    specific_performance_results_folder = performance_results_folder.joinpath("{0}_clients".format(num_available_clients))
    all_approaches = parse_csv_list(all_approaches)
    latejoin_tuples = build_latejoin_tuples(desired_latejoin_tuples,
                                            latejoin_num_clients,
                                            latejoin_entry_rounds,
                                            latejoin_performance_types)
    availability_scenario_list = build_availability_scenarios(availability_scenarios)
    analysis_groups = build_analysis_groups(latejoin_tuples, availability_scenario_list)
    target_testing_accuracies = {"cifar_10": {"iid": 0.75, "non_iid": 0.45},
                                 "fashion_mnist": {"iid": 0.85, "non_iid": 0.70},
                                 "emotion": {"iid": 0.90, "non_iid": 0.80}}
    target_weighted_mean_testing_accuracy = target_testing_accuracies[dataset_name][dataset_distribution]
    samples_per_task = infer_samples_per_task(dataset_name, samples_per_task_arg)
    print("\n")
    print("- Dataset Name:", dataset_name)
    print("- Dataset Distribution:", dataset_distribution)
    print("- Target Testing Accuracy:", target_weighted_mean_testing_accuracy)
    print("- Phase:", phase)
    print("- Number of Trials:", num_trials)
    print("- Approaches:", ",".join(all_approaches))
    print("- Baseline:", baseline)
    print("- Analysis Groups:", analysis_groups)
    print("- Samples per Task:", samples_per_task)
    summary_results = []
    for group in analysis_groups:
        result = summarize_one_group(specific_performance_results_folder,
                                     dataset_name,
                                     dataset_distribution,
                                     num_trials,
                                     phase,
                                     all_approaches,
                                     baseline,
                                     target_weighted_mean_testing_accuracy,
                                     group,
                                     samples_per_task,
                                     latejoin_samples_mode)
        summary_results.append(result)
    if print_terminal_table:
        terminal_table = build_terminal_table(summary_results,
                                              all_approaches,
                                              baseline,
                                              target_weighted_mean_testing_accuracy,
                                              mark_baseline_in_latex,
                                              energy_metric)
        print("\n\n==================== FINAL TERMINAL TABLE ====================\n")
        print(terminal_table)
        print("\n================== END FINAL TERMINAL TABLE ==================\n")
    if print_dropout_table and has_availability_groups(summary_results):
        dropout_terminal_table = build_dropout_terminal_table(summary_results,
                                                              all_approaches,
                                                              baseline,
                                                              mark_baseline_in_latex)
        print("\n\n==================== DROPOUT-FOCUSED TERMINAL TABLE ====================\n")
        print(dropout_terminal_table)
        print("\n================== END DROPOUT-FOCUSED TERMINAL TABLE ==================\n")
    if print_latejoin_engagement_table and has_latejoin_groups(summary_results):
        latejoin_engagement_terminal_table = build_latejoin_engagement_terminal_table(summary_results,
                                                                                      all_approaches,
                                                                                      baseline,
                                                                                      mark_baseline_in_latex)
        print("\n\n==================== LATE-JOIN ENGAGEMENT TERMINAL TABLE ====================\n")
        print(latejoin_engagement_terminal_table)
        print("\n================== END LATE-JOIN ENGAGEMENT TERMINAL TABLE ==================\n")
        if latejoin_engagement_output_folder is not None:
            latejoin_engagement_output_folder.mkdir(parents=True, exist_ok=True)
            export_latejoin_engagement_csv(summary_results,
                                           all_approaches,
                                           baseline,
                                           latejoin_engagement_output_folder / "latejoin_engagement_summary.csv")
            plot_latejoin_engagement_bars(summary_results,
                                          all_approaches,
                                          latejoin_engagement_output_folder,
                                          dataset_name,
                                          dataset_distribution)
            print("[INFO] Late-join engagement CSV/plots written to: {0}".format(latejoin_engagement_output_folder))
    if print_latex_table:
        latex_table = build_latex_table(summary_results,
                                        all_approaches,
                                        baseline,
                                        dataset_name,
                                        dataset_distribution,
                                        num_available_clients,
                                        target_weighted_mean_testing_accuracy,
                                        latex_resize_width,
                                        latex_caption,
                                        latex_label,
                                        mark_baseline_in_latex,
                                        energy_metric)
        print("\n\n==================== COPY-READY LATEX TABLE ====================\n")
        print(latex_table)
        print("\n================== END COPY-READY LATEX TABLE ==================\n")
        if latex_output_file is not None:
            latex_output_file.parent.mkdir(parents=True, exist_ok=True)
            latex_output_file.write_text(latex_table + "\n", encoding="utf-8")
            print("[INFO] LaTeX table written to: {0}".format(latex_output_file))
    if print_latejoin_engagement_table and print_latex_table and has_latejoin_groups(summary_results):
        latejoin_engagement_latex_table = build_latejoin_engagement_latex_table(summary_results,
                                                                                all_approaches,
                                                                                baseline,
                                                                                dataset_name,
                                                                                dataset_distribution,
                                                                                num_available_clients,
                                                                                latex_resize_width,
                                                                                mark_baseline_in_latex)
        if latejoin_engagement_latex_table:
            print("\n\n==================== COPY-READY LATE-JOIN ENGAGEMENT LATEX TABLE ====================\n")
            print(latejoin_engagement_latex_table)
            print("\n================== END COPY-READY LATE-JOIN ENGAGEMENT LATEX TABLE ==================\n")
            if latejoin_engagement_latex_output_file is not None:
                latejoin_engagement_latex_output_file.parent.mkdir(parents=True, exist_ok=True)
                latejoin_engagement_latex_output_file.write_text(latejoin_engagement_latex_table + "\n", encoding="utf-8")
                print("[INFO] Late-join engagement LaTeX table written to: {0}".format(latejoin_engagement_latex_output_file))
    if print_dropout_table and print_latex_table and has_availability_groups(summary_results):
        dropout_latex_table = build_dropout_latex_table(summary_results,
                                                        all_approaches,
                                                        baseline,
                                                        dataset_name,
                                                        dataset_distribution,
                                                        num_available_clients,
                                                        latex_resize_width,
                                                        mark_baseline_in_latex)
        if dropout_latex_table:
            print("\n\n==================== COPY-READY DROPOUT LATEX TABLE ====================\n")
            print(dropout_latex_table)
            print("\n================== END COPY-READY DROPOUT LATEX TABLE ==================\n")
            if dropout_latex_output_file is not None:
                dropout_latex_output_file.parent.mkdir(parents=True, exist_ok=True)
                dropout_latex_output_file.write_text(dropout_latex_table + "\n", encoding="utf-8")
                print("[INFO] Dropout LaTeX table written to: {0}".format(dropout_latex_output_file))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--performance-results-folder", type=Path, required=True,
                        help="Relative path to the performance results folder (input)")
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
    parser.add_argument("--baseline", type=str, required=True,
                        help="Baseline approach")
    parser.add_argument("--desired-latejoin-tuples", type=str, default="",
                        help="Optional semicolon-separated tuples: num_late_clients:entry_round:performance_type. Example: '10:10:worst;10:25:best'.")
    parser.add_argument("--latejoin-num-clients", type=str, default="",
                        help="Optional comma-separated late-client counts. Used with --latejoin-entry-rounds and --latejoin-performance-types.")
    parser.add_argument("--latejoin-entry-rounds", type=str, default="",
                        help="Optional comma-separated late-client entry rounds. Used with --latejoin-num-clients and --latejoin-performance-types.")
    parser.add_argument("--latejoin-performance-types", type=str, default="",
                        help="Optional comma-separated performance types, e.g., worst,best. Used with --latejoin-num-clients and --latejoin-entry-rounds.")
    parser.add_argument("--availability-scenarios", type=str, default="",
                        help="Optional comma-separated intermittent availability scenarios, e.g., moderate,severe. Matches folders like approach_moderate_exec_5000.")
    parser.add_argument("--no-terminal-table", action="store_true",
                        help="Disable the final pretty terminal table printed at the end.")
    parser.add_argument("--no-latex-table", action="store_true",
                        help="Disable the final copy-ready LaTeX table printed at the end.")
    parser.add_argument("--latex-output-file", type=Path, default=None,
                        help="Optional path to also save the generated LaTeX table as a .tex file.")
    parser.add_argument("--latex-resize-width", type=str, default="0.8\\columnwidth",
                        help=r"Width passed to \resizebox, e.g., 0.8\columnwidth or 0.6\textwidth.")
    parser.add_argument("--latex-caption", type=str, default="",
                        help="Optional custom LaTeX table caption.")
    parser.add_argument("--latex-label", type=str, default="",
                        help="Optional custom LaTeX table label.")
    parser.add_argument("--no-mark-baseline-in-latex", action="store_true",
                        help="Do not append '(Baseline)' to the baseline approach in the LaTeX table.")
    parser.add_argument("--energy-metric", type=str, default="training", choices=["training", "training_idle"],
                        help="Energy shown in Sigma_total: 'training' uses measured training energy; 'training_idle' also includes idle energy during training.")
    parser.add_argument("--samples-per-task", type=int, default=None,
                        help="Optional override for how many training samples one scheduled task represents. Defaults: cifar_10=1, fashion_mnist=1, emotion=10.")
    parser.add_argument("--no-dropout-table", action="store_true",
                        help="Disable the extra dropout-focused terminal/LaTeX table for intermittent availability results.")
    parser.add_argument("--dropout-latex-output-file", type=Path, default=None,
                        help="Optional path to save the extra dropout-focused LaTeX table as a .tex file.")
    parser.add_argument("--no-latejoin-engagement-table", action="store_true",
                        help="Disable the extra late-join engagement terminal/LaTeX table for late-joining results.")
    parser.add_argument("--latejoin-engagement-latex-output-file", type=Path, default=None,
                        help="Optional path to save the extra late-join engagement LaTeX table as a .tex file.")
    parser.add_argument("--latejoin-engagement-output-folder", type=Path, default=None,
                        help="Optional folder to save the late-join engagement CSV and grouped bar plots.")
    parser.add_argument("--latejoin-samples-mode", type=str, default="completed",
                        choices=["completed", "proportional_scheduled"],
                        help=("How to compute late-join samples. 'completed' uses exact ds_train_i/num_examples "
                              "from individual_fit_metrics_history.csv for late clients that completed. "
                              "'proportional_scheduled' estimates scheduled samples from total round workload "
                              "and the fraction of selected clients that are late joiners."))
    args = parser.parse_args()
    main(args.performance_results_folder,
         args.dataset_name,
         args.dataset_distribution,
         args.num_available_clients,
         args.num_trials,
         args.phase,
         args.all_approaches,
         args.baseline,
         args.desired_latejoin_tuples,
         args.latejoin_num_clients,
         args.latejoin_entry_rounds,
         args.latejoin_performance_types,
         args.availability_scenarios,
         not args.no_terminal_table,
         not args.no_latex_table,
         args.latex_output_file,
         args.latex_resize_width,
         args.latex_caption,
         args.latex_label,
         not args.no_mark_baseline_in_latex,
         args.energy_metric,
         args.samples_per_task,
         not args.no_dropout_table,
         args.dropout_latex_output_file,
         not args.no_latejoin_engagement_table,
         args.latejoin_engagement_latex_output_file,
         args.latejoin_engagement_output_folder,
         args.latejoin_samples_mode)
