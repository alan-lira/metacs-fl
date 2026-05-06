from argparse import ArgumentParser
from numpy import array, bincount, sum as np_sum, std, inf
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
                         num_trials: int) -> list:
    trials_data = []
    parent = base_path_template.parent
    prefix = base_path_template.name.split("_exec_")[0]
    suffix = base_path_template.name[len("{0}_exec_1".format(prefix)):]
    for i in range(1, num_trials + 1):
        trial_path = parent / "{0}_exec_{1}{2}".format(prefix, i, suffix)
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
            print("[WARNING] Missing files for trial {0}:".format(i))
            for m in missing_files:
                print("  - {0}".format(m))
            continue
        trial_data = {"individual_fit_metrics_history": load_dataframe(fit_path),
                      "individual_evaluate_metrics_history": load_dataframe(eval_path),
                      "selected_fit_clients_history": load_dataframe(selected_path),
                      "clients_resources": load_dataframe(resources_path)}
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
    std = combined.std(axis=1)
    return DataFrame({"comm_round": combined.index,
                      "mean_accuracy": mean,
                      "std_accuracy": std}).reset_index(drop=True)


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
                              all_trials_selected_clients: dict) -> dict:
    fl_col = "comm_round"
    time_col = fit_df.columns[fit_df.columns.str.contains("training_time")][0]
    energy_col = fit_df.columns[fit_df.columns.str.contains("training_energy")][0]
    sel_duration_col = selected_df.columns[selected_df.columns.str.contains("selection_duration")][0]
    num_sel_col = selected_df.columns[selected_df.columns.str.contains("num_selected_clients")][0]
    # Training examples column (from fit).
    num_examples_col = fit_df.columns[fit_df.columns.str.contains("examples")][0]
    # Truncate at target round.
    fit_df = fit_df[fit_df[fl_col] <= fl_round_target]
    selected_df = selected_df[selected_df[fl_col] <= fl_round_target]
    # Mean / std number of examples per selected client.
    mean_examples_per_round = fit_df.groupby(fl_col)[num_examples_col].mean()
    mean_examples_scheduled = mean_examples_per_round.mean()
    std_examples_scheduled = mean_examples_per_round.std()
    tasks_per_client = fit_df.groupby("client_id")[num_examples_col].sum()
    cv_tasks = tasks_per_client.std() / tasks_per_client.mean()
    # Base metrics.
    metrics = {"fl_round_@_target_accuracy": fl_round_target,
               "target_accuracy": target_accuracy,
               "max_training_time_across_all_rounds_in_seconds": fit_df.groupby(fl_col)[time_col].max().max(),
               "total_training_time_in_seconds": fit_df.groupby(fl_col)[time_col].max().sum(),
               "total_training_energy_in_joules": fit_df[energy_col].sum(),
               "min_number_selected_clients_training": selected_df[num_sel_col].min(),
               "max_number_selected_clients_training": selected_df[num_sel_col].max(),
               "mean_number_selected_clients_training": selected_df[num_sel_col].mean(),
               "jain_fairness_index_training": calculate_jain_fairness_index(selected_df),
               "mean_examples_per_selected_client_training": mean_examples_scheduled,
               "std_examples_per_selected_client_training": std_examples_scheduled,
               "cv_tasks_per_client_training": cv_tasks}
    all_trials_selected_clients["min"] = min(all_trials_selected_clients["min"],
                                             metrics["min_number_selected_clients_training"])
    all_trials_selected_clients["max"] = max(all_trials_selected_clients["max"],
                                             metrics["max_number_selected_clients_training"])
    # Idle energy (selection + training).
    idle_sel_energy = 0
    idle_train_energy = 0
    total_selection_duration = 0
    for r in range(1, fl_round_target + 1):
        sel_row = selected_df[selected_df[fl_col] == r].iloc[0]
        duration = sel_row[sel_duration_col]
        total_selection_duration += duration
        candidates = list(map(int, sel_row["available_clients"].replace("client_", "").split("|")))
        selected = list(map(int, sel_row["selected_clients"].replace("client_", "").split("|")))
        for cid in candidates:
            idle_power = resources_df[resources_df["client_id"] == cid]["mean_power_consumption_idle_in_watts"].iloc[0]
            idle_sel_energy += duration * idle_power
        makespan = fit_df[fit_df[fl_col] == r][time_col].max()
        for cid in set(candidates) - set(selected):
            idle_power = resources_df[resources_df["client_id"] == cid]["mean_power_consumption_idle_in_watts"].iloc[0]
            idle_train_energy += makespan * idle_power
    # Append metrics.
    metrics.update({"total_selection_duration_in_seconds": total_selection_duration,
                    "total_idle_energy_during_selection_in_joules": idle_sel_energy,
                    "total_idle_energy_during_training_in_joules": idle_train_energy,
                    "total_training+idle_energy_during_training_in_joules": metrics["total_training_energy_in_joules"] + idle_train_energy})
    return metrics


def smart_format(value: float) -> str:
    return ("{0:.2f}".format(value) if float("{0:.4f}".format(value)) == float("{0:.2f}".format(value)) else "{0:.4f}".format(value))


def main(performance_results_folder: Path,
         dataset_name: str,
         dataset_distribution: str,
         num_available_clients: int,
         num_trials: int,
         phase: str,
         all_approaches: str,
         baseline: str) -> None:

    # Get the specific performance results folder.
    specific_performance_results_folder = performance_results_folder.joinpath("{0}_clients".format(num_available_clients))

    # Get the list of all approaches.
    all_approaches = [item.strip() for item in all_approaches.split(",")]

    # Set the target testing accuracies.
    target_testing_accuracies = {"cifar_10": {"iid": 0.75, "non_iid": 0.45},
                                 "fashion_mnist": {"iid": 0.85, "non_iid": 0.70},
                                 "emotion": {"iid": 0.90, "non_iid": 0.80}}

    # Get the corresponding target testing accuracy.
    target_weighted_mean_testing_accuracy = target_testing_accuracies[dataset_name][dataset_distribution]

    print("\n")
    print("- Dataset Name:", dataset_name)
    print("- Dataset Distribution:", dataset_distribution)
    print("- Target Testing Accuracy:", target_weighted_mean_testing_accuracy)
    print("- Phase:", phase)
    print("- Number of Trials:", num_trials)

    # Calculate metrics summary for all approaches.
    # r_list = [1, 10, 25, 50, 75]
    r_list = [""]
    for r in r_list:
        r_suffix = "" if r == "" else "_r_{0}".format(r)
        metrics_means_all = {}
        for name in all_approaches:
            exp_num = name
            all_trials_selected_clients = {"min": inf, "max": 0}
            base_path = construct_base_path(specific_performance_results_folder,
                                            dataset_name,
                                            dataset_distribution,
                                            exp_num,
                                            exec_id=1,
                                            r_suffix=r_suffix)
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
                                                            all_trials_selected_clients)
                trial_metrics.append(metrics_summary)
            means = {k: sum(d[k] for d in trial_metrics) / len(trial_metrics)
                     for k in trial_metrics[0]}
            stds = {k: std([d[k] for d in trial_metrics])
                    for k in trial_metrics[0]}
            print("\n------- {0} -------".format(name.upper()))
            for k in means:
                print("{0}: {1} ± {2}".format(k, smart_format(means[k]), smart_format(stds[k])))
            metrics_means_all[name] = means

        # Baseline comparison.
        ref = metrics_means_all[baseline]

        print("\n------- Percentage Change vs {0} -------".format(baseline.upper()))
        for name, metrics in metrics_means_all.items():
            if name == baseline:
                continue
            print("\n{0}:".format(name.upper()))
            for k, v in metrics.items():
                if ref[k] == 0:
                    print(" - {0}: N/A".format(k))
                else:
                    pct = ((v - ref[k]) / abs(ref[k])) * 100
                    print(" - {0}: {1}%".format(k, smart_format(pct)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--performance-results-folder",
                        type=Path,
                        required=True,
                        help="Relative path to the performance results folder (input)")
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
    parser.add_argument("--baseline",
                        type=str,
                        required=True,
                        help="Baseline approach")
    args = parser.parse_args()
    performance_results_folder = args.performance_results_folder
    dataset_name = args.dataset_name
    dataset_distribution = args.dataset_distribution
    num_available_clients = args.num_available_clients
    num_trials = args.num_trials
    phase = args.phase
    all_approaches = args.all_approaches
    baseline = args.baseline
    main(performance_results_folder,
         dataset_name,
         dataset_distribution,
         num_available_clients,
         num_trials,
         phase,
         all_approaches,
         baseline)
