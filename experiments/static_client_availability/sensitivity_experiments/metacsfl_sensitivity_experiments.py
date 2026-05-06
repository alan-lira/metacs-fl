from argparse import ArgumentParser
from copy import deepcopy
from importlib import import_module
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from traceback import format_exc
from math import inf

from pandas import DataFrame


# Change this if your scalability script has another filename.
# Example: if your file is scalability_experiments.py, keep this as below.
#SCALABILITY_MODULE_NAME = "scalability_experiments"
SCALABILITY_MODULE_NAME = "experiments.static_client_availability.scalability_experiments.scalability_experiments"


OBJECTIVE_WEIGHT_CONFIGS = {
    "current": {
        "M_weight": 0.05,
        "E_weight": 0.05,
        "D_weight": 0.30,
        "K_weight": 0.20,
        "U_weight": 0.40,
    },
    "efficiency_oriented": {
        "M_weight": 0.30,
        "E_weight": 0.30,
        "D_weight": 0.15,
        "K_weight": 0.10,
        "U_weight": 0.15,
    },
    "fairness_oriented": {
        "M_weight": 0.05,
        "E_weight": 0.05,
        "D_weight": 0.45,
        "K_weight": 0.30,
        "U_weight": 0.15,
    },
    "utility_oriented": {
        "M_weight": 0.05,
        "E_weight": 0.05,
        "D_weight": 0.15,
        "K_weight": 0.15,
        "U_weight": 0.60,
    },
}


BASELINE = {
    "train_minimum_diversity_score": 0.85,
    "train_maximum_relative_drop": 0.20,
    "train_diversity_window": 3,

    "makespan_maximum_increase": 0.10,
    "makespan_window": 2,

    "energy_maximum_increase": 0.10,
    "energy_window": 2,

    "accuracy_maximum_decrease": 0.10,
    "accuracy_window": 2,

    "test_minimum_diversity_score": 0.50,
    "test_diversity_window": 5,

    "max_rounds_same_initial_solution": 10,

    "objective_weights_name": "current",
    "alpha": 0.35,
    "beta": 0.70,
    "q": 3,

    "lns_stopping_mode": "time",
    "lns_t_max": 10,
    "lns_it_max": 1000,

    "lns_sf": 0.30,
    "lns_rf_min": 0.15,
    "lns_rf_max": 0.75,

    "lns_D_pc_min": 0.005,
    "lns_M_pc_max": 0.25,
    "lns_E_pc_max": 0.25,

    # Synthetic disturbance control used by your scalability simulator.
    "new_client_selection_trigger_probability": 0.25,
}


SENSITIVITY_VALUES = {
    "train_minimum_diversity_score": [0.80, 0.85, 0.90],
    "train_maximum_relative_drop": [0.05, 0.10, 0.20],
    "train_diversity_window": [2, 3, 5],

    "makespan_maximum_increase": [0.05, 0.10, 0.20],
    "makespan_window": [2, 3, 5],

    "energy_maximum_increase": [0.05, 0.10, 0.20],
    "energy_window": [2, 3, 5],

    "accuracy_maximum_decrease": [0.05, 0.10, 0.20],
    "accuracy_window": [2, 3, 5],

    "test_minimum_diversity_score": [0.40, 0.50, 0.60],
    "test_diversity_window": [3, 5, 7],

    "max_rounds_same_initial_solution": [5, 10, 20],

    "objective_weights_name": [
        "current",
        "efficiency_oriented",
        "fairness_oriented",
        "utility_oriented",
    ],
    "alpha": [0.25, 0.35, 0.50],
    "beta": [0.50, 0.70, 0.90],
    "q": [2, 3, 5],

    "lns_t_max": [1, 5, 10, 20],
    "lns_it_max": [250, 500, 1000, 2000],

    "lns_sf": [0.10, 0.30, 0.50],
    "lns_rf_range": [(0.10, 0.50), (0.15, 0.75), (0.25, 0.90)],

    "lns_accept_tuple": [
        (0.000, 0.10, 0.10),
        (0.005, 0.25, 0.25),
        (0.010, 0.40, 0.40),
    ],

    "new_client_selection_trigger_probability": [0.15, 0.25, 0.50],
}


def load_scalability_module():
    return import_module(SCALABILITY_MODULE_NAME)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def deep_update_meta_cs_fl_settings(settings: dict,
                                    config: dict) -> dict:
    settings = deepcopy(settings)

    # -------------------------------------------------------------------------
    # New client selection criteria: training.
    # -------------------------------------------------------------------------
    training_criteria = settings["new_client_selection_criteria_training"]

    training_criteria["criterion_1"] = {
        "name": "client_diversity_score_training",
        "minimum_score": config["train_minimum_diversity_score"],
        "maximum_relative_drop": config["train_maximum_relative_drop"],
        "num_past_rounds": config["train_diversity_window"],
    }

    training_criteria["criterion_2"] = {
        "name": "makespan_percentage_increase_training",
        "maximum_increase": config["makespan_maximum_increase"],
        "num_past_rounds": config["makespan_window"],
    }

    training_criteria["criterion_3"] = {
        "name": "energy_consumption_percentage_increase_training",
        "maximum_increase": config["energy_maximum_increase"],
        "num_past_rounds": config["energy_window"],
    }

    training_criteria["criterion_4"] = {
        "name": "accuracy_percentage_decrease_testing",
        "maximum_decrease": config["accuracy_maximum_decrease"],
        "num_past_rounds": config["accuracy_window"],
    }

    # -------------------------------------------------------------------------
    # New client selection criteria: testing.
    # -------------------------------------------------------------------------
    settings["new_client_selection_criteria_testing"]["criterion_1"] = {
        "name": "client_diversity_score_testing",
        "minimum_score": config["test_minimum_diversity_score"],
        "num_past_rounds": config["test_diversity_window"],
    }

    # -------------------------------------------------------------------------
    # Initial solution reuse.
    # -------------------------------------------------------------------------
    settings["new_initial_solution_criteria_training"]["criterion_1"] = {
        "name": "max_rounds_same_initial_solution",
        "maximum_rounds": config["max_rounds_same_initial_solution"],
    }

    settings["new_initial_solution_criteria_testing"]["criterion_1"] = {
        "name": "max_rounds_same_initial_solution",
        "maximum_rounds": config["max_rounds_same_initial_solution"],
    }

    # -------------------------------------------------------------------------
    # Objective function.
    # -------------------------------------------------------------------------
    settings["objective_function"]["obj_func_weights"] = deepcopy(
        OBJECTIVE_WEIGHT_CONFIGS[config["objective_weights_name"]]
    )
    settings["objective_function"]["alpha"] = config["alpha"]
    settings["objective_function"]["beta"] = config["beta"]
    settings["objective_function"]["q"] = config["q"]

    # -------------------------------------------------------------------------
    # LNS stopping criteria.
    # -------------------------------------------------------------------------
    if config["lns_stopping_mode"] == "time":
        settings["metaheuristic"]["stopping_criteria"] = {
            "name": "LNS_Stop_Elapsed_Time",
            "t_max": config["lns_t_max"],
        }
    elif config["lns_stopping_mode"] == "iterations":
        settings["metaheuristic"]["stopping_criteria"] = {
            "name": "LNS_Stop_Max_Iterations",
            "it_max": config["lns_it_max"],
        }
    else:
        raise ValueError("Unsupported lns_stopping_mode: {0}".format(config["lns_stopping_mode"]))

    # -------------------------------------------------------------------------
    # LNS destroy approach.
    # -------------------------------------------------------------------------
    settings["metaheuristic"]["destroy_approach"] = {
        "sf": config["lns_sf"],
        "rf_min": config["lns_rf_min"],
        "rf_max": config["lns_rf_max"],
    }

    # -------------------------------------------------------------------------
    # LNS acceptance criteria.
    # -------------------------------------------------------------------------
    settings["metaheuristic"]["accept_criteria"] = {
        "D_pc_min": config["lns_D_pc_min"],
        "M_pc_max": config["lns_M_pc_max"],
        "E_pc_max": config["lns_E_pc_max"],
        "bl_min": 0,
    }

    return settings


def build_one_factor_configs() -> list:
    configs = []

    for param_name, values in SENSITIVITY_VALUES.items():
        for value in values:
            config = deepcopy(BASELINE)

            if param_name == "lns_rf_range":
                config["lns_rf_min"], config["lns_rf_max"] = value
            elif param_name == "lns_accept_tuple":
                config["lns_D_pc_min"], config["lns_M_pc_max"], config["lns_E_pc_max"] = value
            elif param_name == "lns_it_max":
                config["lns_stopping_mode"] = "iterations"
                config["lns_it_max"] = value
            else:
                config[param_name] = value

            config["sensitivity_mode"] = "one_factor"
            config["varied_parameter"] = param_name
            config["varied_value"] = str(value)
            configs.append(config)

    return configs


def build_full_factorial_configs(max_configs: int | None = None) -> list:
    """
    WARNING: This can explode quickly. Use only for small sweeps.
    """
    keys = list(SENSITIVITY_VALUES.keys())
    value_lists = [SENSITIVITY_VALUES[k] for k in keys]
    configs = []

    for idx, values in enumerate(product(*value_lists)):
        if max_configs is not None and idx >= max_configs:
            break

        config = deepcopy(BASELINE)
        for key, value in zip(keys, values):
            if key == "lns_rf_range":
                config["lns_rf_min"], config["lns_rf_max"] = value
            elif key == "lns_accept_tuple":
                config["lns_D_pc_min"], config["lns_M_pc_max"], config["lns_E_pc_max"] = value
            elif key == "lns_it_max":
                # Keep full factorial mostly time-based unless explicitly changed.
                config["lns_it_max"] = value
            else:
                config[key] = value

        config["sensitivity_mode"] = "full_factorial"
        config["varied_parameter"] = "multiple"
        config["varied_value"] = "multiple"
        configs.append(config)

    return configs


def build_logger(scalability_module,
                 output_dir: Path,
                 config_id: str):
    logging_settings = {
        "enable_logging": True,
        "log_to_file": True,
        "log_to_console": False,
        "file_name": str(output_dir / "logs" / "{0}.log".format(config_id)),
        "file_mode": "w",
        "encoding": "utf-8",
        "level": "INFO",
        "format_str": "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        "date_format": "%Y/%m/%d %H:%M:%S",
    }
    ensure_dir(output_dir / "logs")
    return scalability_module.load_logger(logging_settings,
                                          "{0}_logger".format(config_id))


def summarize_numeric(values: list) -> tuple:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def count_internal_history_len(client_selector,
                               attribute_name: str) -> int:
    try:
        value = client_selector.get_attribute(attribute_name)
        if isinstance(value, dict):
            return len(value)
    except Exception:
        pass
    return 0


def run_single_configuration(scalability_module,
                             config: dict,
                             config_id: str,
                             output_dir: Path,
                             num_clients: int,
                             num_tasks: int,
                             num_rounds: int,
                             train_dataset_size: int,
                             test_dataset_size: int,
                             num_classes: int,
                             seed: int,
                             repetition: int) -> dict:
    base_settings = scalability_module.get_strategy_settings("MetaCS-FL", "")
    server_strategy_settings = deep_update_meta_cs_fl_settings(base_settings, config)

    client_selector = scalability_module.load_client_selector(server_strategy_settings, seed)

    samples_per_task = server_strategy_settings["samples_per_task"]
    data_privacy_approach = server_strategy_settings["data_privacy_approach"]["name"]

    max_train_samples_per_class = int((train_dataset_size / num_clients) / num_classes)
    max_test_samples_per_class = int((test_dataset_size / num_clients) / num_classes)

    candidate_clients = scalability_module.generate_candidate_clients(
        num_clients,
        num_classes,
        max_train_samples_per_class,
        max_test_samples_per_class,
        samples_per_task,
        data_privacy_approach,
    )

    clients_profiles = scalability_module.simulate_profiling_metrics(
        candidate_clients,
        [x * samples_per_task for x in server_strategy_settings["num_tasks_profile_training"]],
        [x * samples_per_task for x in server_strategy_settings["num_tasks_profile_testing"]],
        seed + repetition,
    )

    selected_clients_history = {}
    selected_clients_metrics_history = {}

    logger = build_logger(scalability_module, output_dir, config_id)

    root_output_folder = output_dir / "runs" / config_id
    ensure_dir(root_output_folder)

    kwargs = {
        "current_round": 1,
        "current_phase": "train",
        "candidate_clients": candidate_clients,
        "num_rounds": num_rounds,
        "base_learning_rate": 0.001,
        "base_batch_size": 32,
        "base_num_epochs": 5,
        "selected_clients_history": selected_clients_history,
        "selected_clients_metrics_history": selected_clients_metrics_history,
        "clients_profiles": clients_profiles,
        "time_limit": inf,
        "root_output_folder": root_output_folder,
        "logger": logger,
        "num_tasks": num_tasks,
        "samples_per_task": samples_per_task,
        "data_privacy_approach": data_privacy_approach,
        "use_async": False,
    }

    selection_wall_times = []
    selection_cpu_times = []
    selection_rss_peak_values = []
    selection_rss_delta_values = []
    selected_counts = []
    selected_set_changes = 0

    metaheuristic_runs = 0
    initial_solution_generations = 0

    previous_selected_ids = None
    previous_metaheuristic_history_len = 0
    previous_initial_solution_history_len = 0

    for server_round in range(1, num_rounds + 1):
        kwargs.update({
            "current_round": server_round,
            "selected_clients_history": selected_clients_history,
            "selected_clients_metrics_history": selected_clients_metrics_history,
        })

        selected_clients, resource_metrics = scalability_module._measure_selection_resources(
            client_selector,
            kwargs,
        )
        selected_clients = scalability_module.normalize_selected_clients(selected_clients)

        selection_wall_times.append(resource_metrics["selection_wall_time_seconds"])
        selection_cpu_times.append(resource_metrics["selection_cpu_time_seconds"])
        selection_rss_peak_values.append(resource_metrics["selection_rss_peak_mb"])
        selection_rss_delta_values.append(resource_metrics["selection_rss_delta_mb"])

        selected_ids = set(scalability_module.extract_selected_client_ids(selected_clients))
        selected_counts.append(len(selected_ids))

        if previous_selected_ids is not None and selected_ids != previous_selected_ids:
            selected_set_changes += 1
        previous_selected_ids = selected_ids

        current_metaheuristic_history_len = count_internal_history_len(
            client_selector,
            "_metaheuristic_execution_history",
        )
        current_initial_solution_history_len = count_internal_history_len(
            client_selector,
            "_initial_solution_generation_history",
        )

        if current_metaheuristic_history_len > previous_metaheuristic_history_len:
            metaheuristic_runs += current_metaheuristic_history_len - previous_metaheuristic_history_len

        if current_initial_solution_history_len > previous_initial_solution_history_len:
            initial_solution_generations += (
                current_initial_solution_history_len - previous_initial_solution_history_len
            )

        previous_metaheuristic_history_len = current_metaheuristic_history_len
        previous_initial_solution_history_len = current_initial_solution_history_len

        train_metrics = scalability_module.simulate_clients_metrics_for_round(
            selected_clients,
            clients_profiles,
            server_round,
            phase="train",
            new_client_selection_trigger_probability=config["new_client_selection_trigger_probability"],
        )

        test_metrics = scalability_module.simulate_clients_metrics_for_round(
            selected_clients,
            clients_profiles,
            server_round,
            phase="test",
            new_client_selection_trigger_probability=config["new_client_selection_trigger_probability"],
        )

        selected_clients_history[server_round] = {"train": selected_clients}

        if server_round not in selected_clients_metrics_history:
            selected_clients_metrics_history[server_round] = {}

        selected_clients_metrics_history[server_round]["train"] = {
            "clients_metrics_dicts": train_metrics
        }
        selected_clients_metrics_history[server_round]["test"] = {
            "clients_metrics_dicts": test_metrics
        }

    wall_mean, wall_std = summarize_numeric(selection_wall_times)
    cpu_mean, cpu_std = summarize_numeric(selection_cpu_times)
    selected_mean, selected_std = summarize_numeric(selected_counts)

    row = {
        "config_id": config_id,
        "repetition": repetition,
        "sensitivity_mode": config["sensitivity_mode"],
        "varied_parameter": config["varied_parameter"],
        "varied_value": config["varied_value"],

        "num_clients": num_clients,
        "num_tasks": num_tasks,
        "num_rounds": num_rounds,

        "selection_wall_time_total_seconds": sum(selection_wall_times),
        "selection_wall_time_avg_seconds": wall_mean,
        "selection_wall_time_std_seconds": wall_std,

        "selection_cpu_time_total_seconds": sum(selection_cpu_times),
        "selection_cpu_time_avg_seconds": cpu_mean,
        "selection_cpu_time_std_seconds": cpu_std,

        "selection_rss_peak_avg_mb": mean(selection_rss_peak_values) if selection_rss_peak_values else 0.0,
        "selection_rss_peak_max_mb": max(selection_rss_peak_values) if selection_rss_peak_values else 0.0,
        "selection_rss_delta_avg_mb": mean(selection_rss_delta_values) if selection_rss_delta_values else 0.0,
        "selection_rss_delta_max_mb": max(selection_rss_delta_values) if selection_rss_delta_values else 0.0,

        "avg_selected_clients_per_round": selected_mean,
        "std_selected_clients_per_round": selected_std,
        "selected_set_changes": selected_set_changes,

        "metaheuristic_runs": metaheuristic_runs,
        "initial_solution_generations": initial_solution_generations,

        "train_minimum_diversity_score": config["train_minimum_diversity_score"],
        "train_maximum_relative_drop": config["train_maximum_relative_drop"],
        "train_diversity_window": config["train_diversity_window"],

        "makespan_maximum_increase": config["makespan_maximum_increase"],
        "makespan_window": config["makespan_window"],

        "energy_maximum_increase": config["energy_maximum_increase"],
        "energy_window": config["energy_window"],

        "accuracy_maximum_decrease": config["accuracy_maximum_decrease"],
        "accuracy_window": config["accuracy_window"],

        "test_minimum_diversity_score": config["test_minimum_diversity_score"],
        "test_diversity_window": config["test_diversity_window"],

        "max_rounds_same_initial_solution": config["max_rounds_same_initial_solution"],

        "objective_weights_name": config["objective_weights_name"],
        "M_weight": OBJECTIVE_WEIGHT_CONFIGS[config["objective_weights_name"]]["M_weight"],
        "E_weight": OBJECTIVE_WEIGHT_CONFIGS[config["objective_weights_name"]]["E_weight"],
        "D_weight": OBJECTIVE_WEIGHT_CONFIGS[config["objective_weights_name"]]["D_weight"],
        "K_weight": OBJECTIVE_WEIGHT_CONFIGS[config["objective_weights_name"]]["K_weight"],
        "U_weight": OBJECTIVE_WEIGHT_CONFIGS[config["objective_weights_name"]]["U_weight"],

        "alpha": config["alpha"],
        "beta": config["beta"],
        "q": config["q"],

        "lns_stopping_mode": config["lns_stopping_mode"],
        "lns_t_max": config["lns_t_max"],
        "lns_it_max": config["lns_it_max"],

        "lns_sf": config["lns_sf"],
        "lns_rf_min": config["lns_rf_min"],
        "lns_rf_max": config["lns_rf_max"],

        "lns_D_pc_min": config["lns_D_pc_min"],
        "lns_M_pc_max": config["lns_M_pc_max"],
        "lns_E_pc_max": config["lns_E_pc_max"],

        "new_client_selection_trigger_probability": config["new_client_selection_trigger_probability"],
    }

    return row


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("results/static_client_availability/sensitivity_results"),
                        help="Output directory.")
    parser.add_argument("--num-clients",
                        type=int,
                        default=100)
    parser.add_argument("--num-tasks",
                        type=int,
                        default=20000)
    parser.add_argument("--num-rounds",
                        type=int,
                        default=50)
    parser.add_argument("--train-dataset-size",
                        type=int,
                        default=50000)
    parser.add_argument("--test-dataset-size",
                        type=int,
                        default=10000)
    parser.add_argument("--num-classes",
                        type=int,
                        default=10)
    parser.add_argument("--seed",
                        type=int,
                        default=777)
    parser.add_argument("--num-repetitions",
                        type=int,
                        default=1)
    parser.add_argument("--full-factorial",
                        action="store_true")
    parser.add_argument("--max-full-factorial-configs",
                        type=int,
                        default=None)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    scalability_module = load_scalability_module()

    if args.full_factorial:
        configs = build_full_factorial_configs(args.max_full_factorial_configs)
    else:
        configs = build_one_factor_configs()

    results_csv = args.output_dir / "metacsfl_sensitivity_results.csv"

    print("Number of configurations: {0}".format(len(configs)))
    print("Output CSV: {0}".format(results_csv))

    for config_idx, config in enumerate(configs):
        for repetition in range(1, args.num_repetitions + 1):
            config_id = "{0:04d}_{1}_{2}_rep{3}".format(
                config_idx,
                config["varied_parameter"],
                str(config["varied_value"]).replace(" ", "").replace("/", "_"),
                repetition,
            )

            print("[RUN] {0}".format(config_id))

            try:
                row = run_single_configuration(
                    scalability_module=scalability_module,
                    config=config,
                    config_id=config_id,
                    output_dir=args.output_dir,
                    num_clients=args.num_clients,
                    num_tasks=args.num_tasks,
                    num_rounds=args.num_rounds,
                    train_dataset_size=args.train_dataset_size,
                    test_dataset_size=args.test_dataset_size,
                    num_classes=args.num_classes,
                    seed=args.seed,
                    repetition=repetition,
                )

                df = DataFrame([row])
                write_header = not results_csv.exists()
                df.to_csv(results_csv, mode="a", header=write_header, index=False)

            except Exception as e:
                print("[SENSITIVITY ERROR]")
                print("config_id: {0}".format(config_id))
                print("exception: {0}".format(repr(e)))
                print(format_exc())
                raise

    print("Done.")


if __name__ == "__main__":
    main()
