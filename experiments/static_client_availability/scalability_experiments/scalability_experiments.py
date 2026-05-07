from argparse import ArgumentParser
from copy import deepcopy
from os import getpid
from pathlib import Path
from random import choice, random, seed as random_seed, uniform
from threading import Event, Thread
from time import sleep, time
from traceback import format_exc

from psutil import Process
from pandas import DataFrame
from numpy import empty, inf, int64, uint8
from numpy.random import laplace, randint

from metacs_fl.client_selector.metacsfl import MetaCSFL
from metacs_fl.client_selector.divfl import DivFL
from metacs_fl.client_selector.ecsm import ECSM
from metacs_fl.client_selector.oort import Oort
from metacs_fl.client_selector.random import Random
from metacs_fl.client_selector.sbacpad_2024 import SBACPAD2024
from metacs_fl.utils.config_parser_util import parse_config_file
from metacs_fl.utils.dataset_loader_util import get_task_assignment_capacities, get_classes_distribution
from metacs_fl.utils.logger_util import load_logger


def _measure_selection_resources(client_selector: any,
                                 kwargs: dict,
                                 rss_sample_interval_in_seconds: float = 0.01) -> tuple:
    process = Process(getpid())
    rss_samples_bytes = []
    stop_event = Event()

    def _rss_sampler() -> None:
        while not stop_event.is_set():
            try:
                rss_samples_bytes.append(process.memory_info().rss)
            except Exception:
                pass
            sleep(rss_sample_interval_in_seconds)
        try:
            rss_samples_bytes.append(process.memory_info().rss)
        except Exception:
            pass

    rss_start_bytes = process.memory_info().rss
    cpu_times_start = process.cpu_times()
    sampler_thread = Thread(target=_rss_sampler, daemon=True)
    sampler_thread.start()
    start_time = time()
    try:
        selected_clients = client_selector.run_client_selection_procedure(**kwargs)
    finally:
        end_time = time()
        stop_event.set()
        sampler_thread.join()
    cpu_times_end = process.cpu_times()
    cpu_time_seconds = ((cpu_times_end.user + cpu_times_end.system)
                        - (cpu_times_start.user + cpu_times_start.system))
    rss_peak_bytes = max(rss_samples_bytes) if rss_samples_bytes else rss_start_bytes
    rss_avg_bytes = (sum(rss_samples_bytes) / len(rss_samples_bytes)) if rss_samples_bytes else rss_start_bytes
    resource_metrics = {"selection_wall_time_seconds": end_time - start_time,
                        "selection_cpu_time_seconds": cpu_time_seconds,
                        "selection_rss_start_mb": rss_start_bytes / (1024 ** 2),
                        "selection_rss_peak_mb": rss_peak_bytes / (1024 ** 2),
                        "selection_rss_avg_mb": rss_avg_bytes / (1024 ** 2)}
    resource_metrics["selection_rss_delta_mb"] = resource_metrics["selection_rss_peak_mb"] - resource_metrics["selection_rss_start_mb"]
    return selected_clients, resource_metrics


def normalize_selected_clients(selected_clients: any) -> dict:
    if isinstance(selected_clients, dict):
        return selected_clients
    if hasattr(selected_clients, "result"):
        return normalize_selected_clients(selected_clients.result())
    if isinstance(selected_clients, list):
        if not selected_clients:
            return {}
        first_item = selected_clients[0]
        if hasattr(first_item, "result"):
            first_item = first_item.result()
        if isinstance(first_item, dict):
            first_value = next(iter(first_item.values()))
            if isinstance(first_value, dict) and "selected_clients" in first_value:
                return first_value["selected_clients"]
            if "selected_clients" in first_item:
                return first_item["selected_clients"]
            return first_item
        return first_item
    return selected_clients


def generate_candidate_clients(num_clients: int,
                               num_classes: int,
                               max_train_samples_per_class: int,
                               max_test_samples_per_class: int,
                               samples_per_task: int,
                               data_privacy_approach: str) -> dict:
    x_train = empty((max_train_samples_per_class * num_classes, 32, 32, 3), dtype=uint8)
    x_test = empty((max_test_samples_per_class * num_classes, 32, 32, 3), dtype=uint8)
    y_train = randint(0, 10, size=(max_train_samples_per_class * num_classes,), dtype=int64)
    y_test = randint(0, 10, size=(max_test_samples_per_class * num_classes,), dtype=int64)
    task_assignment_capacities_settings = {"task_assignment_capacities_train": [],
                                           "task_assignment_capacities_test": [],
                                           "lower_bound": 0,
                                           "upper_bound": "client_capacity",
                                           "step": 50}
    # Initialize the dictionary of candidate clients.
    candidate_clients = {}
    for i in range(num_clients):
        client_id = "client_{0}".format(i)
        # Get the task assignment capacities.
        task_assignment_capacities_train, task_assignment_capacities_test \
            = get_task_assignment_capacities(x_train, x_test, task_assignment_capacities_settings, samples_per_task)
        # Get the tasks' occurrence per class.
        client_tasks_per_class_train = get_classes_distribution(y_train)
        client_tasks_per_class_test = get_classes_distribution(y_test)
        # Initialize the client map.
        client_map = {}
        # Non-private clients: dict of class -> tasks.
        if data_privacy_approach == "Non_Private":
            client_map = {"client_proxy": "proxy_{0}".format(client_id),
                          "client_task_assignment_capacities_train": task_assignment_capacities_train,
                          "client_task_assignment_capacities_test": task_assignment_capacities_test,
                          "client_current_download_bandwidth_in_bytes_per_second": 1e6,
                          "client_current_upload_bandwidth_in_bytes_per_second": 1e6,
                          "client_current_latency_in_milliseconds": 50,
                          "client_mean_power_consumption_idle_mode": 5.0,
                          "client_remaining_battery_energy": 1000.0,
                          "client_tasks_per_class_train": client_tasks_per_class_train,
                          "client_tasks_per_class_test": client_tasks_per_class_test}
        # Differentially private clients: histogram + index mapping.
        elif data_privacy_approach == "Differentially_Private":
            class_index_map = {cls: cls for cls in range(num_classes)}
            epsilon = 0.1 # privacy budget
            sensitivity = 1.0
            client_dp_histogram_train = {}
            client_dp_histogram_test = {}
            for cls in range(num_classes):
                true_train = client_tasks_per_class_train.get(str(cls), 0)
                noise_train = laplace(0, sensitivity / epsilon)
                client_dp_histogram_train[cls] = max(0, true_train + int(round(noise_train)))
                true_test = client_tasks_per_class_test.get(str(cls), 0)
                noise_test = laplace(0, sensitivity / epsilon)
                client_dp_histogram_test[cls] = max(0, true_test + int(round(noise_test)))
            client_map = {"client_proxy": "proxy_{0}".format(client_id),
                          "client_task_assignment_capacities_train": task_assignment_capacities_train,
                          "client_task_assignment_capacities_test": task_assignment_capacities_test,
                          "client_current_download_bandwidth_in_bytes_per_second": 1e6,
                          "client_current_upload_bandwidth_in_bytes_per_second": 1e6,
                          "client_current_latency_in_milliseconds": 50,
                          "client_mean_power_consumption_idle_mode": 5.0,
                          "client_remaining_battery_energy": 1000.0,
                          "class_index_map": class_index_map,
                          "client_dp_histogram_train": client_dp_histogram_train,
                          "client_dp_histogram_test": client_dp_histogram_test}
        candidate_clients[client_id] = client_map
    return candidate_clients


def simulate_profiling_metrics(candidate_clients: dict,
                               num_samples_profile_training_list: list,
                               num_samples_profile_testing_list: list,
                               seed = None) -> dict:
    if seed is not None:
        random_seed(seed)
    clients_profiles = {}
    for client_id, v in candidate_clients.items():
        profile_train = {}
        profile_test = {}
        # Hardware / network profile.
        bw_down = uniform(4e7, 8e7)
        bw_up = uniform(2e7, 4e7)
        bw_mem = choice([1.28e10, 2.98e10])
        cpu_freq = choice([1.2e9, 2.0e9])
        nc_cpu = 4
        rtt_down = uniform(60, 90)
        rtt_up = uniform(55, 95)
        # Hyperparameters.
        learning_rate = choice([0.001])
        batch_size = choice([32])
        epochs = choice([5])
        # Training profiles.
        for n_samples in num_samples_profile_training_list:
            training_time = uniform(70, 120)
            computation_time = training_time * uniform(0.85, 0.92)
            upload_time = uniform(2.5, 4.0)
            download_time = uniform(3.0, 4.0)
            computation_energy = uniform(900, 1200)
            upload_energy = uniform(20, 35)
            download_energy = uniform(20, 30)
            training_energy = computation_energy + upload_energy + download_energy
            acc_train = uniform(0.88, 0.95)
            loss_train = uniform(0.12, 0.25)
            profile_train[n_samples] = {"ack_down": 44,
                                        "ack_up": 44,
                                        "batch_size": batch_size,
                                        "bs_train_i": batch_size,
                                        "bw_down_i": bw_down,
                                        "bw_mem_i": bw_mem,
                                        "bw_up_i": bw_up,
                                        "comm_round": 1,
                                        "computation_energy_in_joules": computation_energy,
                                        "computation_time_in_seconds": computation_time,
                                        "download_energy_in_joules": download_energy,
                                        "download_time_in_seconds": download_time,
                                        "ds_train_i": n_samples,
                                        "e_i": epochs,
                                        "epochs": epochs,
                                        "fpo_train_m": 239542553,
                                        "fpocc_m_i_train": uniform(8.0, 12.0),
                                        "fpp_m": 4.0,
                                        "fq_cpu_i": cpu_freq,
                                        "is_train_i": uniform(50, 70),
                                        "learning_rate": learning_rate,
                                        "loss": loss_train,
                                        "mpc_comp_i": uniform(12, 18),
                                        "mpc_idle_i": uniform(2.0, 3.0),
                                        "mpc_recv_i": uniform(7.0, 10.0),
                                        "mpc_send_i": uniform(9.0, 12.0),
                                        "ms_train_i": 152.0,
                                        "mt_m_train": uniform(0.08, 0.12),
                                        "nc_cpu_i": nc_cpu,
                                        "np_m": 143373.0,
                                        "num_examples": n_samples,
                                        "num_lost_packets": 0,
                                        "num_packets": 393,
                                        "retransmission_time_in_seconds": 0.0,
                                        "rtt_down_i": rtt_down,
                                        "rtt_up_i": rtt_up,
                                        "sparse_categorical_accuracy": acc_train,
                                        "total_float_ops": int(5.5e11),
                                        "training_energy_in_joules": training_energy,
                                        "training_time_in_seconds": training_time,
                                        "upload_energy_in_joules": upload_energy,
                                        "upload_time_in_seconds": upload_time}
        # Testing profiles.
        for n_samples in num_samples_profile_testing_list:
            testing_time = uniform(30, 70)
            computation_time_test = testing_time * uniform(0.8, 0.9)
            testing_energy = uniform(150, 350)
            computation_energy_test = testing_energy * uniform(0.75, 0.85)
            acc_test = uniform(0.45, 0.6)
            loss_test = uniform(0.7, 0.9)
            profile_test[n_samples] = {"ack_down": 44,
                                       "ack_up": 44,
                                       "batch_size": batch_size,
                                       "bs_test_i": batch_size,
                                       "bw_down_i": bw_down,
                                       "bw_mem_i": bw_mem,
                                       "bw_up_i": bw_up,
                                       "comm_round": 1,
                                       "computation_energy_in_joules": computation_energy_test,
                                       "computation_time_in_seconds": computation_time_test,
                                       "download_energy_in_joules": uniform(10, 25),
                                       "download_time_in_seconds": uniform(2.5, 4.5),
                                       "ds_test_i": n_samples,
                                       "fpo_test_m": 79384270,
                                       "fpocc_m_i_test": uniform(3.0, 5.0),
                                       "fpp_m": 4.0,
                                       "fq_cpu_i": cpu_freq,
                                       "is_test_i": uniform(20, 40),
                                       "loss": loss_test,
                                       "mpc_comp_i": uniform(4.0, 7.0),
                                       "mpc_idle_i": uniform(1.5, 2.5),
                                       "mpc_recv_i": uniform(4.0, 6.0),
                                       "mpc_send_i": uniform(5.0, 7.0),
                                       "ms_test_i": 136.0,
                                       "mt_m_test": uniform(0.04, 0.07),
                                       "nc_cpu_i": nc_cpu,
                                       "np_m": 143373.0,
                                       "num_examples": n_samples,
                                       "num_lost_packets": 0,
                                       "num_packets": 1,
                                       "retransmission_time_in_seconds": 0.0,
                                       "rtt_down_i": rtt_down,
                                       "rtt_up_i": rtt_up,
                                       "sparse_categorical_accuracy": acc_test,
                                       "testing_energy_in_joules": testing_energy,
                                       "testing_time_in_seconds": testing_time,
                                       "total_float_ops": int(7.6e9),
                                       "upload_energy_in_joules": uniform(10, 25),
                                       "upload_time_in_seconds": uniform(3.0, 6.0)}
        # Store the client profiles.
        clients_profiles[client_id] = {"profiled_at_round": 1,
                                       "train": profile_train,
                                       "test": profile_test}
    return clients_profiles


def simulate_clients_metrics_for_round(selected_clients: dict,
                                       clients_profiles: dict,
                                       server_round: int,
                                       phase: str = "train",
                                       variability: bool = True,
                                       new_client_selection_trigger_probability: float = 0) -> list:
    """
    Probabilistic variance model:
    - With a given probability, metrics cross MetaCS-FL thresholds.
    - Otherwise, metrics remain stable
    """
    assert phase in ["train", "test"]
    clients_metrics_dicts = []
    # Decide if this round is unstable.
    trigger_event = variability and (random() < new_client_selection_trigger_probability)
    # Shock magnitudes.
    if trigger_event:
        # Cross thresholds.
        round_time_shock = 1 + uniform(0.25, 0.60)
        round_energy_shock = 1 + uniform(0.25, 0.55)
        if phase == "test":
            round_accuracy_drop = uniform(0.15, 0.40)
        else:
            round_accuracy_drop = uniform(0.10, 0.30)
    else:
        # Stay below thresholds.
        round_time_shock = 1 + uniform(-0.05, 0.08)
        round_energy_shock = 1 + uniform(-0.05, 0.08)
        round_accuracy_drop = uniform(0.0, 0.04)
    # Phase-specific metric keys.
    if phase == "train":
        runtime_keys = ["training_time_in_seconds",
                        "computation_time_in_seconds",
                        "download_time_in_seconds",
                        "upload_time_in_seconds"]
        energy_keys = ["training_energy_in_joules",
                       "computation_energy_in_joules",
                       "download_energy_in_joules",
                       "upload_energy_in_joules"]
    else:
        runtime_keys = ["testing_time_in_seconds",
                        "computation_time_in_seconds",
                        "download_time_in_seconds",
                        "upload_time_in_seconds"]
        energy_keys = ["testing_energy_in_joules",
                       "computation_energy_in_joules",
                       "download_energy_in_joules",
                       "upload_energy_in_joules"]
    # Per-client metrics.
    for client_id, client_info in selected_clients.items():
        profile_phase = clients_profiles[client_id][phase]
        # Select the closest profiled sample size
        if "client_num_samples_scheduled" in client_info:
            n_samples = client_info["client_num_samples_scheduled"]
        elif "client_num_tasks_scheduled" in client_info:
            n_samples = client_info["client_num_tasks_scheduled"]
        else:
            raise KeyError("Neither 'client_num_samples_scheduled' nor 'client_num_tasks_scheduled' was found in client_info.")
        closest_x = min(profile_phase.keys(), key=lambda x: abs(x - n_samples))
        metrics = deepcopy(profile_phase[closest_x])
        metrics["comm_round"] = server_round
        metrics["num_examples"] = n_samples
        if variability:
            # Time metrics.
            for key in runtime_keys:
                if key in metrics and isinstance(metrics[key], float):
                    local_noise = uniform(-0.05, 0.10)
                    metrics[key] *= (1 + local_noise) * round_time_shock
            # Energy metrics.
            for key in energy_keys:
                if key in metrics and isinstance(metrics[key], float):
                    local_noise = uniform(-0.05, 0.10)
                    metrics[key] *= (1 + local_noise) * round_energy_shock
            # Loss.
            if "loss" in metrics:
                metrics["loss"] *= 1 + (uniform(0.05, 0.40) if trigger_event else uniform(-0.03, 0.05))
            # Accuracy.
            if "sparse_categorical_accuracy" in metrics:
                metrics["sparse_categorical_accuracy"] *= (1 - round_accuracy_drop)
                metrics["sparse_categorical_accuracy"] = max(0.0, metrics["sparse_categorical_accuracy"])
        clients_metrics_dicts.append({client_id: metrics})
    return clients_metrics_dicts


def extract_selected_client_ids(selected_clients: list | dict) -> list:
    selected_clients = normalize_selected_clients(selected_clients)
    if isinstance(selected_clients, dict):
        return list(selected_clients.keys())
    if isinstance(selected_clients, list):
        return selected_clients
    return []


def get_strategy_settings(strategy_name: str,
                          substrategy_name: str) -> dict:
    server_strategy_settings = {}
    match strategy_name:
        case "SBAC-PAD_2024":
            match substrategy_name:
                case "Random":
                    server_strategy_settings = {'strategy': 'SBAC-PAD_2024',
                                                'seed': 777,
                                                'samples_per_task': 1,
                                                'num_tasks_profile_training': [200],
                                                'num_tasks_profile_testing': [200],
                                                'num_tasks_training': 20000,
                                                'num_tasks_testing': 10000,
                                                'client_selector_training': {'seed': 777,
                                                                             'samples_per_task': 1,
                                                                             'num_tasks_training': 20000,
                                                                             'num_tasks_testing': 10000,
                                                                             'fraction_clients_training': 0.49,
                                                                             'fraction_clients_testing': 1.0,
                                                                             'model_aggregator': 'FedAvg',
                                                                             'metrics_aggregator': 'Weighted_Average',
                                                                             'name': 'Random'},
                                                'client_selector_testing': {'seed': 777,
                                                                            'samples_per_task': 1,
                                                                            'num_tasks_training': 20000,
                                                                            'num_tasks_testing': 10000,
                                                                            'fraction_clients_training': 0.49,
                                                                            'fraction_clients_testing': 1.0,
                                                                            'model_aggregator': 'FedAvg',
                                                                            'metrics_aggregator': 'Weighted_Average',
                                                                            'name': 'Random'},
                                                'select_clients_for_immediate_next_round_while_executing_current_training': False,
                                                'select_clients_for_immediate_next_round_while_executing_current_testing': False,
                                                'use_async': False,
                                                'model_aggregator': {'inplace_aggregation': False, 'name': 'FedAvg'},
                                                'metrics_aggregator': 'Weighted_Average',
                                                'history_checker': 'All_Previous_Rounds',
                                                'query_clients_data_distribution': True,
                                                'data_privacy_approach': {'true_presence_probability': 1.0,
                                                                          'presence_cutoff': 0.5,
                                                                          'epsilon': 0.1,
                                                                          'name': 'Differentially_Private'}}
                case x if x in ["MEC", "ECMTC"]:
                    server_strategy_settings = {'strategy': 'SBAC-PAD_2024',
                                                'seed': 777,
                                                'samples_per_task': 1,
                                                'num_tasks_profile_training': [200],
                                                'num_tasks_profile_testing': [200],
                                                'num_tasks_training': 20000,
                                                'num_tasks_testing': 10000,
                                                'client_selector_training': {'name': '{0}'.format(x)},
                                                'client_selector_testing': {'seed': 777,
                                                                            'fraction_clients_training': 0.1,
                                                                            'fraction_clients_testing': 1.0,
                                                                            'model_aggregator': 'FedAvg',
                                                                            'metrics_aggregator': 'Weighted_Average',
                                                                            'name': 'Random'},
                                                'select_clients_for_immediate_next_round_while_executing_current_training': False,
                                                'select_clients_for_immediate_next_round_while_executing_current_testing': False,
                                                'use_async': False,
                                                'model_aggregator': {'inplace_aggregation': False, 'name': 'FedAvg'},
                                                'metrics_aggregator': 'Weighted_Average',
                                                'history_checker': 'All_Previous_Rounds',
                                                'query_clients_data_distribution': True,
                                                'data_privacy_approach': {'true_presence_probability': 1.0,
                                                                          'presence_cutoff': 0.5,
                                                                          'epsilon': 0.1,
                                                                          'name': 'Differentially_Private'}}
        case "Oort":
            server_strategy_settings = {'strategy': 'Oort',
                                        'seed': 777,
                                        'samples_per_task': 1,
                                        'num_tasks_profile_training': [200],
                                        'num_tasks_profile_testing': [200],
                                        'num_tasks_training': 20000,
                                        'num_tasks_testing': 10000,
                                        'client_selector_training': {'seed': 777,
                                                                     'samples_per_task': 1,
                                                                     'num_tasks_profile_training': [200],
                                                                     'num_tasks_profile_testing': [200],
                                                                     'num_tasks_training': 20000,
                                                                     'num_tasks_testing': 10000,
                                                                     'client_selector_training': 'Oort',
                                                                     'client_selector_testing': 'Random',
                                                                     'model_aggregator': 'FedAvg',
                                                                     'metrics_aggregator': 'Weighted_Average',
                                                                     'query_clients_data_distribution': True,
                                                                     'data_privacy_approach': 'Differentially_Private',
                                                                     'oort_mode': 'binary',
                                                                     'alpha': 2.0,
                                                                     'explore_frac': 0.9,
                                                                     'pacer_step': 20,
                                                                     'pacer_delta': 0.1,
                                                                     'name': 'Oort'},
                                        'client_selector_testing': {'seed': 777,
                                                                    'fraction_clients_training': 0.2,
                                                                    'fraction_clients_testing': 1.0,
                                                                    'model_aggregator': 'FedAvg',
                                                                    'metrics_aggregator': 'Weighted_Average',
                                                                    'name': 'Random'},
                                        'model_aggregator': {'inplace_aggregation': False, 'name': 'FedAvg'},
                                        'metrics_aggregator': 'Weighted_Average',
                                        'query_clients_data_distribution': True,
                                        'data_privacy_approach': {'true_presence_probability': 1.0,
                                                                  'presence_cutoff': 0.5,
                                                                  'epsilon': 0.1,
                                                                  'name': 'Differentially_Private'},
                                        'oort_mode': 'binary',
                                        'alpha': 2.0,
                                        'explore_frac': 0.9,
                                        'pacer_step': 20,
                                        'pacer_delta': 0.1}
        case "DivFL":
            server_strategy_settings = {'strategy': 'DivFL',
                                        'seed': 777,
                                        'samples_per_task': 1,
                                        'num_tasks_profile_training': [200],
                                        'num_tasks_profile_testing': [200],
                                        'num_tasks_training': 20000,
                                        'num_tasks_testing': 10000,
                                        'client_selector_training': {'seed': 777,
                                                                     'samples_per_task': 1,
                                                                     'num_tasks_profile_training': [200],
                                                                     'num_tasks_profile_testing': [200],
                                                                     'num_tasks_training': 20000,
                                                                     'num_tasks_testing': 10000,
                                                                     'client_selector_training': 'DivFL',
                                                                     'client_selector_testing': 'Random',
                                                                     'model_aggregator': 'FedAvg',
                                                                     'metrics_aggregator': 'Weighted_Average',
                                                                     'query_clients_data_distribution': True,
                                                                     'data_privacy_approach': 'Differentially_Private',
                                                                     'name': 'DivFL'},
                                        'client_selector_testing': {'seed': 777,
                                                                    'fraction_clients_training': 0.2,
                                                                    'fraction_clients_testing': 1.0,
                                                                    'model_aggregator': 'FedAvg',
                                                                    'metrics_aggregator': 'Weighted_Average',
                                                                    'name': 'Random'},
                                        'model_aggregator': {'inplace_aggregation': False, 'name': 'FedAvg'},
                                        'metrics_aggregator': 'Weighted_Average',
                                        'query_clients_data_distribution': True,
                                        'data_privacy_approach': {'true_presence_probability': 1.0,
                                                                  'presence_cutoff': 0.5,
                                                                  'epsilon': 0.1,
                                                                  'name': 'Differentially_Private'}}
        case "ECSM":
            server_strategy_settings = {'strategy': 'ECSM',
                                        'seed': 777,
                                        'samples_per_task': 1,
                                        'num_tasks_profile_training': [200],
                                        'num_tasks_profile_testing': [200],
                                        'num_tasks_training': 20000,
                                        'num_tasks_testing': 10000,
                                        'client_selector_training': {'seed': 777,
                                                                     'samples_per_task': 1,
                                                                     'num_tasks_profile_training': [200],
                                                                     'num_tasks_profile_testing': [200],
                                                                     'num_tasks_training': 20000,
                                                                     'num_tasks_testing': 10000,
                                                                     'client_selector_training': 'ECSM',
                                                                     'client_selector_testing': 'Random',
                                                                     'model_aggregator': 'FedAvg',
                                                                     'metrics_aggregator': 'Weighted_Average',
                                                                     'query_clients_data_distribution': True,
                                                                     'data_privacy_approach': 'Differentially_Private',
                                                                     'accuracy_ratio': 20,
                                                                     'reputation_ratio': 70,
                                                                     'random_ratio': 10,
                                                                     'name': 'ECSM'},
                                        'client_selector_testing': {'seed': 777,
                                                                    'fraction_clients_training': 0.2,
                                                                    'fraction_clients_testing': 1.0,
                                                                    'model_aggregator': 'FedAvg',
                                                                    'metrics_aggregator': 'Weighted_Average',
                                                                    'name': 'Random'},
                                        'model_aggregator': {'inplace_aggregation': False, 'name': 'FedAvg'},
                                        'metrics_aggregator': 'Weighted_Average',
                                        'query_clients_data_distribution': True,
                                        'data_privacy_approach': {'true_presence_probability': 1.0,
                                                                  'presence_cutoff': 0.5,
                                                                  'epsilon': 0.1,
                                                                  'name': 'Differentially_Private'},
                                        'accuracy_ratio': 20,
                                        'reputation_ratio': 70,
                                        'random_ratio': 10}
        case "MetaCS-FL":
            server_strategy_settings = {'strategy': 'MetaCS-FL',
                                        'seed': 777,
                                        'samples_per_task': 1,
                                        'num_tasks_profile_training': [200],
                                        'num_tasks_profile_testing': [200],
                                        'num_tasks_training': 20000,
                                        'num_tasks_testing': 10000,
                                        'new_client_selection_criteria_training': {'criterion_1': {'name': 'client_diversity_score_training', 'maximum_relative_drop': 0.2, 'num_past_rounds': 3},
                                                                                   'criterion_2': {'name': 'makespan_percentage_increase_training', 'maximum_increase': 0.1, 'num_past_rounds': 2},
                                                                                   'criterion_3': {'name': 'energy_consumption_percentage_increase_training', 'maximum_increase': 0.1, 'num_past_rounds': 2},
                                                                                   'criterion_4': {'name': 'accuracy_percentage_decrease_testing', 'maximum_decrease': 0.1, 'num_past_rounds': 2}},
                                        'new_initial_solution_criteria_training': {'criterion_1': {'name': 'max_rounds_same_initial_solution', 'maximum_rounds': 10}},
                                        'initial_solution_generator_training': {'name': 'ECMTC'},
                                        'run_metaheuristic_training': True,
                                        'new_client_selection_criteria_testing': {'criterion_1': {'name': 'client_diversity_score_testing', 'minimum_score': 0.5, 'num_past_rounds': 5}},
                                        'new_initial_solution_criteria_testing': {'criterion_1': {'name': 'max_rounds_same_initial_solution', 'maximum_rounds': 10}},
                                        'initial_solution_generator_testing': {'seed': 777, 'fraction_clients_training': 0.2, 'fraction_clients_testing': 1.0, 'model_aggregator': 'FedAvg', 'metrics_aggregator': 'Weighted_Average', 'name': 'Random'},
                                        'run_metaheuristic_testing': False,
                                        'initial_solution_tasks_distribution_scheme': 'locally_balanced',
                                        'metaheuristic_solution_tasks_distribution_scheme': 'globally_balanced',
                                        'objective_function': {'obj_func_weights': {'M_weight': 0.05, 'E_weight': 0.05, 'D_weight': 0.3, 'K_weight': 0.2, 'U_weight': 0.4},
                                                               'alpha': 0.35,
                                                               'beta': 0.7,
                                                               'q': 3},
                                        'metaheuristic': {'write_traces_to_output_file': True,
                                                          'stopping_criteria': {'t_max': 10, 'name': 'LNS_Stop_Elapsed_Time'},
                                                          'destroy_approach': {'sf': 0.3, 'rf_min': 0.15, 'rf_max': 0.75},
                                                          'repair_approach': {},
                                                          'accept_criteria': {'D_pc_min': 0.005, 'M_pc_max': 0.25, 'E_pc_max': 0.25, 'bl_min': 0},
                                                          'name': 'LNS'},
                                        'model_aggregator': {'inplace_aggregation': False, 'name': 'FedAvg'},
                                        'metrics_aggregator': 'Weighted_Average', 'undesired_metrics_for_fit_aggregation': [],
                                        'undesired_metrics_for_evaluate_aggregation': [],
                                        'query_clients_data_distribution': True,
                                        'data_privacy_approach': {'true_presence_probability': 1.0,
                                                                  'presence_cutoff': 0.5,
                                                                  'epsilon': 0.1,
                                                                  'name': 'Differentially_Private'}}
    return server_strategy_settings


def load_client_selector(server_strategy_settings: dict,
                         seed=None) -> any:
    client_selector = None
    strategy = server_strategy_settings["strategy"]
    match strategy:
        case "Random":
            # Instantiate the Random's client selector.
            client_selector = Random(server_strategy_settings, seed)
        case "SBAC-PAD_2024":
            # Instantiate the SBAC-PAD_2024's client selector.
            client_selector = SBACPAD2024(server_strategy_settings, seed)
        case "Oort":
            # Instantiate the Oort's client selector.
            client_selector = Oort(server_strategy_settings, seed)
        case "DivFL":
            # Instantiate the DivFL's client selector.
            client_selector = DivFL(server_strategy_settings, seed)
        case "ECSM":
            # Instantiate the ECSM's client selector.
            client_selector = ECSM(server_strategy_settings, seed)
        case "MetaCS-FL":
            # Instantiate the MetaCS-FL's client selector.
            client_selector = MetaCSFL(server_strategy_settings, seed)
    return client_selector


def main(scalability_experiments_config_file: Path) -> None:
    scalability_experiments_settings = parse_config_file(scalability_experiments_config_file)
    approach_settings = scalability_experiments_settings["Approach Settings"]
    dataset_settings = scalability_experiments_settings["Dataset Settings"]
    experiments_combinations_settings = scalability_experiments_settings["Experiments Combinations Settings"]
    output_settings = scalability_experiments_settings["Output Settings"]
    # Set the output directory.
    output_directory = output_settings["output_directory"]
    # Set the strategy name (and substrategy if any).
    approach_name = approach_settings["approach_name"]
    strategy_name = ""
    substrategy_name = ""
    match approach_name:
        case "FedAvg":
            strategy_name = "SBAC-PAD_2024"
            substrategy_name = "Random"
        case "MEC":
            strategy_name = "SBAC-PAD_2024"
            substrategy_name = "MEC"
        case "ECMTC":
            strategy_name = "SBAC-PAD_2024"
            substrategy_name = "ECMTC"
        case "Oort":
            strategy_name = "Oort"
        case "DivFL":
            strategy_name = "DivFL"
        case "ECSM":
            strategy_name = "ECSM"
        case "MetaCS-FL":
            strategy_name = "MetaCS-FL"
    # Set the dataset settings.
    train_dataset_size = dataset_settings["train_dataset_size"]
    test_dataset_size = dataset_settings["test_dataset_size"]
    num_classes = dataset_settings["num_classes"]
    # Set the experiments combinations (Candidate client sizes, number of tasks, and rounds).
    num_clients_list = experiments_combinations_settings["num_clients_list"]
    num_tasks_list = experiments_combinations_settings["num_tasks_list"]
    num_rounds_list = experiments_combinations_settings["num_rounds_list"]
    # Generate all valid (clients, tasks, rounds) tuples.
    valid_tuples = [(num_clients, num_tasks, num_rounds)
                    for num_clients in num_clients_list
                    for num_tasks in num_tasks_list
                    for num_rounds in num_rounds_list]
    # Define already completed experiments (clients, tasks, rounds).
    already_done = {}
    # Filter out already-done experiments.
    remaining_tuples = [(c, t, r)
                        for (c, t, r) in valid_tuples
                        if (c, t, r) not in already_done]
    # Inspect generated experiments.
    print("Total valid experiments: {0}".format(len(valid_tuples)))
    print("Already done experiments: {0}".format(len(already_done)))
    print("Remaining experiments: {0}".format(len(remaining_tuples)))
    print("\nRemaining experiment tuples:")
    for tup in remaining_tuples:
        num_clients = tup[0]
        num_tasks = tup[1]
        num_rounds = tup[2]
        print("num_clients = {0} | num_tasks = {1} | num_rounds = {2}".format(num_clients, num_tasks, num_rounds))
    # Set the base logging settings.
    logging_settings = {'enable_logging': True,
                        'log_to_file': True,
                        'log_to_console': True,
                        'file_name': 'scalability_experiment/logging/',
                        'file_mode': 'w',
                        'encoding': 'utf-8',
                        'level': 'INFO',
                        'format_str': '%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
                        'date_format': '%Y/%m/%d %H:%M:%S'}
    # Get the strategy settings.
    server_strategy_settings = get_strategy_settings(strategy_name, substrategy_name)
    # Get the seed, if any.
    seed = None
    if "seed" in server_strategy_settings:
        seed = server_strategy_settings["seed"]
    # Initializations.
    selected_clients_history = {}
    selected_clients_metrics_history = {}
    results = []
    # Run the scalability experiments.
    for tup in remaining_tuples:
        # Initialize the client selector.
        client_selector = load_client_selector(server_strategy_settings, seed)
        num_clients = tup[0]
        num_tasks = tup[1]
        num_rounds = tup[2]
        print("{0}: num_clients = {1} | num_tasks = {2} | num_rounds = {3}".format(approach_name, num_clients, num_tasks, num_rounds))
        # Load the logger.
        logging_settings["file_name"] = "{0}/{1}_{2}_clients_{3}_tasks.log" \
                                        .format(output_directory, approach_name, num_clients, num_tasks)
        logger_name = "{0}_Scalability_Logger".format(approach_name)
        logger = load_logger(logging_settings, logger_name)
        # Generate the candidate clients.
        max_train_samples_per_class = int((train_dataset_size / num_clients) / num_classes)
        max_test_samples_per_class = int((test_dataset_size / num_clients) / num_classes)
        samples_per_task = server_strategy_settings["samples_per_task"]
        data_privacy_approach = server_strategy_settings["data_privacy_approach"]["name"]
        candidate_clients = generate_candidate_clients(num_clients,
                                                       num_classes,
                                                       max_train_samples_per_class,
                                                       max_test_samples_per_class,
                                                       samples_per_task,
                                                       data_privacy_approach)
        # Generate the clients profiles.
        samples_per_task = server_strategy_settings["samples_per_task"]
        num_tasks_profile_training = server_strategy_settings["num_tasks_profile_training"]
        num_tasks_profile_testing = server_strategy_settings["num_tasks_profile_testing"]
        num_samples_profile_training_list = [x * samples_per_task for x in num_tasks_profile_training]
        num_samples_profile_testing_list = [x * samples_per_task for x in num_tasks_profile_testing]
        clients_profiles = simulate_profiling_metrics(candidate_clients,
                                                      num_samples_profile_training_list,
                                                      num_samples_profile_testing_list,
                                                      seed)
        # Set the client selection procedure kwargs.
        server_round = 1
        base_learning_rate = 0.001
        base_batch_size = 32
        base_num_epochs = 5
        round_timeout_in_seconds = inf

        #
        root_output_folder = Path(output_directory) / "{0}_{1}_clients_{2}_tasks" \
                            .format(approach_name, num_clients, num_tasks)

        kwargs = {"current_round": server_round,
                  "current_phase": "train",
                  "candidate_clients": candidate_clients,
                  "num_rounds": num_rounds,
                  "base_learning_rate": base_learning_rate,
                  "base_batch_size": base_batch_size,
                  "base_num_epochs": base_num_epochs,
                  "selected_clients_history": selected_clients_history,
                  "selected_clients_metrics_history": selected_clients_metrics_history,
                  "clients_profiles": clients_profiles,
                  "time_limit": round_timeout_in_seconds,
                  "root_output_folder": root_output_folder,
                  "logger": logger}
        if "num_tasks_training" in server_strategy_settings:
            # Set the number of tasks to be scheduled to the selected clients.
            kwargs.update({"num_tasks": num_tasks})
        if "samples_per_task" in server_strategy_settings:
            # Get the number of samples per task.
            samples_per_task = server_strategy_settings["samples_per_task"]
            kwargs.update({"samples_per_task": samples_per_task})
        if "data_privacy_approach" in server_strategy_settings:
            # Get the privacy approach.
            data_privacy_approach = server_strategy_settings["data_privacy_approach"]["name"]
            kwargs.update({"data_privacy_approach": data_privacy_approach})
        if "use_async" in server_strategy_settings:
            # Whether asynchronous selection is enabled or not.
            use_async = server_strategy_settings["use_async"]
            kwargs.update({"use_async": use_async})
        # Initializations.
        total_selection_time = 0.0
        total_selection_cpu_time = 0.0
        total_selection_rss_peak_mb = 0.0
        total_selection_rss_avg_mb = 0.0
        total_selection_rss_delta_mb = 0.0
        max_selection_rss_peak_mb = 0.0
        max_selection_rss_delta_mb = 0.0
        total_selected_clients = 0
        # Iterate through FL rounds.
        for server_round in range(1, num_rounds + 1):
            try:
                print("Current FL round: {0} (num_clients = {1} | num_tasks = {2} | num_rounds = {3})"                       .format(server_round, num_clients, num_tasks, num_rounds))
                # Update kwargs.
                kwargs.update({"current_round": server_round,
                               "selected_clients_history": selected_clients_history,
                               "selected_clients_metrics_history": selected_clients_metrics_history})
                # Select clients using the defined approach (training phase only).
                selected_clients, selection_resource_metrics = _measure_selection_resources(client_selector, kwargs)
                selected_clients = normalize_selected_clients(selected_clients)
                # Update the total selection time.
                total_selection_time += selection_resource_metrics["selection_wall_time_seconds"]
                total_selection_cpu_time += selection_resource_metrics["selection_cpu_time_seconds"]
                total_selection_rss_peak_mb += selection_resource_metrics["selection_rss_peak_mb"]
                total_selection_rss_avg_mb += selection_resource_metrics["selection_rss_avg_mb"]
                total_selection_rss_delta_mb += selection_resource_metrics["selection_rss_delta_mb"]
                max_selection_rss_peak_mb = max(max_selection_rss_peak_mb, selection_resource_metrics["selection_rss_peak_mb"])
                max_selection_rss_delta_mb = max(max_selection_rss_delta_mb, selection_resource_metrics["selection_rss_delta_mb"])
                # Update the total selected clients.
                selected_client_ids = extract_selected_client_ids(selected_clients)
                total_selected_clients += len(selected_client_ids)
                # With a given probability, each round metrics cross MetaCS-FL thresholds (MetaCS-FL only).
                new_client_selection_trigger_probability = 0
                if strategy_name == "MetaCS-FL":
                    new_client_selection_trigger_probability = approach_settings["new_client_selection_trigger_probability"]
                # Simulate the training metrics for this round.
                train_metrics = simulate_clients_metrics_for_round(selected_clients,
                                                                   clients_profiles,
                                                                   server_round,
                                                                   phase="train",
                                                                   new_client_selection_trigger_probability=new_client_selection_trigger_probability)
                # Simulate the test metrics for this round.
                test_metrics = simulate_clients_metrics_for_round(selected_clients,
                                                                  clients_profiles,
                                                                  server_round,
                                                                  phase="test",
                                                                  new_client_selection_trigger_probability=new_client_selection_trigger_probability)
                # Update the selected clients' history (training phase only).
                selected_clients_history[server_round] = {"train": selected_clients}
                # Update the selected clients metrics' history (both phases).
                if server_round not in selected_clients_metrics_history:
                    selected_clients_metrics_history[server_round] = {}
                for phase in ["train", "test"]:
                    if phase not in selected_clients_metrics_history[server_round]:
                        selected_clients_metrics_history[server_round][phase] = {}
                selected_clients_metrics_history[server_round]["train"]["clients_metrics_dicts"] = train_metrics
                selected_clients_metrics_history[server_round]["test"]["clients_metrics_dicts"] = test_metrics
            except Exception as e:
                print("[SCALABILITY EXPERIMENT ERROR]")
                print("approach_name: {0}".format(approach_name))
                print("strategy_name: {0}".format(strategy_name))
                print("num_clients: {0}".format(num_clients))
                print("num_tasks: {0}".format(num_tasks))
                print("num_rounds: {0}".format(num_rounds))
                print("server_round: {0}".format(server_round))
                print("exception: {0}".format(repr(e)))
                print(format_exc())
                raise
        # Append the tuple outcome to the list of results.
        results.append({"approach_name": approach_name,
                        "num_clients": num_clients,
                        "num_tasks": num_tasks,
                        "num_rounds": num_rounds,
                        "total_client_selection_overhead": total_selection_time,
                        "avg_overhead_per_round": total_selection_time / num_rounds,
                        "total_selection_cpu_time_seconds": total_selection_cpu_time,
                        "avg_selection_cpu_time_seconds": total_selection_cpu_time / num_rounds,
                        "avg_selection_rss_peak_mb": total_selection_rss_peak_mb / num_rounds,
                        "avg_selection_rss_avg_mb": total_selection_rss_avg_mb / num_rounds,
                        "avg_selection_rss_delta_mb": total_selection_rss_delta_mb / num_rounds,
                        "max_selection_rss_peak_mb": max_selection_rss_peak_mb,
                        "max_selection_rss_delta_mb": max_selection_rss_delta_mb,
                        "avg_selected_clients_per_round": total_selected_clients / num_rounds})
        # Write the tuple outcome to CSV file.
        results_csv_path = Path(output_directory) / "{0}_{1}_clients_{2}_tasks_scalability_results.csv" \
                            .format(approach_name, num_clients, num_tasks)
        df_row = DataFrame([results[-1]])
        write_header = not results_csv_path.exists()
        df_row.to_csv(results_csv_path, mode="a", header=write_header, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-file",
                        type=Path,
                        required=True,
                        help="Relative path to the configuration file")
    args = parser.parse_args()
    config_file = args.config_file
    main(config_file)
