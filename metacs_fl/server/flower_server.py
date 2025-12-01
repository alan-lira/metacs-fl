from collections import defaultdict
from concurrent.futures import as_completed
from copy import deepcopy
from logging import Logger
from numpy import array, inf, sum as numpy_sum, mean, clip, float32
from numpy.random import default_rng
from pathlib import Path
from threading import Thread
from time import process_time, sleep
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, Metrics, NDArrays, Parameters, \
    parameters_to_ndarrays, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from metacs_fl.client_selector.metacsfl import MetaCSFL
from metacs_fl.client_selector.random import Random
from metacs_fl.client_selector.sbacpad_2024 import SBACPAD2024
from metacs_fl.metrics_aggregator.flower_weighted_average import aggregate_loss_by_weighted_average, \
    aggregate_metrics_by_weighted_average
from metacs_fl.model_aggregator.flower_fed_avg import aggregate_parameters_with_fed_avg
from metacs_fl.model_aggregator.flower_fed_avg_e import aggregate_parameters_with_fed_avg_e
from metacs_fl.model_aggregator.flower_fed_prox import aggregate_parameters_with_fed_prox
from metacs_fl.utils.logger_util import log_message


class FlowerServer(Strategy):

    def __init__(self,
                 *,
                 id_: int,
                 fl_settings: dict,
                 server_strategy_settings: dict,
                 fit_config: dict,
                 evaluate_config: dict,
                 output_settings: dict,
                 initial_parameters: Optional[NDArrays],
                 logger: Logger) -> None:
        # Initialize the attributes.
        super().__init__()
        self._server_id = id_
        self._fl_settings = fl_settings
        self._server_strategy_settings = server_strategy_settings
        self._fit_config = fit_config
        self._evaluate_config = evaluate_config
        self._output_settings = output_settings
        self._initial_parameters = initial_parameters
        self._global_parameters = initial_parameters
        self._logger = logger
        self._fit_pairs_history = {}
        self._evaluate_pairs_history = {}
        self._candidate_clients_history = {}
        self._selected_clients_history = {}
        self._client_selection_duration_history = {}
        self._selected_clients_metrics_history = {}
        self._clients_histograms = {}
        self._clients_reliability_score_history = {}
        # Initialize the random number generator with a fixed seed to allow replicable results.
        seed = None
        if "seed" in self._server_strategy_settings:
            seed = self._server_strategy_settings["seed"]
        self._rng = default_rng(seed=seed)
        # Initialize the list of profiling rounds.
        profiling_rounds = []
        # Initialize the client selector.
        client_selector = None
        strategy = server_strategy_settings["strategy"]
        match strategy:
            case "Random":
                # Instantiate the Random's client selector.
                client_selector = Random(server_strategy_settings, seed)
            case "SBAC-PAD_2024":
                # Instantiate the SBAC-PAD_2024's client selector.
                client_selector = SBACPAD2024(server_strategy_settings, seed)
            case "MetaCS-FL":
                # Instantiate the MetaCS-FL's client selector.
                client_selector = MetaCSFL(server_strategy_settings, seed)
        # Set the list of profiling rounds (starting at the first round).
        num_profiling_rounds = server_strategy_settings.get("num_profiling_rounds", 0)
        if num_profiling_rounds > 0:
            profiling_rounds = list(range(1, num_profiling_rounds + 1))
            self._profiling_rounds = profiling_rounds
        self._profiling_rounds = profiling_rounds
        self._client_selector = client_selector

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    def _get_metrics_of_past_x_rounds(selected_clients_metrics_history: dict,
                                      current_round: int,
                                      phase_of_interest: str,
                                      x: int) -> dict:
        # Sort rounds in history.
        history_rounds = sorted(selected_clients_metrics_history.keys())
        last_round_on_history = history_rounds[-1]
        # Clip current_round if it is beyond the last round in history.
        if current_round > last_round_on_history:
            current_round = last_round_on_history + 1
        # Search backward for up to x valid past rounds.
        valid_past_rounds = []
        r = current_round - 1
        while r > 0 and len(valid_past_rounds) < x:
            if r in selected_clients_metrics_history:
                round_data = selected_clients_metrics_history[r]
                # Check if phase is present AND has non-empty metrics.
                if (phase_of_interest in round_data and
                        "clients_metrics_dicts" in round_data[phase_of_interest] and
                        len(round_data[phase_of_interest]["clients_metrics_dicts"]) > 0):
                    valid_past_rounds.append(r)
            r -= 1
        # Sort chronologically.
        valid_past_rounds = sorted(valid_past_rounds)
        # Filter history to only valid rounds.
        selected_clients_metrics_history_filtered = {r: selected_clients_metrics_history[r]
                                                     for r in valid_past_rounds}
        # Initialize metric lists.
        makespans = []
        energy_consumptions = []
        weighted_mean_accuracies = []
        # Compute metrics for each valid round.
        for past_round in valid_past_rounds:
            # Extract client metric dicts.
            clients_metrics_dicts = selected_clients_metrics_history_filtered[past_round][phase_of_interest]["clients_metrics_dicts"]
            # Initialize per-round accumulators.
            makespan = 0
            energy_consumption = 0
            sum_accuracy_product = 0
            sum_num_examples_used = 0
            # Loop through each client's metrics.
            for client_metrics_dict in clients_metrics_dicts:
                client_id = next(iter(client_metrics_dict))
                client_metrics = client_metrics_dict[client_id]
                # Get the number of tasks executed by the client i on round r.
                x_i = client_metrics.get("num_examples", 0)
                # Get the time cost of the client i on round r (if available).
                time_key = "{0}ing_time_in_seconds".format(phase_of_interest)
                time_i = client_metrics.get(time_key, 0)
                # Update the makespan of round r.
                makespan = max(makespan, time_i)
                # Get the energy cost of the client i on round r (if available).
                energy_key = "{0}ing_energy_in_joules".format(phase_of_interest)
                energy_i = client_metrics.get(energy_key, 0)
                energy_consumption += energy_i
                # Get the accuracy "cost" of the client i on round r (if available).
                accuracy_i = 0
                for metric_key, val in client_metrics.items():
                    if "accuracy" in metric_key:
                        accuracy_i = val
                # Weighted accuracy accumulators.
                sum_accuracy_product += x_i * accuracy_i
                sum_num_examples_used += x_i
            # Update the weighted mean accuracy of round r.
            weighted_mean_accuracy = (sum_accuracy_product / sum_num_examples_used
                                      if sum_num_examples_used > 0 else 0)
            # Store metrics.
            makespans.append(makespan)
            energy_consumptions.append(energy_consumption)
            weighted_mean_accuracies.append(weighted_mean_accuracy)
        # Set the dictionary of the metrics lists.
        metrics_past_rounds = {"makespans": makespans,
                               "energy_consumptions": energy_consumptions,
                               "weighted_mean_accuracies": weighted_mean_accuracies}
        # Return the dictionary of the metrics lists.
        return metrics_past_rounds

    def _fl_execution_should_stop(self,
                                  current_round: int) -> bool:
        # Get the necessary attributes.
        fl_settings = self.get_attribute("_fl_settings")
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        stopping_crit = fl_settings["stopping_crit"]
        profiling_rounds = self.get_attribute("_profiling_rounds")
        selected_clients_metrics_history = self.get_attribute("_selected_clients_metrics_history")
        # Initialize the list of FL stopping criteria results.
        fl_stopping_criteria_results = []
        # Iterate through the FL stopping criteria.
        for k, v in stopping_crit.items():
            if "criterion_" in k:
                crit_name = v["name"]
                match crit_name:
                    case "Max_Rounds":
                        max_rounds = v["max_rounds"]
                        all_rounds_executed = (current_round > max_rounds)
                        restore_initial_parameters_after_profiling = False
                        if "restore_initial_parameters_after_profiling" in server_strategy_settings:
                            restore_initial_parameters_after_profiling \
                                = server_strategy_settings["restore_initial_parameters_after_profiling"]
                        if restore_initial_parameters_after_profiling:
                            all_rounds_executed = (current_round > max_rounds + len(profiling_rounds))
                        fl_stopping_criteria_results.append({crit_name: all_rounds_executed})
                    case "Min_Test_Accuracy":
                        min_test_accuracy = v["min_test_accuracy"]
                        metrics_past_rounds = self._get_metrics_of_past_x_rounds(selected_clients_metrics_history,
                                                                                 current_round,
                                                                                 "test",
                                                                                 1)
                        weighted_mean_accuracy = metrics_past_rounds["weighted_mean_accuracies"][0]
                        min_test_accuracy_achieved = (weighted_mean_accuracy >= min_test_accuracy)
                        fl_stopping_criteria_results.append({crit_name: min_test_accuracy_achieved})
        # Verify if the FL execution should stop.
        fl_execution_should_stop = False
        logical_operator = stopping_crit["logical_operator"]
        match logical_operator:
            case "AND":
                # Check if all the FL stopping criteria were met.
                fl_execution_should_stop = all(criterion_result[next(iter(criterion_result))]
                                           for criterion_result in fl_stopping_criteria_results)
            case "OR":
                # Check if at least one of the FL stopping criteria was met.
                fl_execution_should_stop = any(criterion_result[next(iter(criterion_result))]
                                           for criterion_result in fl_stopping_criteria_results)
        return fl_execution_should_stop

    @staticmethod
    def _estimate_global_class_presence(bits: list,
                                        true_presence_probability: float) -> float:
        # Unbiased estimate for true presence probability from randomized response bits.
        if not bits:
            return 0.0
        mean_bits = float(mean(bits))
        estimate = (mean_bits - 0.5 * (1 - true_presence_probability)) / (true_presence_probability - 0.5)
        estimate = float(clip(estimate, 0.0, 1.0))
        return estimate

    def _map_available_clients(self,
                               current_round: int,
                               client_manager: ClientManager | None) -> dict:
        # Get the necessary attributes.
        candidate_clients_history = self.get_attribute("_candidate_clients_history")
        selected_clients_history = self.get_attribute("_selected_clients_history")
        client_selection_duration_history = self.get_attribute("_client_selection_duration_history")
        selected_clients_metrics_history = self.get_attribute("_selected_clients_metrics_history")
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        profiling_rounds = self.get_attribute("_profiling_rounds")
        # Get the list of past rounds.
        past_rounds = list(range(1, current_round + 1))
        # Initialize the summary of past rounds' idle events.
        past_rounds_idle_events_summary = {}
        for r in past_rounds:
            if all(r in hist for hist in (candidate_clients_history, selected_clients_history, selected_clients_metrics_history)):
                candidate_clients_train_r = []
                selected_clients_train_r = []
                client_selection_time_train_r = 0
                training_time_key = "training_time_in_seconds"
                makespan_train_r = 0
                candidate_clients_test_r = []
                selected_clients_test_r = []
                client_selection_time_test_r = 0
                testing_time_key = "testing_time_in_seconds"
                makespan_test_r = 0
                candidate_clients_r = candidate_clients_history[r]
                selected_clients_r = selected_clients_history[r]
                client_selection_duration_r = client_selection_duration_history[r]
                selected_clients_metrics_r = selected_clients_metrics_history[r]
                if "train" in candidate_clients_r:
                    candidate_clients_train_r = list(candidate_clients_r["train"].keys())
                if "train" in selected_clients_r:
                    selected_clients_train_r = [client_id for client_id, _ in selected_clients_r["train"].items()]
                if "train" in client_selection_duration_r:
                    client_selection_time_train_r = client_selection_duration_r["train"]
                if "train" in selected_clients_metrics_r:
                    makespan_train_r = max(client_metrics[list(client_metrics.keys())[0]][training_time_key]
                                           for client_metrics in selected_clients_metrics_r["train"]["clients_metrics_dicts"])
                if "test" in candidate_clients_r:
                    candidate_clients_test_r = list(candidate_clients_r["test"].keys())
                if "test" in selected_clients_r:
                    selected_clients_test_r = [client_id for client_id, _ in selected_clients_r["test"].items()]
                if "test" in client_selection_duration_r:
                    client_selection_time_test_r = client_selection_duration_r["test"]
                if "test" in selected_clients_metrics_r:
                    makespan_test_r = max(client_metrics[list(client_metrics.keys())[0]][testing_time_key]
                                          for client_metrics in selected_clients_metrics_r["test"]["clients_metrics_dicts"])
                # Get the list of idle clients of the round r.
                idle_clients_train_r = [i for i in candidate_clients_train_r if i not in selected_clients_train_r]
                idle_clients_test_r = [i for i in candidate_clients_test_r if i not in selected_clients_test_r]
                # Update the summary of past rounds' idle events.
                r_dict = {}
                if client_selection_time_train_r != 0:
                    r_dict.update({"client_selection_time_train": client_selection_time_train_r})
                if len(idle_clients_train_r) > 0:
                    r_dict.update({"idle_clients_train": idle_clients_train_r})
                if makespan_train_r > 0:
                    r_dict.update({"makespan_train": makespan_train_r})
                if client_selection_time_test_r != 0:
                    r_dict.update({"client_selection_time_test": client_selection_time_test_r})
                if len(idle_clients_test_r) > 0:
                    r_dict.update({"idle_clients_test": idle_clients_test_r})
                if makespan_test_r > 0:
                    r_dict.update({"makespan_test": makespan_test_r})
                past_rounds_idle_events_summary.update({r: r_dict})
        idle_events_data_dict = {}
        for r, metrics in past_rounds_idle_events_summary.items():
            for prefix in ["train", "test"]:
                # Handle client selection duration (always include).
                duration_key = "client_selection_time_{0}".format(prefix)
                if duration_key in metrics:
                    var_name = "{0}_comm_round_{1}".format(duration_key, r)
                    idle_events_data_dict[var_name] = metrics[duration_key]
                # Handle idle clients and makespan (conditionally).
                idle_clients_key = "idle_clients_{0}".format(prefix)
                makespan_key = "makespan_{0}".format(prefix)
                if idle_clients_key in metrics and metrics[idle_clients_key]:
                    var_name = "{0}_comm_round_{1}".format(idle_clients_key, r)
                    idle_events_data_dict[var_name] = "|".join(metrics[idle_clients_key])
                    if makespan_key in metrics:
                        var_name = "{0}_comm_round_{1}".format(makespan_key, r)
                        idle_events_data_dict[var_name] = metrics[makespan_key]
        available_clients_map = {}
        if client_manager is not None:
            # Get the available clients.
            available_clients = client_manager.all()
            # Get the available clients data distribution, if allowed.
            data_privacy_approach_name = "Non_Private"
            data_privacy_approach_settings = {}
            query_clients_data_distribution = server_strategy_settings.get("query_clients_data_distribution", False)
            if query_clients_data_distribution:
                if "data_privacy_approach" in server_strategy_settings:
                    data_privacy_approach_settings = server_strategy_settings["data_privacy_approach"]
                    data_privacy_approach_name = data_privacy_approach_settings["name"]
                # Keep only active clients in the clients histograms cache.
                active_client_proxies = set(available_clients.values())
                self._clients_histograms = {client_proxy: client_info_cached
                                            for client_proxy, client_info_cached in self._clients_histograms.items()
                                            if client_proxy in active_client_proxies}
                # Query client info, if not cached.
                for _, client_proxy in available_clients.items():
                    client_info_cached = self._clients_histograms.get(client_proxy, {})
                    match data_privacy_approach_name:
                        case "Non_Private":
                            # Non-private: actual query counts per class.
                            if "client_tasks_per_class_train" not in client_info_cached:
                                client_id_property = "client_id"
                                client_tasks_per_class_train_property = "client_tasks_per_class_train"
                                client_tasks_per_class_test_property = "client_tasks_per_class_test"
                                gpi_dict = {client_id_property: "?",
                                            client_tasks_per_class_train_property: "?",
                                            client_tasks_per_class_test_property: "?",
                                            "samples_per_task": server_strategy_settings.get("samples_per_task", 1)}
                                gpi = GetPropertiesIns(gpi_dict)
                                client_prompted = client_proxy.get_properties(gpi, timeout=None, group_id=None)
                                client_id = client_prompted.properties[client_id_property]
                                client_tasks_per_class_train_str = client_prompted.properties[client_tasks_per_class_train_property]
                                client_tasks_per_class_train = {s.split("=")[0]: int(s.split("=")[1])
                                                                for s in client_tasks_per_class_train_str.split("|") if s}
                                client_tasks_per_class_test_str = client_prompted.properties[client_tasks_per_class_test_property]
                                client_tasks_per_class_test = {s.split("=")[0]: int(s.split("=")[1])
                                                               for s in client_tasks_per_class_test_str.split("|") if s}
                                self._clients_histograms[client_proxy] = {"client_id": client_id,
                                                                          "client_tasks_per_class_train": client_tasks_per_class_train,
                                                                          "client_tasks_per_class_test": client_tasks_per_class_test}
                        case "Differentially_Private":
                            # Query presence and local classes claims, if not cached.
                            true_presence_probability = data_privacy_approach_settings["true_presence_probability"]
                            if "client_dp_presence" not in client_info_cached:
                                client_id_property = "client_id"
                                client_dp_presence_property = "client_dp_presence"
                                client_local_classes_claimed_property = "client_local_classes_claimed"
                                gpi_dict = {client_id_property: "?",
                                            client_dp_presence_property: "?",
                                            client_local_classes_claimed_property: "?",
                                            "true_presence_probability": true_presence_probability}
                                gpi = GetPropertiesIns(gpi_dict)
                                client_prompted = client_proxy.get_properties(gpi, timeout=None, group_id=None)
                                client_id = client_prompted.properties[client_id_property]
                                client_dp_presence_str = client_prompted.properties[client_dp_presence_property]
                                client_dp_presence = list(map(int, client_dp_presence_str.split("|")))
                                client_local_classes_claimed_str = client_prompted.properties[client_local_classes_claimed_property]
                                client_local_classes_claimed = client_local_classes_claimed_str.split("|")
                                self._clients_histograms[client_proxy] = {"client_id": client_id,
                                                                          "client_dp_presence": client_dp_presence,
                                                                          "client_local_classes_claimed": client_local_classes_claimed}
                if data_privacy_approach_name == "Differentially_Private":
                    # Estimate the global classes and build the class-index mapping.
                    true_presence_probability = data_privacy_approach_settings["true_presence_probability"]
                    presence_cutoff = data_privacy_approach_settings["presence_cutoff"]
                    epsilon = data_privacy_approach_settings["epsilon"]
                    global_presence = defaultdict(list)
                    for client_proxy, _ in self._clients_histograms.items():
                        client_dp_presence = self._clients_histograms[client_proxy]["client_dp_presence"]
                        client_local_classes_claimed = self._clients_histograms[client_proxy]["client_local_classes_claimed"]
                        for cls, bit in zip(client_local_classes_claimed, client_dp_presence):
                            global_presence[cls].append(bit)
                    estimates = {cls: self._estimate_global_class_presence(bits, true_presence_probability)
                                 for cls, bits in global_presence.items()}
                    global_classes = [cls for cls, est in estimates.items() if est > presence_cutoff]
                    num_global_classes = len(global_classes)
                    class_index_map = {lbl: idx for idx, lbl in enumerate(global_classes)}
                    # Set the class-index mapping.
                    self._clients_histograms["class_index_map"] = class_index_map
                    # Query DP histograms, if not cached.
                    class_index_map_str = "|".join("{0}={1}".format(k, v) for k, v in class_index_map.items())
                    for _, client_proxy in available_clients.items():
                        client_info_cached = self._clients_histograms[client_proxy]
                        if "client_dp_histogram" not in client_info_cached:
                            dp_histogram_property = "client_dp_histogram"
                            gpi_dict = {dp_histogram_property: "?",
                                        "num_global_classes": num_global_classes,
                                        "class_index_map": class_index_map_str,
                                        "epsilon": epsilon}
                            gpi = GetPropertiesIns(gpi_dict)
                            client_prompted = client_proxy.get_properties(gpi, timeout=None, group_id=None)
                            client_dp_histogram_str = client_prompted.properties[dp_histogram_property]
                            client_dp_histogram = array(list(map(float, client_dp_histogram_str.split("|"))), dtype=float32)
                            self._clients_histograms[client_proxy].update({"client_dp_histogram": client_dp_histogram})
            # Generate the available clients map.
            for _, client_proxy in available_clients.items():
                client_id_property = "client_id"
                client_hostname_property = "client_hostname"
                client_num_cpus_property = "client_num_cpus"
                client_cpu_cores_list_property = "client_cpu_cores_list"
                client_num_training_examples_available_property = "client_num_training_examples_available"
                client_num_testing_examples_available_property = "client_num_testing_examples_available"
                client_task_assignment_capacities_train_property = "client_task_assignment_capacities_train"
                client_task_assignment_capacities_test_property = "client_task_assignment_capacities_test"
                client_remaining_battery_energy_property = "client_remaining_battery_energy"
                client_mean_power_consumption_idle_mode_property = "client_mean_power_consumption_idle_mode"
                client_current_download_bandwidth_in_bytes_per_second_property = "client_current_download_bandwidth_in_bytes_per_second"
                client_current_upload_bandwidth_in_bytes_per_second_property = "client_current_upload_bandwidth_in_bytes_per_second"
                client_current_latency_in_milliseconds_property = "client_current_latency_in_milliseconds"
                gpi_dict = {client_id_property: "?",
                            client_hostname_property: "?",
                            client_num_cpus_property: "?",
                            client_cpu_cores_list_property: "?",
                            client_num_training_examples_available_property: "?",
                            client_num_testing_examples_available_property: "?",
                            client_task_assignment_capacities_train_property: "?",
                            client_task_assignment_capacities_test_property: "?",
                            client_remaining_battery_energy_property: "?",
                            client_mean_power_consumption_idle_mode_property: "?",
                            client_current_download_bandwidth_in_bytes_per_second_property: "?",
                            client_current_upload_bandwidth_in_bytes_per_second_property: "?",
                            client_current_latency_in_milliseconds_property: "?",
                            "samples_per_task": server_strategy_settings.get("samples_per_task", 1),
                            "comm_round": current_round,
                            "is_profiling_round": current_round in profiling_rounds}
                gpi_dict.update(idle_events_data_dict)
                gpi = GetPropertiesIns(gpi_dict)
                client_prompted = client_proxy.get_properties(gpi, timeout=None, group_id=None)
                client_id = client_prompted.properties[client_id_property]
                client_hostname = client_prompted.properties[client_hostname_property]
                client_num_cpus = client_prompted.properties[client_num_cpus_property]
                client_cpu_cores_list = client_prompted.properties[client_cpu_cores_list_property]
                client_num_training_examples_available = \
                    client_prompted.properties[client_num_training_examples_available_property]
                client_num_testing_examples_available = \
                    client_prompted.properties[client_num_testing_examples_available_property]
                client_task_assignment_capacities_train = \
                    client_prompted.properties[client_task_assignment_capacities_train_property]
                client_task_assignment_capacities_train = client_task_assignment_capacities_train.split("|")
                client_task_assignment_capacities_train = [int(i) for i in client_task_assignment_capacities_train]
                client_task_assignment_capacities_test = \
                    client_prompted.properties[client_task_assignment_capacities_test_property]
                client_task_assignment_capacities_test = client_task_assignment_capacities_test.split("|")
                client_task_assignment_capacities_test = [int(i) for i in client_task_assignment_capacities_test]
                client_remaining_battery_energy = client_prompted.properties[client_remaining_battery_energy_property]
                client_mean_power_consumption_idle_mode = client_prompted.properties[client_mean_power_consumption_idle_mode_property]
                client_current_download_bandwidth_in_bytes_per_second = client_prompted.properties[client_current_download_bandwidth_in_bytes_per_second_property]
                client_current_upload_bandwidth_in_bytes_per_second = client_prompted.properties[client_current_upload_bandwidth_in_bytes_per_second_property]
                client_current_latency_in_milliseconds = client_prompted.properties[client_current_latency_in_milliseconds_property]
                client_available = client_prompted.properties["client_available"]
                if client_available:
                    client_id_str = "client_{0}".format(client_id)
                    client_map = {"client_proxy": client_proxy,
                                  "client_hostname": client_hostname,
                                  "client_num_cpus": client_num_cpus,
                                  "client_cpu_cores_list": client_cpu_cores_list,
                                  "client_num_training_examples_available": client_num_training_examples_available,
                                  "client_num_testing_examples_available": client_num_testing_examples_available,
                                  "client_task_assignment_capacities_train": client_task_assignment_capacities_train,
                                  "client_task_assignment_capacities_test": client_task_assignment_capacities_test,
                                  "client_remaining_battery_energy": client_remaining_battery_energy,
                                  "client_mean_power_consumption_idle_mode": client_mean_power_consumption_idle_mode,
                                  "client_current_download_bandwidth_in_bytes_per_second": client_current_download_bandwidth_in_bytes_per_second,
                                  "client_current_upload_bandwidth_in_bytes_per_second": client_current_upload_bandwidth_in_bytes_per_second,
                                  "client_current_latency_in_milliseconds": client_current_latency_in_milliseconds}
                    # Attach the data distribution info (histogram), if allowed.
                    if query_clients_data_distribution:
                        if client_proxy in self._clients_histograms:
                            client_info = self._clients_histograms[client_proxy]
                            match data_privacy_approach_name:
                                case "Non_Private":
                                    if "client_tasks_per_class_train" in client_info:
                                        client_map.update({"client_tasks_per_class_train": client_info["client_tasks_per_class_train"]})
                                    if "client_tasks_per_class_test" in client_info:
                                        client_map.update({"client_tasks_per_class_test": client_info["client_tasks_per_class_test"]})
                                case "Differentially_Private":
                                    if "client_dp_histogram" in client_info:
                                        client_map.update({"client_dp_histogram": client_info["client_dp_histogram"]})
                                    if "class_index_map" in self._clients_histograms:
                                        client_map.update({"class_index_map": self._clients_histograms["class_index_map"]})
                    available_clients_map.update({client_id_str: client_map})
            sorted_keys = sorted(list(available_clients_map.keys()), key=lambda x: (len(x), x))
            available_clients_map = {k: available_clients_map[k] for k in sorted_keys}
        else:
            available_clients_map = self.get_attribute("_available_clients_map_replay")
        return available_clients_map

    def _run_post_client_selection_tasks(self,
                                         current_round: int,
                                         current_phase: str,
                                         candidate_clients: dict,
                                         selected_clients: dict,
                                         selection_duration_in_seconds: float,
                                         parameters: Parameters) -> None:
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        candidate_clients_history = self.get_attribute("_candidate_clients_history")
        selected_clients_history = self.get_attribute("_selected_clients_history")
        client_selection_duration_history = self.get_attribute("_client_selection_duration_history")
        # Get the list of profiling rounds, if any.
        profiling_rounds = self.get_attribute("_profiling_rounds")
        # Set the base configuration.
        phase_config = self._update_config(current_round, current_phase)
        # Update the candidate clients' history.
        if current_round not in candidate_clients_history:
            candidate_clients_history.update({current_round: {current_phase: candidate_clients}})
        else:
            candidate_clients_history[current_round].update({current_phase: candidate_clients})
        self._set_attribute("_candidate_clients_history", candidate_clients_history)
        # Update the selected clients' history.
        if current_round not in selected_clients_history:
            selected_clients_history.update({current_round: {current_phase: selected_clients}})
            client_selection_duration_history.update({current_round: {current_phase: selection_duration_in_seconds}})
        else:
            selected_clients_history[current_round].update({current_phase: selected_clients})
            client_selection_duration_history[current_round].update({current_phase: selection_duration_in_seconds})
        self._set_attribute("_selected_clients_history", selected_clients_history)
        self._set_attribute("_client_selection_duration_history", client_selection_duration_history)
        # Set the list of (client_proxy, client_instructions) phase pairs.
        phase_pairs = []
        for _, client_info in selected_clients.items():
            selected_client_proxy = client_info["client_proxy"]
            selected_client_config = deepcopy(phase_config)
            selected_client_config.update({"client_selection_time_in_seconds": selection_duration_in_seconds})
            if current_round in profiling_rounds:
                selected_client_config.update({"is_profiling_round": True})
            if "client_num_samples_scheduled" in client_info:
                selected_client_config.update({"num_{0}ing_examples_to_use".format(current_phase):
                                                   client_info["client_num_samples_scheduled"]})
            if "client_num_samples_per_class_scheduled" in client_info:
                selected_client_config.update({"num_{0}ing_examples_per_class_to_use".format(current_phase):
                                                   client_info["client_num_samples_per_class_scheduled"]})
            if "client_learning_rate" in client_info:
                selected_client_config.update({"learning_rate": client_info["client_learning_rate"]})
            if "client_batch_size" in client_info:
                selected_client_config.update({"batch_size": client_info["client_batch_size"]})
            if "client_epochs" in client_info:
                selected_client_config.update({"epochs": client_info["client_epochs"]})
            selected_client_instructions = None
            match current_phase:
                case "train":
                    selected_client_instructions = FitIns(parameters, selected_client_config)
                case "test":
                    selected_client_instructions = EvaluateIns(parameters, selected_client_config)
            # Restore the initial parameters after the profiling rounds, if needed.
            restore_initial_parameters_after_profiling = False
            if "restore_initial_parameters_after_profiling" in server_strategy_settings:
                restore_initial_parameters_after_profiling = server_strategy_settings[
                    "restore_initial_parameters_after_profiling"]
            if profiling_rounds and current_round <= profiling_rounds[-1] + 1 and restore_initial_parameters_after_profiling:
                initial_parameters = self.get_attribute("_initial_parameters")
                match current_phase:
                    case "train":
                        selected_client_instructions = FitIns(initial_parameters, selected_client_config)
                    case "test":
                        selected_client_instructions = EvaluateIns(initial_parameters, selected_client_config)
            phase_pairs.append((selected_client_proxy, selected_client_instructions))
        # Update the repository of phase pairs.
        self._update_phase_pairs_history(current_round, current_phase, phase_pairs)
        # Log a 'number of clients selected' message.
        message = "[Server {0} | Round {1}] {2} {3}ing {4} (out of {5}) {6} selected for round {7}!" \
            .format(server_id,
                    current_round,
                    len(selected_clients),
                    current_phase,
                    "clients" if len(selected_clients) != 1 else "client",
                    len(candidate_clients),
                    "were" if len(selected_clients) != 1 else "was",
                    current_round)
        log_message(logger, message, "INFO")

    def _monitor_future_objects(self,
                                future_objects: list,
                                phase: str,
                                candidate_clients: dict,
                                parameters: Parameters) -> None:
        for future_object in as_completed(future_objects):
            future_object_result = future_object.result()
            round_key, round_values = next(iter(future_object_result.items()))
            selected_clients = round_values["selected_clients"]
            selection_duration_in_seconds = round_values["selection_duration_in_seconds"]
            # Run the post-client selection tasks.
            self._run_post_client_selection_tasks(round_key,
                                                  phase,
                                                  candidate_clients,
                                                  selected_clients,
                                                  selection_duration_in_seconds,
                                                  parameters)

    def _update_config(self,
                       current_round: int,
                       current_phase: str) -> Optional[dict]:
        """Updates the configuration that will be sent to clients."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Get the configuration.
        config_attribute = None
        if current_phase == "train":
            config_attribute = "_fit_config"
        elif current_phase == "test":
            config_attribute = "_evaluate_config"
        config = self.get_attribute(config_attribute)
        # Update its current communication round.
        config.update({"comm_round": current_round})
        self._set_attribute(config_attribute, config)
        # Replace None values to 'None' (necessary workaround on Flower).
        config = {k: ("None" if v is None else v) for k, v in config.items()}
        # Log the current configuration.
        message = "[Server {0} | Round {1}] Current {2}ing config: {3}" \
                  .format(server_id, current_round, current_phase, config)
        log_message(logger, message, "DEBUG")
        # Return the configuration.
        return config

    def _update_phase_pairs_history(self,
                                    current_round: int,
                                    current_phase: str,
                                    phase_pairs: list) -> None:
        phase_pairs_history_attribute = None
        if current_phase == "train":
            phase_pairs_history_attribute = "_fit_pairs_history"
        elif current_phase == "test":
            phase_pairs_history_attribute = "_evaluate_pairs_history"
        phase_pairs_history = self.get_attribute(phase_pairs_history_attribute)
        if current_round not in phase_pairs_history:
            current_round_phase_pairs = {current_round: phase_pairs}
            phase_pairs_history.update(current_round_phase_pairs)
            self._set_attribute(phase_pairs_history_attribute, phase_pairs_history)

    def _update_individual_metrics_history(self,
                                           current_round: int,
                                           current_phase: str,
                                           metrics: list[tuple[int, Metrics]]) -> None:
        candidate_clients_history = self.get_attribute("_candidate_clients_history")
        selected_clients_history = self.get_attribute("_selected_clients_history")
        selected_clients_metrics_history = self.get_attribute("_selected_clients_metrics_history")
        is_task_based_selection = all("client_num_tasks_scheduled" in client_info
                                      for client_info in selected_clients_history[current_round][current_phase].values())
        num_available_clients = len(candidate_clients_history[current_round][current_phase])
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        client_selector = server_strategy_settings["strategy"]
        clients_metrics_dicts = []
        for metric_tuple in metrics:
            client_metrics = dict(metric_tuple[1])
            client_id = client_metrics["client_id"]
            client_id_str = "client_{0}".format(client_id)
            client_metrics_copy = client_metrics.copy()
            client_metrics_copy.pop("client_id")
            clients_metrics_dicts.append({client_id_str: client_metrics_copy})
        current_round_values = {"client_selector": client_selector,
                                "num_available_clients": num_available_clients,
                                "clients_metrics_dicts": clients_metrics_dicts}
        if is_task_based_selection:
            # Conditionally insert "num_tasks".
            num_tasks = sum([client_info["client_num_tasks_scheduled"]
                             for _, client_info in selected_clients_history[current_round][current_phase].items()])
            current_round_values.update({"num_tasks": num_tasks})
        if current_round not in selected_clients_metrics_history:
            selected_clients_metrics_history.update({current_round: {current_phase: current_round_values}})
        else:
            selected_clients_metrics_history[current_round].update({current_phase: current_round_values})
        self._set_attribute("_selected_clients_metrics_history", selected_clients_metrics_history)

    @staticmethod
    def _keep_numeric_metrics_only(metrics_tuples: list[tuple[int, Metrics]]) -> list:
        filtered_metrics_tuples = [(metric_tuple[0],
                                    {k: v for k, v in metric_tuple[1].items() if isinstance(v, (int, float))})
             for metric_tuple in metrics_tuples]
        return filtered_metrics_tuples

    @staticmethod
    def _remove_undesired_metrics(metrics_tuples: list[tuple[int, Metrics]],
                                  undesired_metrics: list) -> list[tuple[int, Metrics]]:
        filtered_metrics_tuples = []
        for metric_tuple in metrics_tuples:
            client_metrics = metric_tuple[1]
            client_metrics = {k: v for k, v in client_metrics.items() if k not in undesired_metrics}
            metric_tuple = (metric_tuple[0], client_metrics)
            filtered_metrics_tuples.append(metric_tuple)
        return filtered_metrics_tuples

    def _aggregate_fit_metrics(self,
                               current_round: int,
                               fit_metrics: list[tuple[int, Metrics]]) -> Optional[Metrics]:
        # Get the necessary attributes.
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        metrics_aggregator = server_strategy_settings["metrics_aggregator"]
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Set the phase value.
        phase = "train"
        # Update the individual training metrics history.
        self._update_individual_metrics_history(current_round, phase, fit_metrics)
        # Keep only numeric metrics for the aggregation.
        fit_metrics = self._keep_numeric_metrics_only(fit_metrics)
        # Remove the undesired metrics, if any.
        if "undesired_metrics_for_fit_aggregation" in server_strategy_settings:
            undesired_metrics_for_fit_aggregation = server_strategy_settings["undesired_metrics_for_fit_aggregation"]
            if undesired_metrics_for_fit_aggregation:
                fit_metrics = self._remove_undesired_metrics(fit_metrics, undesired_metrics_for_fit_aggregation)
        # Initialize the aggregated training metrics dictionary (aggregated_fit_metrics).
        aggregated_fit_metrics = {}
        # Aggregate the training metrics according to the user-defined aggregator.
        if metrics_aggregator == "Weighted_Average":
            aggregated_fit_metrics = aggregate_metrics_by_weighted_average(fit_metrics)
        # Get the number of participating clients.
        num_participating_clients = len(fit_metrics)
        num_participating_clients_str = "".join([str(num_participating_clients),
                                                 " Clients" if num_participating_clients > 1 else " Client"])
        # Log the aggregated training metrics.
        message = "[Server {0} | Round {1}] Aggregated training metrics ({2} of {3}): {4}" \
                  .format(server_id, current_round, metrics_aggregator, num_participating_clients_str,
                          aggregated_fit_metrics)
        log_message(logger, message, "DEBUG")
        # Return the aggregated training metrics (aggregated_fit_metrics).
        return aggregated_fit_metrics

    def _initialize_history_output_files(self,
                                         phase: str) -> None:
        # Get the necessary attributes.
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        selected_clients_metrics_history = self.get_attribute("_selected_clients_metrics_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        phase_substrings = []
        is_task_based_selection = False
        if phase == "train":
            phase_substrings = ["fit", "train"]
            is_task_based_selection = "num_tasks_training" in server_strategy_settings
        elif phase == "test":
            phase_substrings = ["evaluate", "test"]
            is_task_based_selection = "num_tasks_testing" in server_strategy_settings
        output_files_phase = [{k: v} for k, v in output_settings.items()
                              if any(substring in k for substring in phase_substrings)]
        # Remove the history output files, if removing is enabled.
        if remove_output_files:
            for output_file_dict in output_files_phase:
                history_output_file_key = next(iter(output_file_dict))
                history_output_file = Path(output_file_dict[history_output_file_key]).absolute()
                history_output_file.unlink(missing_ok=True)
        # Write the header line to the history output files (if not exist yet).
        for output_file_dict in output_files_phase:
            history_output_file_key = next(iter(output_file_dict))
            history_output_file = Path(output_file_dict[history_output_file_key]).absolute()
            # Create the parents directories of the history output files (if not exist yet).
            history_output_file.parent.mkdir(exist_ok=True, parents=True)
            header_line = None
            match history_output_file_key:
                case "selected_fit_clients_history_file":
                    # Base header columns.
                    header_columns = ["comm_round", "client_selector", "selection_duration", "num_available_clients",
                                      "available_clients", "num_selected_clients", "selected_clients"]
                    # Conditionally insert "num_tasks".
                    if is_task_based_selection:
                        header_columns.insert(3, "num_tasks")
                    # Build header line.
                    header_line = ",".join(header_columns) + "\n"
                case "individual_fit_metrics_history_file":
                    # Get the ordered set of fit metrics names.
                    fit_metrics_names = []
                    selected_fit_clients_metrics_history = selected_clients_metrics_history[1][phase]
                    clients_metrics_dicts = selected_fit_clients_metrics_history["clients_metrics_dicts"]
                    for client_metrics_dict in clients_metrics_dicts:
                        client_metrics = list(client_metrics_dict.values())[0]
                        fit_metrics_names.extend(client_metrics.keys())
                    fit_metrics_names = sorted(set(fit_metrics_names))
                    # Base header columns.
                    header_columns = ["comm_round", "client_selector", "num_available_clients", "client_id"]
                    # Conditionally insert "num_tasks".
                    if is_task_based_selection:
                        header_columns.insert(2, "num_tasks")
                    # Append metric names at the end.
                    header_columns.extend(fit_metrics_names)
                    # Build header line.
                    header_line = ",".join(header_columns) + "\n"
                case "metaheuristic_summary_fit_history_file":
                    header_line = ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n"
                                   .format("comm_round",
                                           "metaheuristic",
                                           "specific_metrics",
                                           "num_iterations",
                                           "elapsed_time",
                                           "initial_solution",
                                           "best_solution",
                                           "diff_solutions",
                                           "initial_solution_costs",
                                           "best_solution_costs",
                                           "diff_solutions_costs"))
                case "selected_evaluate_clients_history_file":
                    # Base header columns.
                    header_columns = ["comm_round", "client_selector", "selection_duration", "num_available_clients",
                                      "available_clients", "num_selected_clients", "selected_clients"]
                    # Conditionally insert "num_tasks".
                    if is_task_based_selection:
                        header_columns.insert(3, "num_tasks")
                    # Build header line.
                    header_line = ",".join(header_columns) + "\n"
                case "individual_evaluate_metrics_history_file":
                    # Get the ordered set of evaluate metrics names.
                    evaluate_metrics_names = []
                    selected_evaluate_clients_metrics_history = selected_clients_metrics_history[1][phase]
                    clients_metrics_dicts = selected_evaluate_clients_metrics_history["clients_metrics_dicts"]
                    for client_metrics_dict in clients_metrics_dicts:
                        client_metrics = list(client_metrics_dict.values())[0]
                        evaluate_metrics_names.extend(client_metrics.keys())
                    evaluate_metrics_names = sorted(set(evaluate_metrics_names))
                    # Base header columns.
                    header_columns = ["comm_round", "client_selector", "num_available_clients", "client_id"]
                    # Conditionally insert "num_tasks".
                    if is_task_based_selection:
                        header_columns.insert(2, "num_tasks")
                    # Append evaluation metric names at the end.
                    header_columns.extend(evaluate_metrics_names)
                    # Build header line.
                    header_line = ",".join(header_columns) + "\n"
                case "metaheuristic_summary_evaluate_history_file":
                    header_line = ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n"
                                   .format("comm_round",
                                           "metaheuristic",
                                           "specific_metrics",
                                           "num_iterations",
                                           "elapsed_time",
                                           "initial_solution",
                                           "best_solution",
                                           "diff_solutions",
                                           "initial_solution_costs",
                                           "best_solution_costs",
                                           "diff_solutions_costs"))
            if not history_output_file.exists() and header_line:
                with open(file=history_output_file, mode="a", encoding="utf-8") as file:
                    file.write(header_line)

    def _append_round_data_to_history_files(self,
                                            current_round: int,
                                            current_phase: str) -> None:
        # Get the necessary attributes.
        candidate_clients_history = self.get_attribute("_candidate_clients_history")
        selected_clients_history = self.get_attribute("_selected_clients_history")
        client_selection_duration_history = self.get_attribute("_client_selection_duration_history")
        selected_clients_metrics_history = self.get_attribute("_selected_clients_metrics_history")
        output_settings = self.get_attribute("_output_settings")
        fl_settings = self.get_attribute("_fl_settings")
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        client_selector = server_strategy_settings["strategy"]
        if "client_selector_{0}ing".format(current_phase) in server_strategy_settings:
            client_selector_name = server_strategy_settings["client_selector_{0}ing".format(current_phase)]["name"]
            client_selector = client_selector + "_" + client_selector_name
        if "initial_solution_generator_{0}ing".format(current_phase) in server_strategy_settings:
            initial_solution_generator_name = server_strategy_settings["initial_solution_generator_{0}ing".format(current_phase)]["name"]
            client_selector = client_selector + "_" + initial_solution_generator_name
            if "run_metaheuristic_{0}ing".format(current_phase) in server_strategy_settings:
                run_metaheuristic_phase = server_strategy_settings["run_metaheuristic_{0}ing".format(current_phase)]
                if run_metaheuristic_phase:
                    if "metaheuristic" in server_strategy_settings:
                        metaheuristic_name = server_strategy_settings["metaheuristic"]["name"]
                        client_selector = client_selector + "_" + metaheuristic_name
        round_timeout_in_seconds = fl_settings["round_timeout_in_seconds"]
        if round_timeout_in_seconds != "infinity":
            client_selector = client_selector + "_D{0}".format(round_timeout_in_seconds)
        phase_substrings = []
        is_task_based_selection = False
        if current_phase == "train":
            phase_substrings = ["fit", "train"]
            is_task_based_selection = "num_tasks_training" in server_strategy_settings
        elif current_phase == "test":
            phase_substrings = ["evaluate", "test"]
            is_task_based_selection = "num_tasks_testing" in server_strategy_settings
        output_files_phase = [{k: v} for k, v in output_settings.items()
                              if any(substring in k for substring in phase_substrings)]
        # Write the data line to the history output files.
        for output_file_dict in output_files_phase:
            history_output_file_key = next(iter(output_file_dict))
            history_output_file = Path(output_file_dict[history_output_file_key]).absolute()
            data_lines = []
            match history_output_file_key:
                case "selected_fit_clients_history_file":
                    selection_duration = client_selection_duration_history[current_round][current_phase]
                    num_available_clients = len(candidate_clients_history[current_round][current_phase])
                    available_clients = "|".join(list(candidate_clients_history[current_round][current_phase].keys()))
                    num_selected_clients = len(selected_clients_history[current_round][current_phase])
                    selected_clients = "|".join([client_id
                                                 for client_id, _ in selected_clients_history[current_round][current_phase].items()])
                    # Base data columns.
                    data_values = [str(current_round), str(client_selector), str(selection_duration),
                                   str(num_available_clients), str(available_clients if available_clients else None),
                                   str(num_selected_clients), str(selected_clients if selected_clients else None)]
                    # Conditionally insert "num_tasks".
                    if is_task_based_selection:
                        num_tasks = sum([client_info["client_num_tasks_scheduled"]
                                         for _, client_info in
                                         selected_clients_history[current_round][current_phase].items()])
                        data_values.insert(3, str(num_tasks))
                    # Build data line.
                    data_line = ",".join(data_values) + "\n"
                    data_lines.append(data_line)
                case "individual_fit_metrics_history_file":
                    current_round_values = selected_clients_metrics_history[current_round][current_phase]
                    fit_metrics_names = []
                    clients_metrics_dicts = current_round_values["clients_metrics_dicts"]
                    for client_metrics_dict in clients_metrics_dicts:
                        client_metrics = list(client_metrics_dict.values())[0]
                        fit_metrics_names.extend(client_metrics.keys())
                    fit_metrics_names = sorted(set(fit_metrics_names))
                    num_available_clients = current_round_values["num_available_clients"]
                    clients_metrics_dicts = current_round_values["clients_metrics_dicts"]
                    clients_metrics_dicts = sorted(clients_metrics_dicts, key=lambda x: list(x.keys()))
                    for client_metrics_dict in clients_metrics_dicts:
                        client_id_str = list(client_metrics_dict.keys())[0]
                        client_metrics = list(client_metrics_dict.values())[0]
                        fit_metrics_values = []
                        for fit_metric_name in fit_metrics_names:
                            fit_metric_value = "N/A"
                            if fit_metric_name in client_metrics:
                                fit_metric_value = str(client_metrics[fit_metric_name])
                            fit_metrics_values.append(fit_metric_value)
                        # Base data columns.
                        data_values = [str(current_round), str(client_selector), str(num_available_clients),
                                       str(client_id_str)]
                        # Conditionally insert "num_tasks".
                        if is_task_based_selection:
                            num_tasks = current_round_values["num_tasks"]
                            data_values.insert(2, str(num_tasks))
                        # Append metrics at the end.
                        data_values.extend(map(str, fit_metrics_values))
                        # Build the data line.
                        data_line = ",".join(data_values) + "\n"
                        data_lines.append(data_line)
                case "metaheuristic_summary_fit_history_file":
                    metaheuristic_summary_fit_history = {}  # TODO
                    if current_round in metaheuristic_summary_fit_history:
                        mh_summary = metaheuristic_summary_fit_history[current_round]
                        metaheuristic = mh_summary["metaheuristic"]
                        specific_metrics = mh_summary["specific_metrics"]
                        num_iterations = mh_summary["num_iterations"]
                        elapsed_time = mh_summary["elapsed_time"]
                        initial_solution = mh_summary["initial_solution"]
                        best_solution = mh_summary["best_solution"]
                        diff_solutions = mh_summary["diff_solutions"]
                        initial_solution_costs = mh_summary["initial_solution_costs"]
                        best_solution_costs = mh_summary["best_solution_costs"]
                        diff_solutions_costs = mh_summary["diff_solutions_costs"]
                        data_line = ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n"
                                     .format(current_round,
                                             metaheuristic,
                                             specific_metrics,
                                             num_iterations,
                                             elapsed_time,
                                             initial_solution,
                                             best_solution,
                                             diff_solutions,
                                             initial_solution_costs,
                                             best_solution_costs,
                                             diff_solutions_costs))
                        data_lines.append(data_line)
                case "selected_evaluate_clients_history_file":
                    selection_duration = client_selection_duration_history[current_round][current_phase]
                    num_available_clients = len(candidate_clients_history[current_round][current_phase])
                    available_clients = "|".join(list(candidate_clients_history[current_round][current_phase].keys()))
                    num_selected_clients = len(selected_clients_history[current_round][current_phase])
                    selected_clients = "|".join([client_id
                                                 for client_id, _ in selected_clients_history[current_round][current_phase].items()])
                    # Base data columns.
                    data_values = [str(current_round), str(client_selector), str(selection_duration),
                                   str(num_available_clients), str(available_clients if available_clients else None),
                                   str(num_selected_clients), str(selected_clients if selected_clients else None)]
                    # Conditionally insert "num_tasks".
                    if is_task_based_selection:
                        num_tasks = sum([client_info["client_num_tasks_scheduled"]
                                         for _, client_info in
                                         selected_clients_history[current_round][current_phase].items()])
                        data_values.insert(3, str(num_tasks))
                    # Build data line.
                    data_line = ",".join(data_values) + "\n"
                    data_lines.append(data_line)
                case "individual_evaluate_metrics_history_file":
                    current_round_values = selected_clients_metrics_history[current_round][current_phase]
                    evaluate_metrics_names = []
                    clients_metrics_dicts = current_round_values["clients_metrics_dicts"]
                    for client_metrics_dict in clients_metrics_dicts:
                        client_metrics = list(client_metrics_dict.values())[0]
                        evaluate_metrics_names.extend(client_metrics.keys())
                    evaluate_metrics_names = sorted(set(evaluate_metrics_names))
                    num_available_clients = current_round_values["num_available_clients"]
                    clients_metrics_dicts = current_round_values["clients_metrics_dicts"]
                    clients_metrics_dicts = sorted(clients_metrics_dicts, key=lambda x: list(x.keys()))
                    for client_metrics_dict in clients_metrics_dicts:
                        client_id_str = list(client_metrics_dict.keys())[0]
                        client_metrics = list(client_metrics_dict.values())[0]
                        evaluate_metrics_values = []
                        for evaluate_metric_name in evaluate_metrics_names:
                            evaluate_metric_value = "N/A"
                            if evaluate_metric_name in client_metrics:
                                evaluate_metric_value = str(client_metrics[evaluate_metric_name])
                            evaluate_metrics_values.append(evaluate_metric_value)
                        # Base data columns.
                        data_values = [str(current_round), str(client_selector), str(num_available_clients),
                                       str(client_id_str)]
                        # Conditionally insert "num_tasks".
                        if is_task_based_selection:
                            num_tasks = current_round_values["num_tasks"]
                            data_values.insert(2, str(num_tasks))
                        # Append evaluation metric values at the end.
                        data_values.extend(map(str, evaluate_metrics_values))
                        # Build data line.
                        data_line = ",".join(data_values) + "\n"
                        data_lines.append(data_line)
                case "metaheuristic_summary_evaluate_history_file":
                    metaheuristic_summary_evaluate_history = {}  # TODO
                    if current_round in metaheuristic_summary_evaluate_history:
                        mh_summary = metaheuristic_summary_evaluate_history[current_round]
                        metaheuristic = mh_summary["metaheuristic"]
                        specific_metrics = mh_summary["specific_metrics"]
                        num_iterations = mh_summary["num_iterations"]
                        elapsed_time = mh_summary["elapsed_time"]
                        initial_solution = mh_summary["initial_solution"]
                        best_solution = mh_summary["best_solution"]
                        diff_solutions = mh_summary["diff_solutions"]
                        initial_solution_costs = mh_summary["initial_solution_costs"]
                        best_solution_costs = mh_summary["best_solution_costs"]
                        diff_solutions_costs = mh_summary["diff_solutions_costs"]
                        data_line = ("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n"
                                     .format(current_round,
                                             metaheuristic,
                                             specific_metrics,
                                             num_iterations,
                                             elapsed_time,
                                             initial_solution,
                                             best_solution,
                                             diff_solutions,
                                             initial_solution_costs,
                                             best_solution_costs,
                                             diff_solutions_costs))
                        data_lines.append(data_line)
            if history_output_file.exists() and data_lines:
                with open(file=history_output_file, mode="a", encoding="utf-8") as file:
                    file.writelines(data_lines)

    def _update_clients_reliability_score_history(self,
                                                  current_round: int,
                                                  current_phase: str,
                                                  completed_clients: dict | None = None) -> None:
        # Get the necessary attributes.
        candidate_clients_history = self.get_attribute("_candidate_clients_history")
        selected_clients_history = self.get_attribute("_selected_clients_history")
        clients_reliability_score_history = self.get_attribute("_clients_reliability_score_history")
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        clients_reliability_score = server_strategy_settings["clients_reliability_score"]
        non_availability_penalty = clients_reliability_score["non_availability_penalty"]
        non_completion_penalty = clients_reliability_score["non_completion_penalty"]
        recency_weight = clients_reliability_score["recency_weight"]
        if completed_clients is None:
            completed_clients = {}
        available_clients = candidate_clients_history.get(current_round, {}).get(current_phase, {})
        selected_clients = selected_clients_history.get(current_round, {}).get(current_phase, {})
        # Create round entry.
        if current_round not in clients_reliability_score_history:
            clients_reliability_score_history[current_round] = {}
        current_scores = clients_reliability_score_history[current_round]
        # Get the previous round reliability scores.
        prev_scores = clients_reliability_score_history.get(current_round - 1, {})
        # All clients that we may need to compute scores for.
        all_client_ids = (set(prev_scores.keys())
                        | set(available_clients.keys())
                        | set(selected_clients.keys())
                        | set(completed_clients.keys()))
        all_client_ids = {cid for cid in all_client_ids if cid.startswith("client_")}
        for client_id in all_client_ids:
            # Previous reliability score (default 1.0 for new clients).
            prev_score = prev_scores.get(client_id, 1.0)
            # Compute penalty p_i(r).
            if client_id in available_clients:
                # Client was available but not selected.
                if client_id not in selected_clients:
                    p_ir = 0.0  # No penalty.
                else:
                    # Client was available and selected. Outcome depends on completion input.
                    completed = completed_clients.get(client_id, False)
                    p_ir = 0.0 if completed else non_completion_penalty # No penalty if completed.
            else:
                # Client was not available.
                p_ir = non_availability_penalty
            # Recency-weighted update.
            new_score = prev_score * (1 - recency_weight) + (1 - p_ir) * recency_weight
            current_scores[client_id] = new_score
        # Store the updated client reliability scores.
        self._set_attribute("_clients_reliability_score_history", clients_reliability_score_history)

    def _aggregate_evaluate_metrics(self,
                                    current_round: int,
                                    evaluate_metrics: list[tuple[int, Metrics]]) -> Optional[Metrics]:
        # Get the necessary attributes.
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        metrics_aggregator = server_strategy_settings["metrics_aggregator"]
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Set the phase value.
        phase = "test"
        # Update the individual testing metrics history.
        self._update_individual_metrics_history(current_round, phase, evaluate_metrics)
        # Keep only numeric metrics for the aggregation.
        evaluate_metrics = self._keep_numeric_metrics_only(evaluate_metrics)
        # Remove the undesired metrics, if any.
        if "undesired_metrics_for_evaluate_aggregation" in server_strategy_settings:
            undesired_metrics_for_evaluate_aggregation = server_strategy_settings["undesired_metrics_for_evaluate_aggregation"]
            if undesired_metrics_for_evaluate_aggregation:
                evaluate_metrics = self._remove_undesired_metrics(evaluate_metrics, undesired_metrics_for_evaluate_aggregation)
        # Initialize the aggregated testing metrics dictionary (aggregated_evaluate_metrics).
        aggregated_evaluate_metrics = {}
        # Aggregate the testing metrics according to the user-defined aggregator.
        if metrics_aggregator == "Weighted_Average":
            aggregated_evaluate_metrics = aggregate_metrics_by_weighted_average(evaluate_metrics)
        # Get the number of participating clients.
        num_participating_clients = len(evaluate_metrics)
        num_participating_clients_str = "".join([str(num_participating_clients),
                                                 " Clients" if num_participating_clients > 1 else " Client"])
        # Log the aggregated testing metrics.
        message = "[Server {0} | Round {1}] Aggregated testing metrics ({2} of {3}): {4}" \
                  .format(server_id, current_round, metrics_aggregator, num_participating_clients_str,
                          aggregated_evaluate_metrics)
        log_message(logger, message, "DEBUG")
        # Return the aggregated testing metrics (aggregated_evaluate_metrics).
        return aggregated_evaluate_metrics

    @staticmethod
    def _evaluate_centrally_on_server(current_round: int,
                                      model_parameters: NDArrays,
                                      evaluate_config: dict) -> Optional[Metrics]:
        return None

    def initialize_parameters(self,
                              client_manager: ClientManager | None) -> Optional[Parameters]:
        """Initialize the model parameters.
           \nCalled by Flower before the first round.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the initial parameters.
        initial_parameters = self.get_attribute("_initial_parameters")
        # Discard it from the memory.
        self._set_attribute("_initial_parameters", None)
        # Return the initial parameters.
        return initial_parameters

    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager | None) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
           \nCalled by Flower before each training phase.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        fl_settings = self.get_attribute("_fl_settings")
        num_rounds = fl_settings["num_rounds"]
        enable_training = fl_settings["enable_training"]
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        selected_clients_history = self.get_attribute("_selected_clients_history")
        selected_clients_metrics_history = self.get_attribute("_selected_clients_metrics_history")
        client_selector = self.get_attribute("_client_selector")
        # Log a 'start of the configure_fit call' debug message.
        message = "[Server {0} | Round {1}] Start of the 'configure_fit' call!".format(server_id, server_round)
        log_message(logger, message, "DEBUG")
        # Steps to be done before starting the first round:
        if server_round == 1:
            # Log the global model weights' sum (initial weights).
            message = ("[Server {0} | Round {1}] Global model weight's sum (initial weights): {2}"
                       .format(self.get_attribute("_server_id"), server_round, numpy_sum(parameters_to_ndarrays(parameters)[0])))
            log_message(self._logger, message, "DEBUG")
            # Store the initial and global parameters.
            self._set_attribute("_initial_parameters", parameters)
            self._set_attribute("_global_parameters", parameters)
            # Wait for the initial clients to connect.
            wait_for_initial_clients = fl_settings["wait_for_initial_clients"]
            num_clients_to_wait = wait_for_initial_clients["num_clients_to_wait"]
            waiting_timeout_in_seconds = wait_for_initial_clients["waiting_timeout_in_seconds"]
            if waiting_timeout_in_seconds == "infinity":
                waiting_timeout_in_seconds = None
            if client_manager is not None:
                client_manager.wait_for(num_clients_to_wait, waiting_timeout_in_seconds)
        # Verify if the FL execution should stop.
        if server_round > 1:
            fl_execution_should_stop = self._fl_execution_should_stop(server_round)
            # Verify if the federated training configuration should not proceed.
            if not enable_training or (enable_training and fl_execution_should_stop):
                return []
        # Get the list of profiling rounds, if any.
        profiling_rounds = self.get_attribute("_profiling_rounds")
        # Set the phase value.
        phase = "train"
        # Get the set of active (candidate) clients.
        candidate_clients = self._map_available_clients(server_round, client_manager)
        # Get the base training instructions to be used by the selected clients.
        fit_config = self.get_attribute("_fit_config")
        base_learning_rate = fit_config["learning_rate"]
        base_batch_size = fit_config["batch_size"]
        base_num_epochs = fit_config["epochs"]
        # Get the round time limit (deadline).
        round_timeout_in_seconds = fl_settings["round_timeout_in_seconds"]
        if round_timeout_in_seconds == "infinity":
            round_timeout_in_seconds = inf
        # Start the clients' selection duration timer.
        selection_duration_start = process_time()
        # Set the client selection procedure kwargs.
        kwargs = {"current_round": server_round,
                  "current_phase": phase,
                  "candidate_clients": candidate_clients,
                  "num_rounds": num_rounds,
                  "base_learning_rate": base_learning_rate,
                  "base_batch_size": base_batch_size,
                  "base_num_epochs": base_num_epochs,
                  "selected_clients_history": selected_clients_history,
                  "selected_clients_metrics_history": selected_clients_metrics_history,
                  "profiling_rounds": profiling_rounds,
                  "time_limit": round_timeout_in_seconds,
                  "logger": logger}
        if server_strategy_settings.get("monitor_clients_reliability_score", False):
            # Get the clients' reliability score history.
            clients_reliability_score_history = self.get_attribute("_clients_reliability_score_history")
            kwargs.update({"clients_reliability_score_history": clients_reliability_score_history})
        if "num_tasks_training" in server_strategy_settings:
            # Get the number of tasks to be scheduled to the selected clients.
            num_tasks = server_strategy_settings["num_tasks_training"]
            kwargs.update({"num_tasks": num_tasks})
        if "samples_per_task" in server_strategy_settings:
            # Get the number of samples per task.
            samples_per_task = server_strategy_settings["samples_per_task"]
            kwargs.update({"samples_per_task": samples_per_task})
        if "data_privacy_approach" in server_strategy_settings:
            # Get the privacy approach.
            data_privacy_approach = server_strategy_settings["data_privacy_approach"]["name"]
            kwargs.update({"data_privacy_approach": data_privacy_approach})
        # Run the client selection procedure.
        selected_clients = client_selector.run_client_selection_procedure(**kwargs)
        if isinstance(selected_clients, list):
            # Monitor the future objects in a daemon thread (non-blocking).
            monitor_future_objects_thread = Thread(target=self._monitor_future_objects,
                                                   args=(selected_clients, phase, candidate_clients, parameters),
                                                   daemon=True)
            monitor_future_objects_thread.start()
        else:
            # Get the clients' selection duration.
            selection_duration_in_seconds = process_time() - selection_duration_start
            # Log a 'clients' selection duration' message.
            message = "[Server {0} | Round {1}] The client selection procedure took {2} seconds!" \
                      .format(server_id, server_round, round(selection_duration_in_seconds, 2))
            log_message(logger, message, "INFO")
            # Run the post-client selection tasks.
            self._run_post_client_selection_tasks(server_round,
                                                  phase,
                                                  candidate_clients,
                                                  selected_clients,
                                                  selection_duration_in_seconds,
                                                  parameters)
        # Wait for the selection of training clients for the current round.
        while True:
            fit_pairs_history = self.get_attribute("_fit_pairs_history")
            if server_round in fit_pairs_history:
                fit_pairs = fit_pairs_history[server_round]
                break
            sleep(1)
        # Log an 'end of the configure_fit call' debug message.
        message = "[Server {0} | Round {1}] End of the 'configure_fit' call!".format(server_id, server_round)
        log_message(logger, message, "DEBUG")
        # Log the start of the current communication round.
        if fit_pairs:
            message = "[Server {0} | Round {1}] Starting the {2}ing phase...".format(server_id, server_round, phase)
            log_message(logger, message, "INFO")
        # Return the list of (fit_client_proxy, fit_client_instructions) pairs.
        return fit_pairs

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) \
            -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate the model parameters and the training metrics based on the fit_results.
           \nCalled by Flower after each training phase.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        fl_settings = self.get_attribute("_fl_settings")
        accept_clients_failures = fl_settings["accept_clients_failures"]
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        model_aggregator = server_strategy_settings["model_aggregator"]
        model_aggregator_name = model_aggregator["name"]
        logger = self.get_attribute("_logger")
        # Log a 'start of the aggregate_fit call' debug message.
        message = "[Server {0} | Round {1}] Start of the 'aggregate_fit' call!".format(server_id, server_round)
        log_message(logger, message, "DEBUG")
        # Log a 'received messages statistics' message.
        message = "[Server {0} | Round {1} | Training Phase] Received {2} results and {3} failures." \
            .format(self.get_attribute("_server_id"),
                    server_round,
                    len(results),
                    len(failures))
        log_message(logger, message, "INFO")
        # Do not aggregate if there are no results or if there are clients' failures and failures are not accepted.
        if not results or (failures and not accept_clients_failures):
            return None, {}
        # Log the global model weights' sum (before aggregation).
        global_parameters = self.get_attribute("_global_parameters")
        message = ("[Server {0} | Round {1}] Global model weight's sum (before aggregation): {2}"
                   .format(self.get_attribute("_server_id"), server_round, numpy_sum(parameters_to_ndarrays(global_parameters)[0])))
        log_message(self._logger, message, "DEBUG")
        # Initialize the aggregated model parameters.
        aggregated_model_parameters = []
        match model_aggregator_name:
            case "FedAvg":
                # Aggregate the model parameters using the FedAvg approach.
                inplace_aggregation = model_aggregator["inplace_aggregation"]
                aggregated_model_parameters = aggregate_parameters_with_fed_avg(results, inplace_aggregation)
            case "FedAvgE":
                # Aggregate the model parameters using the FedAvgE approach.
                aggregated_model_parameters = aggregate_parameters_with_fed_avg_e(results)
            case "FedProx":
                # Aggregate the model parameters using the FedProx approach.
                global_parameters = self.get_attribute("_global_parameters")
                mu = model_aggregator["mu"]
                aggregated_model_parameters = aggregate_parameters_with_fed_prox(global_parameters, results, mu)
        # Log the global model weights' sum (after aggregation).
        message = ("[Server {0} | Round {1}] Global model weight's sum (after aggregation): {2}"
                   .format(self.get_attribute("_server_id"), server_round, numpy_sum(parameters_to_ndarrays(aggregated_model_parameters)[0])))
        log_message(self._logger, message, "DEBUG")
        # Aggregate the training metrics.
        fit_metrics = [(result.num_examples, result.metrics) for _, result in results]
        aggregated_fit_metrics = self._aggregate_fit_metrics(server_round, fit_metrics)
        # Set the phase value.
        phase = "train"
        # Initialize the training history files after finishing the first round, if needed.
        if server_round == 1:
            self._initialize_history_output_files(phase)
        # Append the communication round data to the training history files.
        self._append_round_data_to_history_files(server_round, phase)
        # Store the improved global parameters.
        self._set_attribute("_global_parameters", aggregated_model_parameters)
        # Update the clients reliability score, if being monitored.
        if server_strategy_settings.get("monitor_clients_reliability_score", False):
            completed_clients = {"client_{0}".format(result.metrics["client_id"]): True for _, result in results}
            self._update_clients_reliability_score_history(server_round, phase, completed_clients)
        # Log an 'end of the aggregate_fit call' debug message.
        message = "[Server {0} | Round {1}] End of the 'aggregate_fit' call!".format(server_id, server_round)
        log_message(logger, message, "DEBUG")
        # Return the aggregated model parameters and aggregated training metrics.
        return aggregated_model_parameters, aggregated_fit_metrics

    def configure_evaluate(self,
                           server_round: int,
                           parameters: Parameters,
                           client_manager: ClientManager | None) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of testing.
           \nCalled by Flower before each testing phase.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        fl_settings = self.get_attribute("_fl_settings")
        num_rounds = fl_settings["num_rounds"]
        enable_testing = fl_settings["enable_testing"]
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        selected_clients_history = self.get_attribute("_selected_clients_history")
        selected_clients_metrics_history = self.get_attribute("_selected_clients_metrics_history")
        client_selector = self.get_attribute("_client_selector")
        # Log a 'start of the configure_evaluate call' debug message.
        message = "[Server {0} | Round {1}] Start of the 'configure_evaluate' call!".format(server_id, server_round)
        log_message(logger, message, "DEBUG")
        # Verify if the FL execution should stop.
        if server_round > 1:
            fl_execution_should_stop = self._fl_execution_should_stop(server_round)
            # Verify if the federated testing configuration should not proceed.
            if not enable_testing or (enable_testing and fl_execution_should_stop):
                return []
        # Get the list of profiling rounds, if any.
        profiling_rounds = self.get_attribute("_profiling_rounds")
        # Set the phase value.
        phase = "test"
        # Get the set of active (candidate) clients.
        candidate_clients = self._map_available_clients(server_round, client_manager)
        # Get the base testing instructions to be used by the selected clients.
        evaluate_config = self.get_attribute("_evaluate_config")
        base_batch_size = evaluate_config["batch_size"]
        # Get the round time limit (deadline).
        round_timeout_in_seconds = fl_settings["round_timeout_in_seconds"]
        if round_timeout_in_seconds == "infinity":
            round_timeout_in_seconds = inf
        # Start the clients' selection duration timer.
        selection_duration_start = process_time()
        # Set the client selection procedure kwargs.
        kwargs = {"current_round": server_round,
                  "current_phase": phase,
                  "candidate_clients": candidate_clients,
                  "num_rounds": num_rounds,
                  "base_batch_size": base_batch_size,
                  "selected_clients_history": selected_clients_history,
                  "selected_clients_metrics_history": selected_clients_metrics_history,
                  "profiling_rounds": profiling_rounds,
                  "time_limit": round_timeout_in_seconds,
                  "logger": logger}
        if server_strategy_settings.get("monitor_clients_reliability_score", False):
            # Get the clients' reliability score history.
            clients_reliability_score_history = self.get_attribute("_clients_reliability_score_history")
            kwargs.update({"clients_reliability_score_history": clients_reliability_score_history})
        if "num_tasks_testing" in server_strategy_settings:
            # Get the number of tasks to be scheduled to the selected clients.
            num_tasks = server_strategy_settings["num_tasks_testing"]
            kwargs.update({"num_tasks": num_tasks})
        if "samples_per_task" in server_strategy_settings:
            # Get the number of samples per task.
            samples_per_task = server_strategy_settings["samples_per_task"]
            kwargs.update({"samples_per_task": samples_per_task})
        if "data_privacy_approach" in server_strategy_settings:
            # Get the privacy approach.
            data_privacy_approach = server_strategy_settings["data_privacy_approach"]["name"]
            kwargs.update({"data_privacy_approach": data_privacy_approach})
        # Run the client selection procedure.
        selected_clients = client_selector.run_client_selection_procedure(**kwargs)
        if isinstance(selected_clients, list):
            # Monitor the future objects in a daemon thread (non-blocking).
            monitor_future_objects_thread = Thread(target=self._monitor_future_objects,
                                                   args=(selected_clients, phase, candidate_clients, parameters),
                                                   daemon=True)
            monitor_future_objects_thread.start()
        else:
            # Get the clients' selection duration.
            selection_duration_in_seconds = process_time() - selection_duration_start
            # Log a 'clients' selection duration' message.
            message = "[Server {0} | Round {1}] The client selection procedure took {2} seconds!" \
                      .format(server_id, server_round, round(selection_duration_in_seconds, 2))
            log_message(logger, message, "INFO")
            # Run the post-client selection tasks.
            self._run_post_client_selection_tasks(server_round,
                                                  phase,
                                                  candidate_clients,
                                                  selected_clients,
                                                  selection_duration_in_seconds,
                                                  parameters)
        # Wait for the selection of testing clients for the current round.
        while True:
            evaluate_pairs_history = self.get_attribute("_evaluate_pairs_history")
            if server_round in evaluate_pairs_history:
                evaluate_pairs = evaluate_pairs_history[server_round]
                break
            sleep(1)
        # Log an 'end of the configure_evaluate call' debug message.
        message = "[Server {0} | Round {1}] End of the 'configure_evaluate' call!".format(server_id, server_round)
        log_message(logger, message, "DEBUG")
        # Log the start of the current communication round.
        if evaluate_pairs:
            message = "[Server {0} | Round {1}] Starting the {2}ing phase...".format(server_id, server_round, phase)
            log_message(logger, message, "INFO")
        # Return the list of (evaluate_client_proxy, evaluate_client_instructions) pairs.
        return evaluate_pairs

    def aggregate_evaluate(self,
                           server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) \
            -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the testing metrics based on the evaluate_results.
           \nCalled by Flower after each testing phase.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        fl_settings = self.get_attribute("_fl_settings")
        accept_clients_failures = fl_settings["accept_clients_failures"]
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        logger = self.get_attribute("_logger")
        # Log a 'start of the aggregate_evaluate call' debug message.
        message = "[Server {0} | Round {1}] Start of the 'aggregate_evaluate' call!".format(server_id, server_round)
        log_message(logger, message, "DEBUG")
        # Log a 'received messages statistics' message.
        message = "[Server {0} | Round {1} | Test Phase] Received {2} results and {3} failures." \
            .format(self.get_attribute("_server_id"),
                    server_round,
                    len(results),
                    len(failures))
        log_message(logger, message, "INFO")
        # Do not aggregate if there are no results or if there are clients' failures and failures are not accepted.
        if not results or (failures and not accept_clients_failures):
            return None, {}
        # Aggregate the loss by weighted average.
        aggregated_loss = aggregate_loss_by_weighted_average(results)
        # Aggregate the testing metrics.
        evaluate_metrics = [(result.num_examples, result.metrics) for _, result in results]
        aggregated_evaluate_metrics = self._aggregate_evaluate_metrics(server_round, evaluate_metrics)
        # Set the phase value.
        phase = "test"
        # Initialize the testing history files after finishing the first round, if needed.
        if server_round == 1:
            self._initialize_history_output_files(phase)
        # Append the communication round data to the testing history files.
        self._append_round_data_to_history_files(server_round, phase)
        # Update the clients reliability score, if being monitored.
        if server_strategy_settings.get("monitor_clients_reliability_score", False):
            completed_clients = {"client_{0}".format(result.metrics["client_id"]): True for _, result in results}
            self._update_clients_reliability_score_history(server_round, phase, completed_clients)
        # Log an 'end of the aggregate_evaluate call' debug message.
        message = "[Server {0} | Round {1}] End of the 'aggregate_evaluate' call!".format(server_id, server_round)
        log_message(logger, message, "DEBUG")
        # Return the aggregated loss and aggregated testing metrics.
        return aggregated_loss, aggregated_evaluate_metrics

    def evaluate(self,
                 server_round: int,
                 parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate in a centralized fashion the updated model parameters.
           \nCalled by Flower after each training phase.
           \nRequires a local testing dataset on the server.
           \nThe evaluation outcome will be stored in the 'losses_centralized' and 'metrics_centralized' dictionaries.
           \nImplementation of the abstract method from the Strategy class."""
        # Convert the model parameters to NumPy ndarrays.
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        # Evaluate the model parameters centrally on the server.
        eval_res = self._evaluate_centrally_on_server(server_round, parameters_ndarrays, {})
        # Verify if the outcome is empty.
        if eval_res is None:
            return None
        # If not, get the centralized loss and centralized testing metrics.
        centralized_loss, centralized_evaluate_metrics = eval_res
        # Return the centralized loss and centralized testing metrics.
        return centralized_loss, centralized_evaluate_metrics
