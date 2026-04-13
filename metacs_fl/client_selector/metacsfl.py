from copy import deepcopy
from logging import Logger
from numpy.random import default_rng, SeedSequence
from pathlib import Path

from metacs_fl.metaheuristic.lns import run_lns
from metacs_fl.task_scheduler.ecmtc import ecmtc
from metacs_fl.task_scheduler.mec import mec
from metacs_fl.task_scheduler.random import random_selection
from metacs_fl.utils.client_selector_util import get_all_possible_sums, take_closest
from metacs_fl.utils.logger_util import log_message
from metacs_fl.utils.system_modeler_util import calculate_computation_time, calculate_computation_energy, \
    calculate_download_time, calculate_download_energy, calculate_upload_time, calculate_upload_energy
from metacs_fl.utils.task_scheduler_util import adjust_batch_sizes, adjust_learning_rates, adjust_num_epochs, \
    build_class_capacity_vectors_list, calculate_normalized_client_diversity_score_over_past_x_rounds, \
    calculate_percentage_change, distribute_tasks_with_globally_balanced_approach, \
    distribute_tasks_with_locally_balanced_approach, distribute_tasks_with_random_approach, organize_tasks_distribution


class MetaCSFL:

    def __init__(self,
                 client_selection_settings: dict,
                 seed: None | int | SeedSequence) -> None:
        self._client_selection_settings = client_selection_settings
        self._initial_solution_generation_history = {}
        self._internal_candidate_clients_history = {}
        self._internal_selected_clients_history = {}
        self._task_assignment_capacities_list = []
        # Initialize the random number generator with a fixed seed to allow replicable results.
        self._rng = default_rng(seed=seed)

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    def _get_trimmed_task_assignment_capacities(task_assignment_capacities_list: list,
                                                clients_reliability_score_history: dict) -> list:
        if not clients_reliability_score_history:
            latest_client_reliability_scores = {}
        else:
            latest_key = max(clients_reliability_score_history.keys())
            latest_client_reliability_scores = clients_reliability_score_history[latest_key]
        trimmed_task_assignment_capacities_list = []
        for idx, capacities in enumerate(task_assignment_capacities_list):
            client_id_str = "client_{0}".format(idx)
            client_score = latest_client_reliability_scores.get(client_id_str, 1.0)  # Default to full capacity.
            client_task_capacities_cutoff = int(len(capacities) * client_score)
            trimmed_task_assignment_capacities_list.append(capacities[:client_task_capacities_cutoff])
        return trimmed_task_assignment_capacities_list

    @staticmethod
    def _get_latest_selection(selected_clients_history: dict,
                              current_round: int,
                              current_phase: str)-> dict:
        selected_clients = {}
        previous_round = current_round - 1
        if previous_round in selected_clients_history and current_phase in selected_clients_history[previous_round]:
            selected_clients = selected_clients_history[previous_round][current_phase]
        return selected_clients

    @staticmethod
    def _get_metrics_of_past_x_rounds(selected_clients_metrics_history: dict,
                                      current_round: int,
                                      phase_of_interest: str,
                                      x: int) -> dict:
        # Get the necessary attributes.
        last_round_on_history = list(selected_clients_metrics_history.items())[-1][0]
        if current_round > last_round_on_history:
            current_round = last_round_on_history + 1
        # Get the keys of the past x rounds.
        past_rounds_keys = [r_idx for r_idx in list(range(current_round - x, current_round)) if r_idx > 0]
        # Filter the selected clients metrics history, preserving only the past x rounds.
        selected_clients_metrics_history_filtered = dict((past_round, v) for past_round, v in selected_clients_metrics_history.items()
                                                         if past_round in past_rounds_keys)
        # Initialize the metrics lists.
        makespans = []
        energy_consumptions = []
        weighted_mean_accuracies = []
        # Get the clients' metrics of the past x rounds.
        for past_round, _ in selected_clients_metrics_history_filtered.items():
            clients_metrics_dicts = {}
            if phase_of_interest in selected_clients_metrics_history_filtered[past_round]:
                clients_metrics_dicts = selected_clients_metrics_history_filtered[past_round][phase_of_interest]["clients_metrics_dicts"]
            makespan = 0
            energy_consumption = 0
            sum_accuracy_product = 0
            sum_num_examples_used = 0
            for client_metrics_dict in clients_metrics_dicts:
                client_id = next(iter(client_metrics_dict))
                client_metrics = client_metrics_dict[client_id]
                # Get the number of tasks executed by the client i on round r.
                num_examples_key = "num_examples"
                x_i = client_metrics[num_examples_key]
                # Get the time cost of the client i on round r (if available).
                time_key = "{0}ing_time_in_seconds".format(phase_of_interest)
                time_i = client_metrics[time_key] if time_key in client_metrics else 0
                # Update the makespan of round r.
                if time_i > makespan:
                    makespan = time_i
                # Get the energy cost of the client i on round r (if available).
                energy_key = "{0}ing_energy_in_joules".format(phase_of_interest)
                energy_i = client_metrics[energy_key] if energy_key in client_metrics else 0
                # Update the energy consumption of round r.
                energy_consumption += energy_i
                # Get the accuracy "cost" of the client i on round r (if available).
                accuracy_key = "accuracy"
                accuracy_i = 0
                for metric_key, _ in client_metrics.items():
                    if accuracy_key in metric_key:
                        accuracy_i = client_metrics[metric_key]
                # Accumulate auxiliary values for the weighted mean accuracy calculation.
                sum_accuracy_product += x_i * accuracy_i
                sum_num_examples_used += x_i
            # Update the weighted mean accuracy of round r.
            weighted_mean_accuracy = sum_accuracy_product / sum_num_examples_used if sum_num_examples_used > 0 else 0
            # Update the metrics lists.
            makespans.append(makespan)
            energy_consumptions.append(energy_consumption)
            weighted_mean_accuracies.append(weighted_mean_accuracy)
        # Set the dictionary of the metrics lists.
        metrics_past_rounds = {"makespans": makespans,
                               "energy_consumptions": energy_consumptions,
                               "weighted_mean_accuracies": weighted_mean_accuracies}
        # Return the dictionary of the metrics lists.
        return metrics_past_rounds

    @staticmethod
    def _get_profile_loss_for_phase(client_id: str,
                                    candidate_clients: dict,
                                    clients_profiles: dict,
                                    phase: str) -> float | None:
        if client_id not in candidate_clients:
            return None
        if client_id not in clients_profiles:
            return None
        if phase not in clients_profiles[client_id]:
            return None
        profile_phase_dict = clients_profiles[client_id][phase]
        if not profile_phase_dict:
            return None
        task_assignment_capacities_phase_key = "client_task_assignment_capacities_{0}".format(phase)
        task_assignment_capacities_i = candidate_clients[client_id][task_assignment_capacities_phase_key]
        max_ac = max(task_assignment_capacities_i)
        closest_x = min(profile_phase_dict.keys(), key=lambda x: abs(int(x) - max_ac))
        profiled_metrics = profile_phase_dict[closest_x]
        if "loss" not in profiled_metrics:
            return None
        return profiled_metrics["loss"]

    @staticmethod
    def _normalize_value(value: float,
                         min_value: float,
                         max_value: float) -> float:
        if max_value <= min_value:
            return 0.0
        normalized_value = (value - min_value) / (max_value - min_value)
        if normalized_value < 0:
            return 0.0
        if normalized_value > 1:
            return 1.0
        return normalized_value

    def _collect_client_losses_from_history(self,
                                            selected_clients_metrics_history: dict,
                                            candidate_clients: dict,
                                            clients_profiles: dict) -> tuple:
        candidate_client_ids = list(candidate_clients.keys())
        train_losses_by_client = {client_id: [] for client_id in candidate_client_ids}
        test_losses_by_client = {client_id: [] for client_id in candidate_client_ids}
        all_train_losses = []
        all_test_losses = []
        for round_idx in sorted(selected_clients_metrics_history.keys()):
            round_metrics = selected_clients_metrics_history[round_idx]
            # Training losses.
            if "train" in round_metrics and "clients_metrics_dicts" in round_metrics["train"]:
                clients_metrics_dicts = round_metrics["train"]["clients_metrics_dicts"]
                for client_metrics_dict in clients_metrics_dicts:
                    client_id = next(iter(client_metrics_dict))
                    if client_id not in train_losses_by_client:
                        continue
                    client_metrics = client_metrics_dict[client_id]
                    if "loss" in client_metrics:
                        loss_value = client_metrics["loss"]
                        train_losses_by_client[client_id].append(loss_value)
                        all_train_losses.append(loss_value)
            # Testing losses.
            if "test" in round_metrics and "clients_metrics_dicts" in round_metrics["test"]:
                clients_metrics_dicts = round_metrics["test"]["clients_metrics_dicts"]
                for client_metrics_dict in clients_metrics_dicts:
                    client_id = next(iter(client_metrics_dict))
                    if client_id not in test_losses_by_client:
                        continue
                    client_metrics = client_metrics_dict[client_id]
                    if "loss" in client_metrics:
                        loss_value = client_metrics["loss"]
                        test_losses_by_client[client_id].append(loss_value)
                        all_test_losses.append(loss_value)
        # Fallback to profiling losses when no history exists for a client.
        for client_id in candidate_client_ids:
            if not train_losses_by_client[client_id]:
                train_profile_loss = self._get_profile_loss_for_phase(client_id,
                                                                      candidate_clients,
                                                                      clients_profiles,
                                                                      "train")
                if train_profile_loss is not None:
                    train_losses_by_client[client_id].append(train_profile_loss)
                    all_train_losses.append(train_profile_loss)
            if not test_losses_by_client[client_id]:
                test_profile_loss = self._get_profile_loss_for_phase(client_id,
                                                                     candidate_clients,
                                                                     clients_profiles,
                                                                     "test")
                if test_profile_loss is not None:
                    test_losses_by_client[client_id].append(test_profile_loss)
                    all_test_losses.append(test_profile_loss)
        return train_losses_by_client, test_losses_by_client, all_train_losses, all_test_losses

    def _build_utility_inputs(self,
                              candidate_clients: dict,
                              selected_clients_metrics_history: dict,
                              clients_profiles: dict,
                              q: int) -> tuple:
        candidate_client_ids = list(candidate_clients.keys())
        train_losses_by_client, test_losses_by_client, all_train_losses, all_test_losses = \
            self._collect_client_losses_from_history(selected_clients_metrics_history, candidate_clients, clients_profiles)
        # Safe bounds from observed historical values.
        min_train_loss = min(all_train_losses) if all_train_losses else 0.0
        max_train_loss = max(all_train_losses) if all_train_losses else 1.0
        min_test_loss = min(all_test_losses) if all_test_losses else 0.0
        max_test_loss = max(all_test_losses) if all_test_losses else 1.0
        phi_list = []
        psi_list = []
        for client_id in candidate_client_ids:
            client_train_losses = train_losses_by_client.get(client_id, [])
            client_test_losses = test_losses_by_client.get(client_id, [])
            # Use the last q training participation.
            recent_train_losses = client_train_losses[-q:] if q > 0 else client_train_losses
            normalized_recent_train_losses = [self._normalize_value(loss_value, min_train_loss, max_train_loss)
                                              for loss_value in recent_train_losses]
            normalized_latest_test_loss = 1.0
            if client_test_losses:
                latest_test_loss = client_test_losses[-1]
                normalized_latest_test_loss = self._normalize_value(latest_test_loss, min_test_loss, max_test_loss)
            # Assumption: every eligible client has at least one profiling round with train/test loss (defensive fallbacks).
            if normalized_recent_train_losses:
                avg_normalized_train_loss = sum(normalized_recent_train_losses) / len(normalized_recent_train_losses)
                phi_i = 1 - avg_normalized_train_loss
            else:
                phi_i = 0.0
            psi_i = 1 - normalized_latest_test_loss
            # Clamp to [0, 1].
            phi_i = max(0.0, min(1.0, phi_i))
            psi_i = max(0.0, min(1.0, psi_i))
            phi_list.append(phi_i)
            psi_list.append(psi_i)
        return phi_list, psi_list

    @staticmethod
    def _build_fixed_normalization_bounds(current_phase: str,
                                          selected_clients_metrics_history: dict,
                                          candidate_clients: dict,
                                          clients_profiles: dict) -> dict:
        historical_makespans = []
        historical_energy_consumptions = []
        # Collect prior observed round-level metrics for the current phase.
        for round_idx in sorted(selected_clients_metrics_history.keys()):
            round_metrics = selected_clients_metrics_history[round_idx]
            if current_phase not in round_metrics:
                continue
            if "clients_metrics_dicts" not in round_metrics[current_phase]:
                continue
            clients_metrics_dicts = round_metrics[current_phase]["clients_metrics_dicts"]
            makespan_r = 0.0
            energy_r = 0.0
            for client_metrics_dict in clients_metrics_dicts:
                client_id = next(iter(client_metrics_dict))
                client_metrics = client_metrics_dict[client_id]
                time_key = "{0}ing_time_in_seconds".format(current_phase)
                energy_key = "{0}ing_energy_in_joules".format(current_phase)
                time_i = client_metrics[time_key] if time_key in client_metrics else 0.0
                energy_i = client_metrics[energy_key] if energy_key in client_metrics else 0.0
                if time_i > makespan_r:
                    makespan_r = time_i
                energy_r += energy_i
            if makespan_r > 0:
                historical_makespans.append(makespan_r)
            if energy_r > 0:
                historical_energy_consumptions.append(energy_r)
        # Fallback to profiling-derived approximations if history is sparse.
        if not historical_makespans or not historical_energy_consumptions:
            profiled_times = []
            profiled_energies = []
            for client_id, client_map in candidate_clients.items():
                if client_id not in clients_profiles:
                    continue
                if current_phase not in clients_profiles[client_id]:
                    continue
                profile_phase_dict = clients_profiles[client_id][current_phase]
                if not profile_phase_dict:
                    continue
                task_assignment_capacities_i = client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                max_ac = max(task_assignment_capacities_i)
                closest_x = min(profile_phase_dict.keys(), key=lambda x: abs(int(x) - max_ac))
                profiled_metrics = profile_phase_dict[closest_x]
                time_key = "{0}ing_time_in_seconds".format(current_phase)
                energy_key = "{0}ing_energy_in_joules".format(current_phase)
                if time_key in profiled_metrics:
                    profiled_times.append(profiled_metrics[time_key])
                if energy_key in profiled_metrics:
                    profiled_energies.append(profiled_metrics[energy_key])
            if profiled_times:
                # Conservative bounds based on single-client profile observations.
                historical_makespans.extend(profiled_times)
            if profiled_energies:
                historical_energy_consumptions.extend(profiled_energies)
        # Final safe defaults.
        if historical_makespans:
            min_M_X = min(historical_makespans)
            max_M_X = max(historical_makespans)
        else:
            min_M_X = 0.0
            max_M_X = 1.0
        if historical_energy_consumptions:
            min_E_X = min(historical_energy_consumptions)
            max_E_X = max(historical_energy_consumptions)
        else:
            min_E_X = 0.0
            max_E_X = 1.0
        # Avoid degenerate equal bounds.
        if max_M_X <= min_M_X:
            max_M_X = min_M_X + 1.0
        if max_E_X <= min_E_X:
            max_E_X = min_E_X + 1.0
        normalization_bounds = {"min_M_X": min_M_X,
                                "max_M_X": max_M_X,
                                "min_E_X": min_E_X,
                                "max_E_X": max_E_X}
        return normalization_bounds

    def _new_client_selection_needed(self,
                                     current_round: int,
                                     current_phase: str,
                                     candidate_clients: dict,
                                     selected_clients_metrics_history: dict,
                                     logger: Logger) -> bool:
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        new_client_selection_criteria_phase = client_selection_settings["new_client_selection_criteria_{0}ing".format(current_phase)]
        # Initialize the lists of new client selection criteria and reasons.
        new_client_selection_criteria = []
        new_client_selection_reasons = []
        # (i) The current round immediately follows the initial profiling?
        immediately_follows_profiling = (current_round == 1)
        if immediately_follows_profiling:
            new_client_selection_criteria.append(True)
            new_client_selection_reasons.append("The current round immediately follows the initial profiling.")
        else:
            # (ii) The set of candidate clients differs from the previous round?
            internal_candidate_clients_history = self.get_attribute("_internal_candidate_clients_history")
            previous_round_candidate_clients = internal_candidate_clients_history[current_round - 1]
            different_set_of_clients = set(candidate_clients.keys()) != set(previous_round_candidate_clients[current_phase].keys())
            if different_set_of_clients:
                new_client_selection_criteria.append(True)
                new_client_selection_reasons.append("The set of candidate clients differs from the previous round.")
            # (iii) Any user-defined criterion for a new client selection is fulfilled?
            for crit_key, crit_conf in new_client_selection_criteria_phase.items():
                crit_name = crit_conf["name"]
                match crit_name:
                    case "client_diversity_score_training":
                        # (iii-a) Client diversity checks during the training phase:
                        # 1. Does the accumulative client diversity score over the past X rounds fall below the minimum threshold Y?
                        # 2. Has the client diversity score decreased by more than the maximum relative drop over the past X rounds?
                        phase_of_interest = "train"
                        minimum_score = crit_conf.get("minimum_score", None)
                        maximum_relative_drop = crit_conf.get("maximum_relative_drop", None)
                        num_past_rounds = crit_conf["num_past_rounds"]
                        internal_candidate_clients_history = self.get_attribute("_internal_candidate_clients_history")
                        internal_selected_clients_history = self.get_attribute("_internal_selected_clients_history")
                        candidate_clients_history_ids = {}
                        selected_clients_history_ids = {}
                        for round_key, round_candidate_clients in internal_candidate_clients_history.items():
                            if phase_of_interest in round_candidate_clients:
                                candidate_clients_history_ids.update({round_key: list(round_candidate_clients[phase_of_interest].keys())})
                        for round_key, round_selected_clients in internal_selected_clients_history.items():
                            if phase_of_interest in round_selected_clients:
                                selected_clients_history_ids.update({round_key: list(round_selected_clients[phase_of_interest].keys())})
                        if minimum_score:
                            accumulative_client_diversity_score \
                               = calculate_normalized_client_diversity_score_over_past_x_rounds(candidate_clients_history_ids,
                                                                                                selected_clients_history_ids,
                                                                                                num_past_rounds)
                            if accumulative_client_diversity_score < minimum_score:
                               new_client_selection_criteria.append(True)
                               new_client_selection_reasons.append("The accumulative client diversity score over the past {0} rounds has fallen below {1} during the training phase ({2})."
                                                                   .format(num_past_rounds, minimum_score, round(accumulative_client_diversity_score, 2)))
                        elif maximum_relative_drop:
                            current_client_diversity_score = \
                                calculate_normalized_client_diversity_score_over_past_x_rounds(candidate_clients_history_ids,
                                                                                               selected_clients_history_ids,
                                                                                               num_past_rounds)
                            previous_client_diversity_score = \
                                calculate_normalized_client_diversity_score_over_past_x_rounds(candidate_clients_history_ids,
                                                                                               selected_clients_history_ids,
                                                                                               num_past_rounds + 1)
                            if previous_client_diversity_score > 0:
                                relative_change = ((current_client_diversity_score - previous_client_diversity_score) / previous_client_diversity_score) * 100
                                if relative_change < -maximum_relative_drop * 100:
                                    new_client_selection_criteria.append(True)
                                    new_client_selection_reasons.append("The client diversity score decreased by more than {0}% over the past {1} rounds ({2}{3}%)."
                                                                        .format(round(maximum_relative_drop * 100, 2),
                                                                                num_past_rounds,
                                                                                "+" if relative_change > 0 else "",
                                                                                round(relative_change, 2)))
                    case "makespan_percentage_increase_training":
                        # (iii-b) The makespan of the training phase has increased by more than X% in the past Y rounds?
                        phase_of_interest = "train"
                        maximum_increase = crit_conf["maximum_increase"]
                        if 0 <= maximum_increase <= 1:
                            maximum_increase *= 100
                        num_past_rounds = crit_conf["num_past_rounds"]
                        metrics_past_rounds = self._get_metrics_of_past_x_rounds(selected_clients_metrics_history,
                                                                                 current_round,
                                                                                 phase_of_interest,
                                                                                 num_past_rounds)
                        makespans = metrics_past_rounds["makespans"]
                        for idx in range(0, len(makespans) - 1):
                            makespan_prev_idx = makespans[idx]
                            makespan_next_idx = makespans[idx + 1]
                            makespan_percentage_change = calculate_percentage_change(makespan_prev_idx, makespan_next_idx)
                            if makespan_percentage_change > maximum_increase:
                                new_client_selection_criteria.append(True)
                                new_client_selection_reasons.append("The makespan of the training phase has increased by more than {0}% in the past {1} rounds ({2}{3}%)."
                                                                    .format(round(maximum_increase, 2), num_past_rounds, "+" if makespan_percentage_change > 0 else "", round(makespan_percentage_change, 2)))
                                break
                    case "energy_consumption_percentage_increase_training":
                        # (iii-c) The energy consumption of the training phase has increased by more than X% in the past Y rounds?
                        phase_of_interest = "train"
                        maximum_increase = crit_conf["maximum_increase"]
                        if 0 <= maximum_increase <= 1:
                            maximum_increase *= 100
                        num_past_rounds = crit_conf["num_past_rounds"]
                        metrics_past_rounds = self._get_metrics_of_past_x_rounds(selected_clients_metrics_history,
                                                                                 current_round,
                                                                                 phase_of_interest,
                                                                                 num_past_rounds)
                        energy_consumptions = metrics_past_rounds["energy_consumptions"]
                        for idx in range(0, len(energy_consumptions) - 1):
                            energy_consumption_prev_idx = energy_consumptions[idx]
                            energy_consumption_next_idx = energy_consumptions[idx + 1]
                            energy_consumption_percentage_change = calculate_percentage_change(energy_consumption_prev_idx,
                                                                                               energy_consumption_next_idx)
                            if energy_consumption_percentage_change > maximum_increase:
                                new_client_selection_criteria.append(True)
                                new_client_selection_reasons.append("The energy consumption of the training phase has increased by more than {0}% in the past {1} rounds ({2}{3}%)."
                                                                    .format(round(maximum_increase, 2), num_past_rounds, "+" if energy_consumption_percentage_change > 0 else "", round(energy_consumption_percentage_change, 2)))
                                break
                    case "accuracy_percentage_decrease_testing":
                        # (iii-d) The model accuracy of the testing phase has decreased by more than X% in the past Y rounds?
                        phase_of_interest = "test"
                        maximum_decrease = crit_conf["maximum_decrease"]
                        if 0 <= maximum_decrease <= 1:
                            maximum_decrease *= 100
                        num_past_rounds = crit_conf["num_past_rounds"]
                        metrics_past_rounds = self._get_metrics_of_past_x_rounds(selected_clients_metrics_history,
                                                                                 current_round,
                                                                                 phase_of_interest,
                                                                                 num_past_rounds)
                        weighted_mean_accuracies = metrics_past_rounds["weighted_mean_accuracies"]
                        for idx in range(0, len(weighted_mean_accuracies) - 1):
                            weighted_mean_accuracy_prev_idx = weighted_mean_accuracies[idx]
                            weighted_mean_accuracy_next_idx = weighted_mean_accuracies[idx + 1]
                            weighted_mean_accuracy_percentage_change = calculate_percentage_change(weighted_mean_accuracy_prev_idx,
                                                                                                   weighted_mean_accuracy_next_idx)
                            if weighted_mean_accuracy_percentage_change < - maximum_decrease:
                                new_client_selection_criteria.append(True)
                                new_client_selection_reasons.append("The model accuracy of the testing phase has decreased by more than {0}% in the past {1} rounds ({2}{3}%)."
                                                                    .format(round(maximum_decrease, 2), num_past_rounds, "+" if weighted_mean_accuracy_percentage_change > 0 else "", round(weighted_mean_accuracy_percentage_change, 2)))
                                break
        # Check if any criterion for a new client selection is fulfilled.
        if any(new_client_selection_criteria):
            # Log a 'new client selection is needed' message.
            message = "[MetaCS-FL | Round {0}] A new client selection is needed! Reason(s):".format(current_round)
            message += "\n" + "\n".join("- {0}".format(reason) for reason in new_client_selection_reasons)
            log_message(logger, message, "INFO")
            # A new client selection is needed.
            return True
        # A new client selection is not needed.
        return False

    @staticmethod
    def _generate_cost_matrices(current_phase: str,
                                candidate_clients: dict,
                                selected_clients_metrics_history: dict,
                                clients_profiles: dict) -> dict:
        time_costs = []
        energy_costs = []
        for client_id, client_map in candidate_clients.items():
            task_assignment_capacities_i = client_map["client_task_assignment_capacities_{0}".format(current_phase)]
            current_download_bandwidth_in_bytes_per_second_i = client_map["client_current_download_bandwidth_in_bytes_per_second"]
            current_upload_bandwidth_in_bytes_per_second_i = client_map["client_current_upload_bandwidth_in_bytes_per_second"]
            current_latency_in_milliseconds_i = client_map["client_current_latency_in_milliseconds"]
            time_costs_i = []
            energy_costs_i = []
            latest_phase_metrics_i = {}
            for round_key in reversed(selected_clients_metrics_history):
                if current_phase in selected_clients_metrics_history[round_key] and \
                    "clients_metrics_dicts" in selected_clients_metrics_history[round_key][current_phase]:
                    clients_metrics = selected_clients_metrics_history[round_key][current_phase]["clients_metrics_dicts"]
                    for client_dict in clients_metrics:
                        if client_id in client_dict:
                            latest_phase_metrics_i = client_dict[client_id]
                            break
            if not latest_phase_metrics_i:
                profile_phase_dict = clients_profiles[client_id][current_phase]
                # profile_phase_dict: {num_samples: metrics}
                # Pick the closest profiled sample size to the max capacity.
                max_ac = max(task_assignment_capacities_i)
                closest_x = min(profile_phase_dict.keys(), key=lambda x: abs(int(x) - max_ac))
                latest_phase_metrics_i = profile_phase_dict[closest_x]
            latest_phase_metrics_i_copy = deepcopy(latest_phase_metrics_i)
            latest_phase_metrics_i_copy["bw_down_i"] = current_download_bandwidth_in_bytes_per_second_i
            latest_phase_metrics_i_copy["bw_up_i"] = current_upload_bandwidth_in_bytes_per_second_i
            latest_phase_metrics_i_copy["rtt_down_i"] = current_latency_in_milliseconds_i
            latest_phase_metrics_i_copy["rtt_up_i"] = current_latency_in_milliseconds_i
            download_time_i, _ = calculate_download_time(latest_phase_metrics_i_copy, current_phase)
            download_energy_i = calculate_download_energy(latest_phase_metrics_i_copy, download_time_i)
            upload_time_i, _ = calculate_upload_time(latest_phase_metrics_i_copy, current_phase)
            upload_energy_i = calculate_upload_energy(latest_phase_metrics_i_copy, upload_time_i)
            for ac_i in task_assignment_capacities_i:
                latest_phase_metrics_i_copy["ds_{0}_i".format(current_phase)] = ac_i
                computation_time_ac_i = calculate_computation_time(latest_phase_metrics_i_copy, current_phase)
                computation_energy_ac_i = calculate_computation_energy(latest_phase_metrics_i_copy, computation_time_ac_i)
                time_cost_ac_i = download_time_i + computation_time_ac_i + upload_time_i
                energy_cost_ac_i = download_energy_i + computation_energy_ac_i + upload_energy_i
                time_costs_i.append(time_cost_ac_i)
                energy_costs_i.append(energy_cost_ac_i)
            time_costs.append(time_costs_i)
            energy_costs.append(energy_costs_i)
        cost_matrices = {"time_costs": time_costs,
                         "energy_costs": energy_costs}
        return cost_matrices

    def _initial_solution_generation_needed(self,
                                            current_round: int,
                                            current_phase: str,
                                            candidate_clients: dict,
                                            logger: Logger) -> bool:
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        new_initial_solution_criteria_phase = client_selection_settings["new_initial_solution_criteria_{0}ing".format(current_phase)]
        # Initialize the lists of initial solution generation criteria and reasons.
        generate_initial_solution_criteria = []
        generate_initial_solution_reasons = []
        # (i) The current round immediately follows the initial profiling?
        immediately_follows_initial_profiling = (current_round == 1)
        if immediately_follows_initial_profiling:
            generate_initial_solution_criteria.append(True)
            generate_initial_solution_reasons.append("The current round immediately follows the initial profiling.")
        else:
            # (ii) The set of candidate clients differs from the previous round?
            internal_candidate_clients_history = self.get_attribute("_internal_candidate_clients_history")
            previous_round_candidate_clients = internal_candidate_clients_history[current_round - 1]
            different_set_of_clients = set(candidate_clients.keys()) != set(previous_round_candidate_clients[current_phase].keys())
            if different_set_of_clients:
                generate_initial_solution_criteria.append(True)
                generate_initial_solution_reasons.append("The set of candidate clients differs from the previous round.")
            # (iii) Any user-defined criterion for a new initial solution generation is fulfilled?
            for crit_key, crit_conf in new_initial_solution_criteria_phase.items():
                crit_name = crit_conf["name"]
                match crit_name:
                    case "max_rounds_same_initial_solution":
                        # (iii-a) A pre-defined number of rounds has passed since the last generation?
                        maximum_rounds = crit_conf["maximum_rounds"]
                        initial_solution_generation_history = self.get_attribute("_initial_solution_generation_history")
                        last_initial_solution_generation = next(reversed(initial_solution_generation_history.items()), None)
                        if last_initial_solution_generation is not None:
                            last_generation_round, _ = last_initial_solution_generation
                            last_generation_rounds_difference = current_round - last_generation_round
                            if last_generation_rounds_difference >= maximum_rounds:
                                generate_initial_solution_criteria.append(True)
                                generate_initial_solution_reasons.append("{0} rounds have passed since the last generation of the initial solution (maximum allowed is {1})."
                                                                         .format(last_generation_rounds_difference, maximum_rounds))
        # Check if any criterion for an initial solution generation is fulfilled.
        if any(generate_initial_solution_criteria):
            # Log an 'initial solution generation is needed' message.
            message = "[MetaCS-FL | Round {0}] An initial solution generation is needed! Reason(s):".format(current_round)
            message += "\n" + "\n".join("- {0}".format(reason) for reason in generate_initial_solution_reasons)
            log_message(logger, message, "INFO")
            # An initial solution generation is needed.
            return True
        # An initial solution generation is not needed.
        return False

    def _generate_initial_solution(self,
                                   current_round: int,
                                   current_phase: str,
                                   candidate_clients: dict,
                                   num_tasks: int,
                                   samples_per_task: int,
                                   cost_matrices: dict,
                                   time_limit: float,
                                   data_privacy_approach: str,
                                   logger: Logger) -> tuple:
        # Initialize the initial solution.
        X_init = []
        # Build the class capacity vectors list.
        class_capacity_vectors_list, sorted_classes = build_class_capacity_vectors_list(candidate_clients,
                                                                                        data_privacy_approach,
                                                                                        current_phase)
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        initial_solution_generator_phase = client_selection_settings["initial_solution_generator_{0}ing".format(current_phase)]
        initial_solution_generator_name = initial_solution_generator_phase["name"]
        initial_solution_tasks_distribution_scheme = client_selection_settings["initial_solution_tasks_distribution_scheme"]
        # Log a 'generating the initial solution' message.
        message = "[MetaCS-FL | Round {0}] Generating the initial solution using '{1}'..." \
                  .format(current_round, initial_solution_generator_name)
        log_message(logger, message, "INFO")
        match initial_solution_generator_name:
            case "Random":
                # Select clients using the 'Random' algorithm.
                fraction_clients = initial_solution_generator_phase["fraction_clients_{0}ing".format(current_phase)]
                X_init = random_selection(current_phase,
                                          num_tasks,
                                          candidate_clients,
                                          fraction_clients)
            case "MEC":
                # Select clients using the 'MEC' algorithm.
                task_assignment_capacities_list = self.get_attribute("_task_assignment_capacities_list")
                time_costs = cost_matrices["time_costs"]
                energy_costs = cost_matrices["energy_costs"]
                X_init, _, _ = mec(len(candidate_clients),
                                   num_tasks,
                                   task_assignment_capacities_list,
                                   time_costs,
                                   energy_costs)
            case "ECMTC":
                # Select clients using the 'ECMTC' algorithm.
                task_assignment_capacities_list = self.get_attribute("_task_assignment_capacities_list")
                time_costs = cost_matrices["time_costs"]
                energy_costs = cost_matrices["energy_costs"]
                X_init, _, _ = ecmtc(len(candidate_clients),
                                     num_tasks,
                                     task_assignment_capacities_list,
                                     time_costs,
                                     energy_costs,
                                     time_limit)
        # Scale the assigned tasks back to sample-level counts (since distribution is based on samples per class).
        X_init_scaled = [x_i * samples_per_task for x_i in X_init]
        # Distribute the type of tasks scheduled per client.
        X_init_dist = []
        match initial_solution_tasks_distribution_scheme:
            case "random":
                X_init_dist = distribute_tasks_with_random_approach(X_init_scaled, class_capacity_vectors_list)
            case "locally_balanced":
                X_init_dist = distribute_tasks_with_locally_balanced_approach(X_init_scaled, class_capacity_vectors_list)
            case "globally_balanced":
                X_init_dist = distribute_tasks_with_globally_balanced_approach(X_init_scaled, class_capacity_vectors_list)
        # Organize the tasks' distribution.
        X_init_dist = organize_tasks_distribution(X_init_dist, sorted_classes)
        # Log a 'initial solution generated' message.
        message = "[MetaCS-FL | Round {0}] Initial solution generated: {1}" \
                  .format(current_round, X_init)
        log_message(logger, message, "INFO")
        # Return the initial solution generated.
        return X_init, X_init_dist

    def _get_or_generate_initial_solution(self,
                                          current_round: int,
                                          current_phase: str,
                                          candidate_clients: dict,
                                          num_tasks: int,
                                          samples_per_task: int,
                                          cost_matrices: dict,
                                          time_limit: float,
                                          data_privacy_approach: str,
                                          logger: Logger) -> tuple:
        # Verify if an initial solution generation is necessary.
        initial_solution_generation_is_needed = self._initial_solution_generation_needed(current_round,
                                                                                         current_phase,
                                                                                         candidate_clients,
                                                                                         logger)
        if initial_solution_generation_is_needed:
            # Generate an initial solution.
            X_init, X_init_dist = self._generate_initial_solution(current_round,
                                                                  current_phase,
                                                                  candidate_clients,
                                                                  num_tasks,
                                                                  samples_per_task,
                                                                  cost_matrices,
                                                                  time_limit,
                                                                  data_privacy_approach,
                                                                  logger)
            # Update the initial solution generation history.
            initial_solution_generation_history = self.get_attribute("_initial_solution_generation_history")
            if current_round not in initial_solution_generation_history:
                initial_solution_generation_history.update({current_round: {current_phase: {"X_init": X_init, "X_init_dist": X_init_dist}}})
            else:
                initial_solution_generation_history[current_round].update({current_phase: {"X_init": X_init, "X_init_dist": X_init_dist}})
            self._set_attribute("_initial_solution_generation_history", initial_solution_generation_history)
        else:
            # Get the latest initial solution generated.
            initial_solution_generation_history = self.get_attribute("_initial_solution_generation_history")
            phase_rounds = [round_id
                            for round_id, phases in initial_solution_generation_history.items()
                            if current_phase in phases]
            if not phase_rounds:
                raise ValueError("No rounds found for phase '{0}'".format(current_phase))
            last_generation_round = max(phase_rounds)
            X_init = initial_solution_generation_history[last_generation_round][current_phase]["X_init"]
            X_init_dist = initial_solution_generation_history[last_generation_round][current_phase]["X_init_dist"]
            # Log a 'using the latest initial solution generated' message.
            message = "[MetaCS-FL | Round {0}] Using the latest initial solution generated (on round {1})!" \
                .format(current_round, last_generation_round)
            log_message(logger, message, "INFO")
        return X_init, X_init_dist

    def _select_clients(self,
                        current_round: int,
                        current_phase: str,
                        candidate_clients: dict,
                        num_tasks: int,
                        samples_per_task: int,
                        base_learning_rate: float,
                        base_batch_size: int,
                        base_num_epochs: int,
                        selected_clients_metrics_history: dict,
                        clients_profiles: dict,
                        time_limit: float,
                        data_privacy_approach: str,
                        root_output_folder: Path,
                        logger: Logger) -> dict:
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        rng = self.get_attribute("_rng")
        # Generate the cost matrices.
        cost_matrices = self._generate_cost_matrices(current_phase,
                                                     candidate_clients,
                                                     selected_clients_metrics_history,
                                                     clients_profiles)
        # Get or generated the initial solution.
        X_init, X_init_dist = self._get_or_generate_initial_solution(current_round,
                                                                     current_phase,
                                                                     candidate_clients,
                                                                     num_tasks,
                                                                     samples_per_task,
                                                                     cost_matrices,
                                                                     time_limit,
                                                                     data_privacy_approach,
                                                                     logger)
        # Get the necessary properties of the candidate clients.
        task_assignment_capacities_list = self.get_attribute("_task_assignment_capacities_list")
        # Build the class capacity vectors list.
        class_capacity_vectors_list, sorted_classes = build_class_capacity_vectors_list(candidate_clients,
                                                                                        data_privacy_approach,
                                                                                        current_phase)
        mean_power_consumption_idle_mode_list = [client_map["client_mean_power_consumption_idle_mode"]
                                                 for _, client_map in candidate_clients.items()]
        remaining_battery_energy_list = [client_map["client_remaining_battery_energy"]
                                         for _, client_map in candidate_clients.items()]
        time_costs = cost_matrices["time_costs"]
        energy_costs = cost_matrices["energy_costs"]
        # Initialize the metaheuristic execution statistics.
        mh_statistics = {"time_lapsed": 0, "num_iterations": 0}
        # Get the metaheuristic settings.
        metaheuristic = client_selection_settings["metaheuristic"]
        metaheuristic_name = metaheuristic["name"]
        metaheuristic_stopping_criteria = metaheuristic["stopping_criteria"]
        # TODO: Get the utility score settings.
        utility_score_settings = {"alpha": 0.5, "q": 3}
        alpha = utility_score_settings["alpha"]
        q = utility_score_settings["q"]
        # Build the utility inputs for the current candidate clients.
        phi_list, psi_list = self._build_utility_inputs(candidate_clients,
                                                        selected_clients_metrics_history,
                                                        clients_profiles,
                                                        q)
        normalization_bounds = self._build_fixed_normalization_bounds(current_phase,
                                                                      selected_clients_metrics_history,
                                                                      candidate_clients,
                                                                      clients_profiles)
        # Initialize the best solution and its tasks' distribution.
        X_best = deepcopy(X_init)
        X_best_dist = deepcopy(X_init_dist)
        # If the metaheuristic execution for the current phase is enabled...
        run_metaheuristic_phase = client_selection_settings["run_metaheuristic_{0}ing".format(current_phase)]
        if run_metaheuristic_phase:
            # Initialize the costs of the best solution.
            X_best_costs = {}
            # Log a 'running the metaheuristic' message.
            message = "[MetaCS-FL | Round {0}] Running the '{1}' metaheuristic..." \
                      .format(current_round, metaheuristic_name)
            log_message(logger, message, "INFO")
            match metaheuristic_name:
                case "LNS":
                    # Get the LNS settings.
                    destroy_approach = metaheuristic["destroy_approach"]
                    accept_criteria = metaheuristic["accept_criteria"]
                    obj_func_weights = metaheuristic["obj_func_weights"]
                    # Set the LNS traces output file.
                    lns_traces_output_file = None
                    write_traces_to_output_file = metaheuristic.get("write_traces_to_output_file", False)
                    if write_traces_to_output_file:
                        lns_traces_output_file_name = "lns_traces/round_{0}_{1}.csv".format(current_round, current_phase)
                        lns_traces_output_file = Path(root_output_folder).joinpath(lns_traces_output_file_name)
                    # Set the schedules' task distribution approaches.
                    X_dist_approaches = {"X_init": client_selection_settings["initial_solution_tasks_distribution_scheme"],
                                         "X_rpr": client_selection_settings["metaheuristic_solution_tasks_distribution_scheme"],
                                         "X_best": client_selection_settings["metaheuristic_solution_tasks_distribution_scheme"]}
                    # Run the LNS metaheuristic.
                    X_best, X_best_costs, mh_statistics = run_lns(X_init,
                                                                  rng,
                                                                  destroy_approach,
                                                                  len(candidate_clients),
                                                                  time_limit,
                                                                  num_tasks,
                                                                  task_assignment_capacities_list,
                                                                  class_capacity_vectors_list,
                                                                  mean_power_consumption_idle_mode_list,
                                                                  remaining_battery_energy_list,
                                                                  time_costs,
                                                                  energy_costs,
                                                                  phi_list,
                                                                  psi_list,
                                                                  alpha,
                                                                  metaheuristic_stopping_criteria,
                                                                  accept_criteria,
                                                                  obj_func_weights,
                                                                  X_dist_approaches,
                                                                  normalization_bounds,
                                                                  lns_traces_output_file)
            # Log a 'metaheuristic execution time and iterations' message.
            message = "[MetaCS-FL | Round {0}] The '{1}' metaheuristic execution took {2} seconds " \
                      "(number of iterations: {3})." \
                      .format(current_round,
                              metaheuristic_name,
                              round(mh_statistics["time_lapsed"], 2),
                              mh_statistics["num_iterations"])
            log_message(logger, message, "INFO")
            # Log a 'best solution' message.
            message = "[MetaCS-FL | Round {0}] Best solution found: {1}" \
                      .format(current_round, X_best)
            log_message(logger, message, "INFO")
            # Organize the tasks' distribution.
            if "X_dist" in X_best_costs:
                # The best solution already has its own distribution (in tasks);
                # Scale from tasks → samples.
                X_best_dist_scaled = [[x_ij * samples_per_task for x_ij in x_i] for x_i in X_best_costs["X_dist"]]
                X_best_dist = organize_tasks_distribution(X_best_dist_scaled, sorted_classes)
            else:
                # Otherwise, adjust the distribution of the initial solution.
                metaheuristic_solution_tasks_distribution_scheme = \
                    client_selection_settings["metaheuristic_solution_tasks_distribution_scheme"]
                # Scale from tasks → samples before distributing.
                X_init_scaled = [x_i * samples_per_task for x_i in X_init]
                match metaheuristic_solution_tasks_distribution_scheme:
                    case "random":
                        X_best_dist = distribute_tasks_with_random_approach(X_init_scaled, class_capacity_vectors_list)
                    case "locally_balanced":
                        X_best_dist = distribute_tasks_with_locally_balanced_approach(X_init_scaled, class_capacity_vectors_list)
                    case "globally_balanced":
                        X_best_dist = distribute_tasks_with_globally_balanced_approach(X_init_scaled, class_capacity_vectors_list)
                X_best_dist = organize_tasks_distribution(X_best_dist, sorted_classes)
        # Set the lists of base training / testing instructions.
        learning_rate_list = []
        batch_size_list = []
        num_epochs_list = []
        if current_phase == "train":
            learning_rate_list = [base_learning_rate for i, _ in enumerate(X_best)]
            batch_size_list = [base_batch_size for i, _ in enumerate(X_best)]
            num_epochs_list = [base_num_epochs for i, _ in enumerate(X_best)]
        elif current_phase == "test":
            batch_size_list = [base_batch_size for i, _ in enumerate(X_best)]
        if current_phase == "train":
            # Dynamically adjust the learning rate per client to improve training stability and fairness:
            # - Clients with fewer samples receive a higher learning rate to accelerate learning from limited data;
            # - Clients with more samples receive a lower learning rate to ensure more stable updates over more iterations.
            dynamically_adjust_learning_rate_training = client_selection_settings.get("dynamically_adjust_learning_rate_training", False)
            if dynamically_adjust_learning_rate_training:
                learning_rate_list = adjust_learning_rates(X_best, base_learning_rate)
            # Dynamically adjust the batch size per client to optimize training efficiency and convergence:
            # - Clients with fewer samples use smaller batch sizes to increase gradient update frequency and improve generalization;
            # - Clients with more samples use larger batch sizes to better utilize data and improve computational efficiency.
            dynamically_adjust_batch_size_training = client_selection_settings.get("dynamically_adjust_batch_size_training", False)
            if dynamically_adjust_batch_size_training:
                batch_size_list = adjust_batch_sizes(X_best, base_batch_size)
            # Dynamically adjust the number of local epochs per client to balance contribution:
            # - Clients with fewer samples train longer to increase their influence;
            # - Clients with more samples train for fewer epochs to prevent over-dominating the global model.
            dynamically_adjust_num_epochs_training = client_selection_settings.get("dynamically_adjust_num_epochs_training", False)
            if dynamically_adjust_num_epochs_training:
                num_epochs_list = adjust_num_epochs(X_best, base_num_epochs)
        # Initialize the set of selected clients.
        selected_clients = {}
        for i, _ in enumerate(X_best):
            # Get the number of tasks assigned to client i (in tasks/minibatches).
            x_i = int(X_best[i])
            # Get the tasks-per-class distribution (already in samples).
            x_dist_i = X_best_dist[i]
            # Compute the total number of samples (tasks × samples_per_task).
            num_samples_i = x_i * samples_per_task
            x_dist_i_str = "|".join(["{0}={1}".format(k, v) for k, v in x_dist_i.items()])
            if x_i > 0:
                # Update the set of selected clients.
                client_id_str = list(candidate_clients.keys())[i]
                client_proxy = candidate_clients[client_id_str]["client_proxy"]
                capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
                client_task_assignment_capacities_phase = candidate_clients[client_id_str][capacity_key]
                client_max_task_capacity = max(client_task_assignment_capacities_phase)
                client_info = {"client_proxy": client_proxy,
                               capacity_key: client_task_assignment_capacities_phase,
                               "client_max_task_capacity": client_max_task_capacity,
                               "client_num_tasks_scheduled": x_i,
                               "client_num_samples_scheduled": num_samples_i,
                               "client_num_samples_per_class_scheduled": x_dist_i_str}
                if current_phase == "train":
                    client_info.update({"client_learning_rate": learning_rate_list[i],
                                        "client_batch_size": batch_size_list[i],
                                        "client_epochs": num_epochs_list[i]})
                elif current_phase == "test":
                    client_info.update({"client_batch_size": batch_size_list[i]})
                selected_clients.update({client_id_str: client_info})
        # Return the set of selected clients.
        return selected_clients

    def run_client_selection_procedure(self,
                                       **kwargs) -> dict:
        # Get the necessary parameters.
        current_round = kwargs["current_round"]
        current_phase = kwargs["current_phase"]
        candidate_clients = kwargs["candidate_clients"]
        clients_reliability_score_history = kwargs.get("clients_reliability_score_history", {})
        apply_clients_reliability_filter = kwargs.get("apply_clients_reliability_filter", False)
        num_tasks = kwargs["num_tasks"]
        samples_per_task = kwargs["samples_per_task"]
        base_learning_rate = kwargs.get("base_learning_rate", 0.001)
        base_batch_size = kwargs.get("base_batch_size", 32)
        base_num_epochs = kwargs.get("base_num_epochs", 1)
        selected_clients_history = kwargs["selected_clients_history"]
        selected_clients_metrics_history = kwargs["selected_clients_metrics_history"]
        clients_profiles = kwargs["clients_profiles"]
        time_limit = kwargs["time_limit"]
        data_privacy_approach = kwargs["data_privacy_approach"]
        root_output_folder = kwargs["root_output_folder"]
        logger = kwargs["logger"]
        # Get the necessary properties of the candidate clients.
        task_assignment_capacities_list = [client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                                           for _, client_map in candidate_clients.items()]
        if apply_clients_reliability_filter:
            task_assignment_capacities_list = self._get_trimmed_task_assignment_capacities(task_assignment_capacities_list,
                                                                                           clients_reliability_score_history)
        self._set_attribute("_task_assignment_capacities_list", task_assignment_capacities_list)
        scaled_task_assignment_capacities_list = [sorted(set(capacities * samples_per_task))
                                                  for capacities in task_assignment_capacities_list]
        # Calculate the maximum number of tasks that can be scheduled.
        max_num_tasks = sum(max(capacities) for capacities in scaled_task_assignment_capacities_list)
        # Convert num_tasks → task count (fraction adjustment).
        if isinstance(num_tasks, float):
            # Always treat as fraction of total capacity.
            num_tasks = int(num_tasks * max_num_tasks)
        # Clamp to valid range.
        num_tasks = max(1, min(num_tasks, max_num_tasks))
        # Compute all the possible sums of task assignments, considering one assignment per client.
        all_possible_task_assignment_sums = get_all_possible_sums(scaled_task_assignment_capacities_list)
        # If the number of tasks to schedule is infeasible...
        if num_tasks not in all_possible_task_assignment_sums:
            # Set a new valid number of tasks to schedule.
            num_tasks = take_closest(all_possible_task_assignment_sums, num_tasks)
        # Verify if a new client selection is necessary.
        new_client_selection_is_needed = self._new_client_selection_needed(current_round,
                                                                           current_phase,
                                                                           candidate_clients,
                                                                           selected_clients_metrics_history,
                                                                           logger)
        if new_client_selection_is_needed:
            # Select clients.
            selected_clients = self._select_clients(current_round,
                                                    current_phase,
                                                    candidate_clients,
                                                    num_tasks,
                                                    samples_per_task,
                                                    base_learning_rate,
                                                    base_batch_size,
                                                    base_num_epochs,
                                                    selected_clients_metrics_history,
                                                    clients_profiles,
                                                    time_limit,
                                                    data_privacy_approach,
                                                    root_output_folder,
                                                    logger)
        else:
            # Get the latest set of selected clients, if available.
            selected_clients = self._get_latest_selection(selected_clients_history, current_round, current_phase)
            # Filter the inactive clients from the latest selection, if any.
            selected_clients = {client_id: client_info
                                for client_id, client_info in selected_clients.items()
                                if client_id in candidate_clients}
            # Log a 'selected the same subset of clients from the previous round' message.
            message = "[MetaCS-FL | Round {0}] Selected the same subset of clients from the previous round!" \
                      .format(current_round)
            log_message(logger, message, "INFO")
        # Update the candidate clients' history (internal).
        internal_candidate_clients_history = self.get_attribute("_internal_candidate_clients_history")
        if current_round not in internal_candidate_clients_history:
            internal_candidate_clients_history.update({current_round: {current_phase: candidate_clients}})
        else:
            internal_candidate_clients_history[current_round].update({current_phase: candidate_clients})
        self._set_attribute("_internal_candidate_clients_history", internal_candidate_clients_history)
        # Update the selected clients' history (internal).
        internal_selected_clients_history = self.get_attribute("_internal_selected_clients_history")
        if current_round not in internal_selected_clients_history:
            internal_selected_clients_history.update({current_round: {current_phase: selected_clients}})
        else:
            internal_selected_clients_history[current_round].update({current_phase: selected_clients})
        self._set_attribute("_internal_selected_clients_history", internal_selected_clients_history)
        # Log a 'number of tasks → number of samples' per-client message.
        for client_id, client_info in selected_clients.items():
            message = "{0}: {1} tasks → {2} samples".format(client_id,
                                                            client_info['client_num_tasks_scheduled'],
                                                            client_info['client_num_samples_scheduled'])
            log_message(logger, message, "INFO")
        # Return the set of selected clients.
        return selected_clients
