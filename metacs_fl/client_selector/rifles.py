from collections import deque
from logging import Logger
from typing import Any

from numpy.random import default_rng, SeedSequence

from metacs_fl.utils.client_selector_util import get_all_possible_sums, take_closest
from metacs_fl.utils.logger_util import log_message
from metacs_fl.utils.task_scheduler_util import build_class_capacity_vectors_list, \
    distribute_tasks_with_globally_balanced_approach, distribute_tasks_with_locally_balanced_approach, \
    distribute_tasks_with_random_approach, organize_tasks_distribution


class RIFLES:
    """
    Workload-aware RIFLES adaptation for the MetaCS-FL pipeline.

    Behavior preserved:
      - tracks client availability history;
      - builds a predicted availability matrix;
      - builds a RIFLES-style eligibility matrix E_i(s);
      - selects a slot with enough eligible clients;
      - supports RIFLES-GH and RIFLES-LRU policies;
      - uses expected execution/response time in the eligibility condition;
      - uses LRU to avoid repeatedly selecting the same clients.

    Adaptation needed for MetaCS-FL:
      - original RIFLES uses heartbeat matrices and a CNN-LSTM predictor. Here,
        observed round-level availability history is converted into a lightweight
        pseudo-slot prediction because the current MetaCS-FL pipeline exposes
        candidate availability per FL round, not raw heartbeat traces.
      - original RIFLES schedules training slots over a predicted day. Here, the
        best pseudo-slot is selected for the current FL round, then the existing
        MetaCS-FL feasible workload-assignment mechanism assigns tasks.
    """

    def __init__(self,
                 client_selection_settings: dict,
                 seed: None | int | SeedSequence) -> None:
        self._client_selection_settings = client_selection_settings
        self._internal_candidate_clients_history = {}
        self._internal_selected_clients_history = {}
        self._task_assignment_capacities_list = []
        self._availability_history = {}
        self._last_seen_round = {}
        self._last_selected_round = {}
        self._selection_count = {}
        self._lru_cache = deque()
        self._known_clients_order = []
        self._rng = default_rng(seed=seed)

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: Any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> Any:
        return getattr(self, attribute_name)

    @staticmethod
    def _safe_float(value: Any,
                    default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _minimum_clients_required_to_cover_workload(candidate_clients: dict,
                                                    current_phase: str,
                                                    target_num_tasks: int) -> int:
        capacities = []
        for _, client_map in candidate_clients.items():
            capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
            capacities.append(max(client_map[capacity_key]))
        capacities = sorted(capacities, reverse=True)
        accumulated_capacity = 0
        for idx, capacity in enumerate(capacities):
            accumulated_capacity += capacity
            if accumulated_capacity >= target_num_tasks:
                return idx + 1
        return len(capacities)

    @staticmethod
    def _solve_exact_task_assignment_for_selected_clients(selected_indices: list,
                                                          client_aux_info: list,
                                                          target_num_tasks: int) -> list:
        if not selected_indices:
            return []
        selected_capacities = [sorted(set(client_aux_info[idx]["capacities"])) for idx in selected_indices]
        possible_sums = get_all_possible_sums(selected_capacities)
        if not possible_sums:
            return [0 for _ in selected_indices]
        if target_num_tasks not in possible_sums:
            target_num_tasks = take_closest(possible_sums, target_num_tasks)
        reachable = {0: []}
        for capacities in selected_capacities:
            next_reachable = {}
            for partial_sum, partial_solution in reachable.items():
                for capacity in capacities:
                    new_sum = partial_sum + int(capacity)
                    if new_sum > target_num_tasks:
                        continue
                    if new_sum not in next_reachable:
                        next_reachable[new_sum] = partial_solution + [int(capacity)]
            reachable = next_reachable
            if target_num_tasks in reachable:
                return reachable[target_num_tasks]
        if not reachable:
            return [min(capacities) for capacities in selected_capacities]
        best_sum = max(reachable.keys())
        return reachable[best_sum]

    def _update_availability_history(self,
                                     current_round: int,
                                     candidate_clients: dict) -> None:
        availability_history = self.get_attribute("_availability_history")
        last_seen_round = self.get_attribute("_last_seen_round")
        current_available_clients = set(candidate_clients.keys())
        known_clients = set(availability_history.keys()) | current_available_clients
        for client_id in known_clients:
            if client_id not in availability_history:
                availability_history[client_id] = []
            availability_history[client_id].append(1 if client_id in current_available_clients else 0)
            if client_id in current_available_clients:
                last_seen_round[client_id] = current_round
        self._set_attribute("_availability_history", availability_history)
        self._set_attribute("_last_seen_round", last_seen_round)

    def _availability_probability(self,
                                  client_id: str) -> float:
        settings = self.get_attribute("_client_selection_settings")
        availability_history = self.get_attribute("_availability_history")
        window = int(settings.get("availability_window", 10))
        history = availability_history.get(client_id, [])
        if not history:
            return 1.0
        recent_history = history[-window:]
        if not recent_history:
            return 1.0
        return sum(recent_history) / len(recent_history)

    def _extract_latest_metric_from_history(self,
                                            selected_clients_metrics_history: dict,
                                            client_id: str,
                                            current_phase: str,
                                            metric_substrings: list) -> float | None:
        phase_preference_order = [current_phase]
        if current_phase == "train":
            phase_preference_order.append("test")
        elif current_phase == "test":
            phase_preference_order.append("train")
        for phase in phase_preference_order:
            for round_key in reversed(sorted(selected_clients_metrics_history.keys())):
                round_metrics = selected_clients_metrics_history[round_key]
                if phase not in round_metrics:
                    continue
                clients_metrics_dicts = round_metrics[phase].get("clients_metrics_dicts", [])
                for client_metrics_dict in clients_metrics_dicts:
                    if client_id not in client_metrics_dict:
                        continue
                    client_metrics = client_metrics_dict[client_id]
                    for metric_key, metric_value in client_metrics.items():
                        metric_key_lower = str(metric_key).lower()
                        if any(substring in metric_key_lower for substring in metric_substrings):
                            metric_value = self._safe_float(metric_value, -1.0)
                            if metric_value > 0:
                                return metric_value
        return None

    def _extract_profile_metric(self,
                                clients_profiles: dict,
                                client_id: str,
                                current_phase: str,
                                task_capacity: int,
                                metric_substrings: list) -> float | None:
        if client_id not in clients_profiles:
            return None
        phase_preference_order = [current_phase]
        if current_phase == "train":
            phase_preference_order.append("test")
        elif current_phase == "test":
            phase_preference_order.append("train")
        for phase in phase_preference_order:
            if phase not in clients_profiles[client_id]:
                continue
            profile_phase_dict = clients_profiles[client_id][phase]
            if not profile_phase_dict:
                continue
            closest_x = min(profile_phase_dict.keys(), key=lambda x: abs(int(x) - int(task_capacity)))
            profiled_metrics = profile_phase_dict[closest_x]
            for metric_key, metric_value in profiled_metrics.items():
                metric_key_lower = str(metric_key).lower()
                if any(substring in metric_key_lower for substring in metric_substrings):
                    metric_value = self._safe_float(metric_value, -1.0)
                    if metric_value > 0:
                        return metric_value
        return None

    def _estimate_expected_time(self,
                                client_id: str,
                                client_map: dict,
                                clients_profiles: dict,
                                selected_clients_metrics_history: dict,
                                current_phase: str) -> float:
        capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
        max_capacity = int(max(client_map[capacity_key]))
        metric_substrings = ["elapsed_time", "duration", "execution_time", "processing_time",
                             "training_time", "testing_time", "time"]
        latest_metric = self._extract_latest_metric_from_history(selected_clients_metrics_history,
                                                                 client_id,
                                                                 current_phase,
                                                                 metric_substrings)
        if latest_metric is not None and latest_metric > 0:
            return latest_metric
        profile_metric = self._extract_profile_metric(clients_profiles,
                                                      client_id,
                                                      current_phase,
                                                      max_capacity,
                                                      metric_substrings)
        if profile_metric is not None and profile_metric > 0:
            return profile_metric
        return 1.0

    def _ensure_lru_cache(self,
                          client_ids: list) -> None:
        lru_cache = self.get_attribute("_lru_cache")
        known_clients_order = self.get_attribute("_known_clients_order")
        for client_id in client_ids:
            if client_id not in known_clients_order:
                known_clients_order.append(client_id)
            if client_id not in lru_cache:
                lru_cache.append(client_id)
        self._set_attribute("_lru_cache", lru_cache)
        self._set_attribute("_known_clients_order", known_clients_order)

    def _move_selected_clients_to_lru_back(self,
                                           selected_client_ids: list) -> None:
        lru_cache = self.get_attribute("_lru_cache")
        for client_id in selected_client_ids:
            if client_id in lru_cache:
                lru_cache.remove(client_id)
            lru_cache.append(client_id)
        self._set_attribute("_lru_cache", lru_cache)

    def _predict_availability_slots(self,
                                    client_id: str,
                                    num_slots: int) -> list:
        """
        Lightweight replacement for the original CNN-LSTM next-day prediction.

        The original RIFLES predictor outputs a future availability matrix.
        This function outputs a pseudo-slot vector from recent observed availability.
        """
        settings = self.get_attribute("_client_selection_settings")
        availability_history = self.get_attribute("_availability_history")
        window = int(settings.get("availability_window", 10))
        min_probability = self._safe_float(settings.get("min_availability_probability", 0.5), 0.5)
        history = availability_history.get(client_id, [])
        recent_history = history[-window:]
        if not recent_history:
            return [1 for _ in range(num_slots)]
        availability_probability = sum(recent_history) / len(recent_history)
        if availability_probability >= min_probability:
            return [1 for _ in range(num_slots)]
        return [0 for _ in range(num_slots)]

    @staticmethod
    def _predicted_available_window(predicted_slots: list,
                                    start_slot: int) -> int:
        count = 0
        for slot in range(start_slot, len(predicted_slots)):
            if predicted_slots[slot] != 1:
                break
            count += 1
        return count

    def _expected_time_to_slots(self,
                                expected_time: float) -> int:
        settings = self.get_attribute("_client_selection_settings")
        slot_duration_seconds = self._safe_float(settings.get("slot_duration_seconds", 60.0), 60.0)
        return max(1, int((expected_time + slot_duration_seconds - 1) // slot_duration_seconds))

    def _build_eligibility_matrix(self,
                                  candidate_clients: dict,
                                  clients_profiles: dict,
                                  selected_clients_metrics_history: dict,
                                  current_phase: str) -> tuple:
        """
        Build the RIFLES-style eligibility matrix.

        E_i(s) = 1 iff Lambda_i^s >= C_expected(i) + k.
        """
        settings = self.get_attribute("_client_selection_settings")
        num_slots = int(settings.get("num_slots", 24))
        buffer_slots = int(settings.get("buffer_slots", 1))
        eligibility = {}
        expected_times = {}
        for client_id, client_map in candidate_clients.items():
            expected_time = self._estimate_expected_time(client_id,
                                                         client_map,
                                                         clients_profiles,
                                                         selected_clients_metrics_history,
                                                         current_phase)
            expected_times[client_id] = expected_time
            required_slots = self._expected_time_to_slots(expected_time) + buffer_slots
            predicted_slots = self._predict_availability_slots(client_id, num_slots)
            eligibility[client_id] = []
            for slot in range(num_slots):
                available_window = self._predicted_available_window(predicted_slots, slot)
                eligibility[client_id].append(1 if available_window >= required_slots else 0)
        return eligibility, expected_times

    @staticmethod
    def _eligible_clients_at_slot(eligibility: dict,
                                  slot: int) -> list:
        return [client_id for client_id, slot_values in eligibility.items()
                if slot < len(slot_values) and slot_values[slot] == 1]

    def _select_best_rifles_slot(self,
                                 eligibility: dict,
                                 k_min: int) -> int | None:
        """
        RIFLES-GH-inspired slot selection.

        The original GH sorts slots by the number of eligible clients, checks
        K_min, and enforces a gap between scheduled slots. In this round-level
        adaptation, we return the best feasible pseudo-slot for the current FL
        round.
        """
        settings = self.get_attribute("_client_selection_settings")
        gap_slots = int(settings.get("gap_slots", 2))
        if not eligibility:
            return None
        num_slots = max(len(values) for values in eligibility.values())
        slot_counts = {}
        for slot in range(num_slots):
            slot_counts[slot] = len(self._eligible_clients_at_slot(eligibility, slot))
        sorted_slots = sorted(slot_counts.keys(), key=lambda slot: (slot_counts[slot], -slot), reverse=True)
        selected_slots = []
        for slot in sorted_slots:
            if slot_counts[slot] < k_min:
                continue
            if all(abs(slot - selected_slot) >= gap_slots for selected_slot in selected_slots):
                selected_slots.append(slot)
            if selected_slots:
                return selected_slots[0]
        return None

    def _select_clients_rifles_lru(self,
                                   eligible_client_ids: list,
                                   k_min: int,
                                   candidate_clients: dict,
                                   current_phase: str,
                                   num_tasks: int) -> list:
        """
        RIFLES-LRU policy: select least recently used eligible clients first.
        """
        lru_cache = list(self.get_attribute("_lru_cache"))
        ordered_client_ids = [client_id for client_id in lru_cache if client_id in eligible_client_ids]
        selected_client_ids = ordered_client_ids[:k_min]
        capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
        selected_capacity = sum(int(max(candidate_clients[client_id][capacity_key]))
                                for client_id in selected_client_ids)
        for client_id in ordered_client_ids[k_min:]:
            if selected_capacity >= num_tasks:
                break
            selected_client_ids.append(client_id)
            selected_capacity += int(max(candidate_clients[client_id][capacity_key]))
        return selected_client_ids

    def _select_clients_rifles_gh(self,
                                  eligible_client_ids: list,
                                  eligibility: dict,
                                  expected_times: dict,
                                  k_min: int,
                                  candidate_clients: dict,
                                  current_phase: str,
                                  num_tasks: int) -> list:
        """
        RIFLES-GH-inspired client selection.

        Priority:
          1. unique clients with few eligible slots;
          2. shorter expected execution/response time;
          3. least recently used clients as a fairness/rotation tie-breaker.
        """
        settings = self.get_attribute("_client_selection_settings")
        lru_cache = list(self.get_attribute("_lru_cache"))
        unique_threshold = self._safe_float(settings.get("unique_eligibility_threshold", 0.25), 0.25)
        unique_clients = set()
        for client_id, slot_values in eligibility.items():
            if not slot_values:
                continue
            eligibility_rate = sum(slot_values) / len(slot_values)
            if eligibility_rate < unique_threshold:
                unique_clients.add(client_id)

        def lru_position(client_id: str) -> int:
            if client_id in lru_cache:
                return lru_cache.index(client_id)
            return len(lru_cache)

        ranked_client_ids = sorted(eligible_client_ids,
                                   key=lambda client_id: (1 if client_id in unique_clients else 0,
                                                          -expected_times.get(client_id, 1.0),
                                                          -lru_position(client_id),
                                                          str(client_id)),
                                   reverse=True)
        selected_client_ids = ranked_client_ids[:k_min]
        capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
        selected_capacity = sum(int(max(candidate_clients[client_id][capacity_key]))
                                for client_id in selected_client_ids)
        for client_id in ranked_client_ids[k_min:]:
            if selected_capacity >= num_tasks:
                break
            selected_client_ids.append(client_id)
            selected_capacity += int(max(candidate_clients[client_id][capacity_key]))
        return selected_client_ids

    def _assign_tasks_and_build_selected_clients(self,
                                                 current_round: int,
                                                 current_phase: str,
                                                 candidate_clients: dict,
                                                 selected_client_ids: list,
                                                 num_tasks: int,
                                                 samples_per_task: int,
                                                 base_learning_rate: float,
                                                 base_batch_size: int,
                                                 base_num_epochs: int,
                                                 data_privacy_approach: str,
                                                 logger: Logger) -> dict:
        candidate_client_ids = list(candidate_clients.keys())
        client_aux_info = []
        for client_id in candidate_client_ids:
            client_map = candidate_clients[client_id]
            capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
            capacities = client_map[capacity_key]
            client_aux_info.append({"client_id": client_id,
                                    "max_capacity": int(max(capacities)),
                                    "capacities": capacities})
        selected_indices = [candidate_client_ids.index(client_id) for client_id in selected_client_ids]
        x_selected = self._solve_exact_task_assignment_for_selected_clients(selected_indices,
                                                                            client_aux_info,
                                                                            num_tasks)
        x = [0 for _ in candidate_client_ids]
        for pos, selected_idx in enumerate(selected_indices):
            x[selected_idx] = int(x_selected[pos])
        x_scaled = [x_i * samples_per_task for x_i in x]
        class_capacity_vectors_list, sorted_classes = build_class_capacity_vectors_list(candidate_clients,
                                                                                        data_privacy_approach,
                                                                                        current_phase)
        settings = self.get_attribute("_client_selection_settings")
        tasks_distribution_scheme = settings.get("tasks_distribution_scheme", "globally_balanced")
        if tasks_distribution_scheme == "random":
            x_dist = distribute_tasks_with_random_approach(x_scaled, class_capacity_vectors_list)
        elif tasks_distribution_scheme == "locally_balanced":
            x_dist = distribute_tasks_with_locally_balanced_approach(x_scaled, class_capacity_vectors_list)
        else:
            x_dist = distribute_tasks_with_globally_balanced_approach(x_scaled, class_capacity_vectors_list)
        x_dist = organize_tasks_distribution(x_dist, sorted_classes)
        selected_clients = {}
        for idx, x_i in enumerate(x):
            if x_i <= 0:
                continue
            client_id = candidate_client_ids[idx]
            client_map = candidate_clients[client_id]
            client_proxy = client_map["client_proxy"]
            capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
            client_task_assignment_capacities_phase = client_map[capacity_key]
            client_max_task_capacity = max(client_task_assignment_capacities_phase)
            num_samples_i = int(x_i * samples_per_task)
            x_dist_i = x_dist[idx]
            x_dist_i_str = "|".join(["{0}={1}".format(class_label, count) for class_label, count in x_dist_i.items()])
            client_info = {"client_proxy": client_proxy,
                           capacity_key: client_task_assignment_capacities_phase,
                           "client_max_task_capacity": client_max_task_capacity,
                           "client_num_tasks_scheduled": int(x_i),
                           "client_num_samples_scheduled": num_samples_i,
                           "client_num_samples_per_class_scheduled": x_dist_i_str}
            if current_phase == "train":
                client_info.update({"client_learning_rate": base_learning_rate,
                                    "client_batch_size": base_batch_size,
                                    "client_epochs": base_num_epochs})
            elif current_phase == "test":
                client_info.update({"client_batch_size": base_batch_size})
            selected_clients[client_id] = client_info
        message = "[RIFLES | Round {0}] Assigned total tasks: {1} / requested: {2}".format(current_round,
                                                                                           sum(x),
                                                                                           num_tasks)
        log_message(logger, message, "INFO")
        return selected_clients

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
                        data_privacy_approach: str,
                        logger: Logger) -> dict:
        candidate_client_ids = list(candidate_clients.keys())
        if not candidate_client_ids:
            raise ValueError("RIFLES received an empty candidate client set.")
        self._update_availability_history(current_round, candidate_clients)
        self._ensure_lru_cache(candidate_client_ids)
        min_clients_required = self._minimum_clients_required_to_cover_workload(candidate_clients,
                                                                                current_phase,
                                                                                num_tasks)
        eligibility, expected_times = self._build_eligibility_matrix(candidate_clients=candidate_clients,
                                                                     clients_profiles=clients_profiles,
                                                                     selected_clients_metrics_history=selected_clients_metrics_history,
                                                                     current_phase=current_phase)
        selected_slot = self._select_best_rifles_slot(eligibility, min_clients_required)
        if selected_slot is None:
            eligible_client_ids = candidate_client_ids
            selected_slot = -1
        else:
            eligible_client_ids = self._eligible_clients_at_slot(eligibility, selected_slot)
        if not eligible_client_ids:
            eligible_client_ids = candidate_client_ids
            selected_slot = -1
        policy = str(self.get_attribute("_client_selection_settings").get("policy", "GH")).upper()
        if policy == "LRU":
            selected_client_ids = self._select_clients_rifles_lru(eligible_client_ids=eligible_client_ids,
                                                                  k_min=min_clients_required,
                                                                  candidate_clients=candidate_clients,
                                                                  current_phase=current_phase,
                                                                  num_tasks=num_tasks)
        else:
            selected_client_ids = self._select_clients_rifles_gh(eligible_client_ids=eligible_client_ids,
                                                                 eligibility=eligibility,
                                                                 expected_times=expected_times,
                                                                 k_min=min_clients_required,
                                                                 candidate_clients=candidate_clients,
                                                                 current_phase=current_phase,
                                                                 num_tasks=num_tasks)
        if not selected_client_ids:
            raise ValueError("RIFLES could not select any client.")
        message = "[RIFLES-{0} | Round {1}] Selected pseudo-slot: {2}".format(policy,
                                                                              current_round,
                                                                              selected_slot)
        log_message(logger, message, "INFO")
        message = "[RIFLES-{0} | Round {1}] Eligible clients at slot: {2}".format(policy,
                                                                                  current_round,
                                                                                  len(eligible_client_ids))
        log_message(logger, message, "INFO")
        message = "[RIFLES-{0} | Round {1}] Selected clients: {2}".format(policy,
                                                                          current_round,
                                                                          selected_client_ids)
        log_message(logger, message, "INFO")
        selected_clients = self._assign_tasks_and_build_selected_clients(current_round=current_round,
                                                                         current_phase=current_phase,
                                                                         candidate_clients=candidate_clients,
                                                                         selected_client_ids=selected_client_ids,
                                                                         num_tasks=num_tasks,
                                                                         samples_per_task=samples_per_task,
                                                                         base_learning_rate=base_learning_rate,
                                                                         base_batch_size=base_batch_size,
                                                                         base_num_epochs=base_num_epochs,
                                                                         data_privacy_approach=data_privacy_approach,
                                                                         logger=logger)
        last_selected_round = self.get_attribute("_last_selected_round")
        selection_count = self.get_attribute("_selection_count")
        for client_id in selected_clients:
            last_selected_round[client_id] = current_round
            selection_count[client_id] = selection_count.get(client_id, 0) + 1
        self._set_attribute("_last_selected_round", last_selected_round)
        self._set_attribute("_selection_count", selection_count)
        self._move_selected_clients_to_lru_back(list(selected_clients.keys()))
        return selected_clients

    def run_client_selection_procedure(self,
                                       **kwargs) -> dict:
        current_round = kwargs["current_round"]
        current_phase = kwargs["current_phase"]
        candidate_clients = kwargs["candidate_clients"]
        num_tasks = kwargs["num_tasks"]
        samples_per_task = kwargs["samples_per_task"]
        base_learning_rate = kwargs.get("base_learning_rate", 0.001)
        base_batch_size = kwargs.get("base_batch_size", 32)
        base_num_epochs = kwargs.get("base_num_epochs", 1)
        selected_clients_metrics_history = kwargs["selected_clients_metrics_history"]
        clients_profiles = kwargs["clients_profiles"]
        data_privacy_approach = kwargs["data_privacy_approach"]
        logger = kwargs["logger"]
        capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
        task_assignment_capacities_list = [client_map[capacity_key]
                                           for _, client_map in candidate_clients.items()]
        self._set_attribute("_task_assignment_capacities_list", task_assignment_capacities_list)
        scaled_task_assignment_capacities_list = [sorted(set([capacity for capacity in capacities]))
                                                  for capacities in task_assignment_capacities_list]
        max_num_tasks = sum(max(capacities)
                            for capacities in scaled_task_assignment_capacities_list
                            if capacities)
        if isinstance(num_tasks, float):
            num_tasks = int(num_tasks * max_num_tasks)
        num_tasks = max(1, min(int(num_tasks), max_num_tasks))
        all_possible_task_assignment_sums = get_all_possible_sums(scaled_task_assignment_capacities_list)
        if num_tasks not in all_possible_task_assignment_sums:
            num_tasks = take_closest(all_possible_task_assignment_sums, num_tasks)
        selected_clients = self._select_clients(current_round=current_round,
                                                current_phase=current_phase,
                                                candidate_clients=candidate_clients,
                                                num_tasks=num_tasks,
                                                samples_per_task=samples_per_task,
                                                base_learning_rate=base_learning_rate,
                                                base_batch_size=base_batch_size,
                                                base_num_epochs=base_num_epochs,
                                                selected_clients_metrics_history=selected_clients_metrics_history,
                                                clients_profiles=clients_profiles,
                                                data_privacy_approach=data_privacy_approach,
                                                logger=logger)
        internal_candidate_clients_history = self.get_attribute("_internal_candidate_clients_history")
        if current_round not in internal_candidate_clients_history:
            internal_candidate_clients_history.update({current_round: {current_phase: candidate_clients}})
        else:
            internal_candidate_clients_history[current_round].update({current_phase: candidate_clients})
        self._set_attribute("_internal_candidate_clients_history", internal_candidate_clients_history)
        internal_selected_clients_history = self.get_attribute("_internal_selected_clients_history")
        if current_round not in internal_selected_clients_history:
            internal_selected_clients_history.update({current_round: {current_phase: selected_clients}})
        else:
            internal_selected_clients_history[current_round].update({current_phase: selected_clients})
        self._set_attribute("_internal_selected_clients_history", internal_selected_clients_history)
        for client_id, client_info in selected_clients.items():
            message = "{0}: {1} tasks → {2} samples".format(client_id,
                                                            client_info["client_num_tasks_scheduled"],
                                                            client_info["client_num_samples_scheduled"])
            log_message(logger, message, "INFO")
        return selected_clients
