from logging import Logger
from numpy.random import default_rng, SeedSequence

from metacs_fl.utils.client_selector_util import get_all_possible_sums, take_closest
from metacs_fl.utils.logger_util import log_message
from metacs_fl.utils.task_scheduler_util import build_class_capacity_vectors_list, \
    distribute_tasks_with_globally_balanced_approach, distribute_tasks_with_locally_balanced_approach, \
    distribute_tasks_with_random_approach, organize_tasks_distribution


class ECSM:

    def __init__(self,
                 client_selection_settings: dict,
                 seed: None | int | SeedSequence) -> None:
        self._client_selection_settings = client_selection_settings
        self._internal_candidate_clients_history = {}
        self._internal_selected_clients_history = {}
        self._task_assignment_capacities_list = []
        self._rng = default_rng(seed=seed)

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    def _safe_float(value: any,
                    default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_scores(scores_dict: dict) -> dict:
        if not scores_dict:
            return {}
        values = list(scores_dict.values())
        min_v = min(values)
        max_v = max(values)
        if max_v <= min_v:
            return {k: 0.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def _extract_latest_accuracy_for_client(self,
                                            selected_clients_metrics_history: dict,
                                            client_id: str,
                                            phase_preference_order: list[str]) -> float | None:
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
                        if "accuracy" in metric_key:
                            return self._safe_float(metric_value, None)
        return None

    def _extract_previous_accuracy_for_client(self,
                                              selected_clients_metrics_history: dict,
                                              client_id: str,
                                              phase_preference_order: list[str]) -> float | None:
        found_accuracies = []
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
                        if "accuracy" in metric_key:
                            found_accuracies.append(self._safe_float(metric_value, None))
                            break
                    if len(found_accuracies) >= 2:
                        return found_accuracies[1]
        return None

    def _extract_profile_accuracy_for_client(self,
                                             candidate_clients: dict,
                                             clients_profiles: dict,
                                             current_phase: str,
                                             client_id: str) -> float | None:
        phase_preference_order = ["test", "train"]
        for phase in phase_preference_order:
            if client_id not in clients_profiles:
                continue
            if phase not in clients_profiles[client_id]:
                continue
            profile_phase_dict = clients_profiles[client_id][phase]
            if not profile_phase_dict:
                continue
            capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
            capacities = candidate_clients[client_id][capacity_key]
            max_capacity = max(capacities)
            closest_x = min(profile_phase_dict.keys(), key=lambda x: abs(int(x) - max_capacity))
            profiled_metrics = profile_phase_dict[closest_x]
            for metric_key, metric_value in profiled_metrics.items():
                if "accuracy" in metric_key:
                    return self._safe_float(metric_value, None)
        return None

    def _get_latest_average_accuracy(self,
                                     selected_clients_metrics_history: dict,
                                     phase_preference_order: list[str]) -> float:
        for phase in phase_preference_order:
            for round_key in reversed(sorted(selected_clients_metrics_history.keys())):
                round_metrics = selected_clients_metrics_history[round_key]
                if phase not in round_metrics:
                    continue
                clients_metrics_dicts = round_metrics[phase].get("clients_metrics_dicts", [])
                if not clients_metrics_dicts:
                    continue
                accuracy_values = []
                for client_metrics_dict in clients_metrics_dicts:
                    client_metrics = list(client_metrics_dict.values())[0]
                    for metric_key, metric_value in client_metrics.items():
                        if "accuracy" in metric_key:
                            accuracy_values.append(self._safe_float(metric_value, 0.0))
                            break
                if accuracy_values:
                    return sum(accuracy_values) / len(accuracy_values)
        return 0.0

    def _build_ecsm_scores(self,
                           current_round: int,
                           current_phase: str,
                           candidate_clients: dict,
                           selected_clients_metrics_history: dict,
                           clients_profiles: dict) -> tuple[dict, dict, dict]:
        candidate_client_ids = list(candidate_clients.keys())
        phase_preference_order = ["test", "train"] if current_phase == "train" else ["test", "train"]
        latest_average_accuracy = self._get_latest_average_accuracy(selected_clients_metrics_history,
                                                                    phase_preference_order)
        accuracy_scores = {}
        reputation_scores = {}
        for client_id in candidate_client_ids:
            latest_accuracy = self._extract_latest_accuracy_for_client(selected_clients_metrics_history,
                                                                       client_id,
                                                                       phase_preference_order)
            if latest_accuracy is None:
                latest_accuracy = self._extract_profile_accuracy_for_client(candidate_clients,
                                                                           clients_profiles,
                                                                           current_phase,
                                                                           client_id)
            if latest_accuracy is None:
                latest_accuracy = 0.0
            previous_accuracy = self._extract_previous_accuracy_for_client(selected_clients_metrics_history,
                                                                           client_id,
                                                                           phase_preference_order)
            # ECSM reputation adaptation from the paper:
            # t = 1  -> Rep = Acc
            # t > 1 -> Rep = (Acc_{t-1} + Acc_t) / (2 * AvgAcc)
            if current_round <= 1 or previous_accuracy is None:
                reputation_value = latest_accuracy
            else:
                denominator = 2.0 * latest_average_accuracy if latest_average_accuracy > 0 else 0.0
                if denominator > 0:
                    reputation_value = (previous_accuracy + latest_accuracy) / denominator
                else:
                    reputation_value = latest_accuracy
            accuracy_scores[client_id] = latest_accuracy
            reputation_scores[client_id] = reputation_value
        normalized_accuracy_scores = self._normalize_scores(accuracy_scores)
        normalized_reputation_scores = self._normalize_scores(reputation_scores)
        return accuracy_scores, normalized_accuracy_scores, normalized_reputation_scores

    @staticmethod
    def _minimum_clients_required_to_cover_workload(candidate_clients: dict,
                                                    current_phase: str,
                                                    target_num_tasks: int) -> int:
        capacities = []
        for _, client_map in candidate_clients.items():
            capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
            client_capacities = client_map[capacity_key]
            capacities.append(max(client_capacities))
        capacities = sorted(capacities, reverse=True)
        accumulated = 0
        count = 0
        for cap in capacities:
            accumulated += cap
            count += 1
            if accumulated >= target_num_tasks:
                return count
        return len(capacities)

    @staticmethod
    def _solve_exact_task_assignment_for_selected_clients(selected_indices: list[int],
                                                          client_aux_info: list[dict],
                                                          target_num_tasks: int) -> list[int]:
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

    def _select_clients(self,
                        current_round: int,
                        current_phase: str,
                        candidate_clients: dict,
                        num_tasks: int,
                        samples_per_task: int,
                        base_learning_rate: float,
                        base_batch_size: int,
                        base_num_epochs: int,
                        selected_clients_history: dict,
                        selected_clients_metrics_history: dict,
                        clients_profiles: dict,
                        data_privacy_approach: str,
                        logger: Logger) -> dict:
        client_selection_settings = self.get_attribute("_client_selection_settings")
        rng = self.get_attribute("_rng")
        accuracy_ratio = client_selection_settings.get("accuracy_ratio", 20)
        reputation_ratio = client_selection_settings.get("reputation_ratio", 70)
        random_ratio = client_selection_settings.get("random_ratio", 10)
        ratio_sum = accuracy_ratio + reputation_ratio + random_ratio
        if ratio_sum <= 0:
            accuracy_ratio = 20
            reputation_ratio = 70
            random_ratio = 10
            ratio_sum = 100
        accuracy_ratio = accuracy_ratio / ratio_sum
        reputation_ratio = reputation_ratio / ratio_sum
        random_ratio = random_ratio / ratio_sum
        candidate_client_ids = list(candidate_clients.keys())
        client_aux_info = []
        for client_id in candidate_client_ids:
            client_map = candidate_clients[client_id]
            capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
            capacities = client_map[capacity_key]
            client_aux_info.append({"client_id": client_id,
                                    "max_capacity": int(max(capacities)),
                                    "capacities": capacities})
        accuracy_scores_raw, accuracy_scores, reputation_scores = self._build_ecsm_scores(current_round,
                                                                                          current_phase,
                                                                                          candidate_clients,
                                                                                          selected_clients_metrics_history,
                                                                                          clients_profiles)
        # Adaptation: instead of alpha*K fixed-size selection, determine the minimum number of clients
        # needed to satisfy the target workload, then split that pool according to ECSM(A,R,r).
        min_clients_required = self._minimum_clients_required_to_cover_workload(candidate_clients,
                                                                               current_phase,
                                                                               num_tasks)
        if min_clients_required <= 0:
            raise ValueError("ECSM could not determine a valid client pool size.")
        num_accuracy_clients = max(0, int(round(accuracy_ratio * min_clients_required)))
        num_reputation_clients = max(0, int(round(reputation_ratio * min_clients_required)))
        num_random_clients = max(0, int(round(random_ratio * min_clients_required)))
        allocated = num_accuracy_clients + num_reputation_clients + num_random_clients
        if allocated < min_clients_required:
            deficit = min_clients_required - allocated
            num_reputation_clients += deficit
        elif allocated > min_clients_required:
            overflow = allocated - min_clients_required
            reduction = min(overflow, num_reputation_clients)
            num_reputation_clients -= reduction
            overflow -= reduction
            if overflow > 0:
                reduction = min(overflow, num_accuracy_clients)
                num_accuracy_clients -= reduction
                overflow -= reduction
            if overflow > 0:
                num_random_clients = max(0, num_random_clients - overflow)
        accuracy_ranked = sorted(candidate_client_ids,
                                 key=lambda cid: (accuracy_scores[cid], cid),
                                 reverse=True)
        reputation_ranked = sorted(candidate_client_ids,
                                   key=lambda cid: (reputation_scores[cid], cid),
                                   reverse=True)
        selected_client_ids = []
        # Accuracy-based selection.
        for client_id in accuracy_ranked:
            if len([cid for cid in selected_client_ids if cid in accuracy_ranked[:len(selected_client_ids)]]) >= num_accuracy_clients:
                break
            if client_id not in selected_client_ids:
                selected_client_ids.append(client_id)
            if len(selected_client_ids) >= num_accuracy_clients:
                break
        # Reputation-based selection.
        reputation_selected = 0
        for client_id in reputation_ranked:
            if reputation_selected >= num_reputation_clients:
                break
            if client_id not in selected_client_ids:
                selected_client_ids.append(client_id)
                reputation_selected += 1
        # Random-based selection.
        remaining_client_ids = [cid for cid in candidate_client_ids if cid not in selected_client_ids]
        if remaining_client_ids and num_random_clients > 0:
            shuffled_remaining = list(remaining_client_ids)
            rng.shuffle(shuffled_remaining)
            for client_id in shuffled_remaining[:num_random_clients]:
                if client_id not in selected_client_ids:
                    selected_client_ids.append(client_id)
        # If the selected pool still cannot satisfy the workload, add more clients
        # using an ensemble score dominated by reputation and accuracy.
        selected_set = set(selected_client_ids)
        selected_capacity = sum(client_aux_info[candidate_client_ids.index(cid)]["max_capacity"]
                                for cid in selected_client_ids)
        combined_scores = {}
        for client_id in candidate_client_ids:
            combined_scores[client_id] = 0.5 * reputation_scores[client_id] + 0.5 * accuracy_scores[client_id]
        remaining_ranked = sorted([cid for cid in candidate_client_ids if cid not in selected_set],
                                  key=lambda cid: (combined_scores[cid], accuracy_scores_raw[cid], cid),
                                  reverse=True)
        for client_id in remaining_ranked:
            if selected_capacity >= num_tasks:
                break
            selected_client_ids.append(client_id)
            selected_set.add(client_id)
            selected_capacity += client_aux_info[candidate_client_ids.index(client_id)]["max_capacity"]
        if not selected_client_ids:
            raise ValueError("ECSM could not select any client.")
        selected_indices = [candidate_client_ids.index(client_id) for client_id in selected_client_ids]
        X_selected = self._solve_exact_task_assignment_for_selected_clients(selected_indices,
                                                                           client_aux_info,
                                                                           num_tasks)
        X = [0 for _ in candidate_client_ids]
        for pos, selected_idx in enumerate(selected_indices):
            X[selected_idx] = int(X_selected[pos])
        X_scaled = [x_i * samples_per_task for x_i in X]
        class_capacity_vectors_list, sorted_classes = build_class_capacity_vectors_list(candidate_clients,
                                                                                        data_privacy_approach,
                                                                                        current_phase)
        tasks_distribution_scheme = client_selection_settings.get("tasks_distribution_scheme", "globally_balanced")
        if tasks_distribution_scheme == "random":
            X_dist = distribute_tasks_with_random_approach(X_scaled, class_capacity_vectors_list)
        elif tasks_distribution_scheme == "locally_balanced":
            X_dist = distribute_tasks_with_locally_balanced_approach(X_scaled, class_capacity_vectors_list)
        else:
            X_dist = distribute_tasks_with_globally_balanced_approach(X_scaled, class_capacity_vectors_list)
        X_dist = organize_tasks_distribution(X_dist, sorted_classes)
        message = "[ECSM | Round {0}] Accuracy-based ratio: {1}, reputation-based ratio: {2}, random-based ratio: {3}" \
                  .format(current_round,
                          round(accuracy_ratio * 100, 2),
                          round(reputation_ratio * 100, 2),
                          round(random_ratio * 100, 2))
        log_message(logger, message, "INFO")
        message = "[ECSM | Round {0}] Selected clients: {1}" \
                  .format(current_round, selected_client_ids)
        log_message(logger, message, "INFO")
        message = "[ECSM | Round {0}] Assigned total tasks: {1} / requested: {2}" \
                  .format(current_round, sum(X), num_tasks)
        log_message(logger, message, "INFO")
        selected_clients = {}
        for i, x_i in enumerate(X):
            if x_i <= 0:
                continue
            client_id_str = candidate_client_ids[i]
            client_map = candidate_clients[client_id_str]
            client_proxy = client_map["client_proxy"]
            capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
            client_task_assignment_capacities_phase = client_map[capacity_key]
            client_max_task_capacity = max(client_task_assignment_capacities_phase)
            x_dist_i = X_dist[i]
            num_samples_i = int(x_i * samples_per_task)
            x_dist_i_str = "|".join(["{0}={1}".format(k, v) for k, v in x_dist_i.items()])
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
            selected_clients[client_id_str] = client_info
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
        selected_clients_history = kwargs["selected_clients_history"]
        selected_clients_metrics_history = kwargs["selected_clients_metrics_history"]
        clients_profiles = kwargs["clients_profiles"]
        data_privacy_approach = kwargs["data_privacy_approach"]
        logger = kwargs["logger"]
        task_assignment_capacities_list = [client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                                           for _, client_map in candidate_clients.items()]
        self._set_attribute("_task_assignment_capacities_list", task_assignment_capacities_list)
        scaled_task_assignment_capacities_list = [sorted(set([capacity for capacity in capacities]))
                                                  for capacities in task_assignment_capacities_list]
        max_num_tasks = sum(max(capacities) for capacities in scaled_task_assignment_capacities_list if capacities)
        if isinstance(num_tasks, float):
            num_tasks = int(num_tasks * max_num_tasks)
        num_tasks = max(1, min(int(num_tasks), max_num_tasks))
        all_possible_task_assignment_sums = get_all_possible_sums(scaled_task_assignment_capacities_list)
        if num_tasks not in all_possible_task_assignment_sums:
            num_tasks = take_closest(all_possible_task_assignment_sums, num_tasks)
        selected_clients = self._select_clients(current_round,
                                                current_phase,
                                                candidate_clients,
                                                num_tasks,
                                                samples_per_task,
                                                base_learning_rate,
                                                base_batch_size,
                                                base_num_epochs,
                                                selected_clients_history,
                                                selected_clients_metrics_history,
                                                clients_profiles,
                                                data_privacy_approach,
                                                logger)
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
