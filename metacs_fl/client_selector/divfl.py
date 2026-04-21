from logging import Logger
from math import sqrt
from numpy import ndarray, linalg, zeros, array, clip, maximum, sum, concatenate
from numpy.random import default_rng, SeedSequence

from metacs_fl.utils.client_selector_util import get_all_possible_sums, take_closest
from metacs_fl.utils.logger_util import log_message
from metacs_fl.utils.task_scheduler_util import build_class_capacity_vectors_list, \
    distribute_tasks_with_globally_balanced_approach, distribute_tasks_with_locally_balanced_approach, \
    distribute_tasks_with_random_approach, organize_tasks_distribution


class DivFL:

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
    def _normalize_vector(vec: ndarray) -> ndarray:
        norm = linalg.norm(vec)
        if norm <= 0:
            return vec
        return vec / norm

    @staticmethod
    def _min_max_normalize_list(values: list[float]) -> list[float]:
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if max_v <= min_v:
            return [0.0 for _ in values]
        return [(v - min_v) / (max_v - min_v) for v in values]

    def _cosine_similarity_matrix(self,
                                  representations: ndarray) -> ndarray:
        if representations.size == 0:
            return zeros((0, 0), dtype=float)
        normalized_repr = []
        for row in representations:
            normalized_repr.append(self._normalize_vector(row))
        normalized_repr = array(normalized_repr, dtype=float)
        similarity_matrix = normalized_repr @ normalized_repr.T
        similarity_matrix = clip(similarity_matrix, -1.0, 1.0)
        similarity_matrix = (similarity_matrix + 1.0) / 2.0
        return similarity_matrix

    def _extract_latest_metric_for_client(self,
                                          selected_clients_metrics_history: dict,
                                          current_phase: str,
                                          client_id: str,
                                          metric_names: list[str]) -> float | None:
        for round_key in reversed(sorted(selected_clients_metrics_history.keys())):
            round_metrics = selected_clients_metrics_history[round_key]
            if current_phase not in round_metrics:
                continue
            clients_metrics_dicts = round_metrics[current_phase].get("clients_metrics_dicts", [])
            for client_metrics_dict in clients_metrics_dicts:
                if client_id not in client_metrics_dict:
                    continue
                client_metrics = client_metrics_dict[client_id]
                for metric_name in metric_names:
                    if metric_name in client_metrics:
                        return self._safe_float(client_metrics[metric_name], None)
        return None

    def _extract_profile_metric_for_client(self,
                                           candidate_clients: dict,
                                           clients_profiles: dict,
                                           current_phase: str,
                                           client_id: str,
                                           metric_names: list[str]) -> float | None:
        if client_id not in clients_profiles:
            return None
        if current_phase not in clients_profiles[client_id]:
            return None
        profile_phase_dict = clients_profiles[client_id][current_phase]
        if not profile_phase_dict:
            return None
        capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
        capacities = candidate_clients[client_id][capacity_key]
        max_capacity = max(capacities)
        closest_x = min(profile_phase_dict.keys(), key=lambda x: abs(int(x) - max_capacity))
        profiled_metrics = profile_phase_dict[closest_x]
        for metric_name in metric_names:
            if metric_name in profiled_metrics:
                return self._safe_float(profiled_metrics[metric_name], None)
        return None

    def _build_client_representations(self,
                                      current_phase: str,
                                      candidate_clients: dict,
                                      selected_clients_metrics_history: dict,
                                      clients_profiles: dict,
                                      data_privacy_approach: str) -> tuple[list[str], ndarray, list[dict], list]:
        candidate_client_ids = list(candidate_clients.keys())
        class_capacity_vectors_list, sorted_classes = build_class_capacity_vectors_list(candidate_clients,
                                                                                        data_privacy_approach,
                                                                                        current_phase)
        raw_feature_rows = []
        client_aux_info = []
        # First pass: collect raw scalar features for normalization.
        raw_losses = []
        raw_times = []
        raw_energies = []
        raw_batteries = []
        raw_bandwidths = []
        raw_latencies = []
        raw_max_capacities = []
        phase_time_key = "{0}ing_time_in_seconds".format(current_phase)
        phase_energy_key = "{0}ing_energy_in_joules".format(current_phase)
        for client_id in candidate_client_ids:
            client_map = candidate_clients[client_id]
            capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
            capacities = client_map[capacity_key]
            max_capacity = max(capacities)
            min_positive_capacity = min([c for c in capacities if c > 0], default=max_capacity)
            latest_loss = self._extract_latest_metric_for_client(selected_clients_metrics_history,
                                                                 current_phase,
                                                                 client_id,
                                                                 ["loss"])
            if latest_loss is None:
                latest_loss = self._extract_profile_metric_for_client(candidate_clients,
                                                                      clients_profiles,
                                                                      current_phase,
                                                                      client_id,
                                                                      ["loss"])
            latest_time = self._extract_latest_metric_for_client(selected_clients_metrics_history,
                                                                 current_phase,
                                                                 client_id,
                                                                 [phase_time_key])
            if latest_time is None:
                latest_time = self._extract_profile_metric_for_client(candidate_clients,
                                                                      clients_profiles,
                                                                      current_phase,
                                                                      client_id,
                                                                      [phase_time_key])
            latest_energy = self._extract_latest_metric_for_client(selected_clients_metrics_history,
                                                                   current_phase,
                                                                   client_id,
                                                                   [phase_energy_key])
            if latest_energy is None:
                latest_energy = self._extract_profile_metric_for_client(candidate_clients,
                                                                        clients_profiles,
                                                                        current_phase,
                                                                        client_id,
                                                                        [phase_energy_key])
            battery = self._safe_float(client_map.get("client_remaining_battery_energy", 0.0), 0.0)
            bw_down = self._safe_float(client_map.get("client_current_download_bandwidth_in_bytes_per_second", 0.0), 0.0)
            bw_up = self._safe_float(client_map.get("client_current_upload_bandwidth_in_bytes_per_second", 0.0), 0.0)
            bandwidth = (bw_down + bw_up) / 2.0
            latency = self._safe_float(client_map.get("client_current_latency_in_milliseconds", 0.0), 0.0)
            raw_losses.append(0.0 if latest_loss is None else latest_loss)
            raw_times.append(0.0 if latest_time is None else latest_time)
            raw_energies.append(0.0 if latest_energy is None else latest_energy)
            raw_batteries.append(battery)
            raw_bandwidths.append(bandwidth)
            raw_latencies.append(latency)
            raw_max_capacities.append(float(max_capacity))
            client_aux_info.append({"client_id": client_id,
                                    "max_capacity": int(max_capacity),
                                    "min_positive_capacity": int(min_positive_capacity),
                                    "capacities": capacities})
        norm_losses = self._min_max_normalize_list(raw_losses)
        norm_times = self._min_max_normalize_list(raw_times)
        norm_energies = self._min_max_normalize_list(raw_energies)
        norm_batteries = self._min_max_normalize_list(raw_batteries)
        norm_bandwidths = self._min_max_normalize_list(raw_bandwidths)
        norm_latencies = self._min_max_normalize_list(raw_latencies)
        norm_max_capacities = self._min_max_normalize_list(raw_max_capacities)
        # Second pass: create final representation vector.
        for idx, client_id in enumerate(candidate_client_ids):
            class_capacity_vec = array(class_capacity_vectors_list[idx], dtype=float)
            # Normalize class vector by its own sum so the representation focuses on composition,
            # not only on absolute size.
            class_sum = float(sum(class_capacity_vec))
            if class_sum > 0:
                class_capacity_vec = class_capacity_vec / class_sum
            # Scalar proxy features.
            # We convert "good" quantities to larger-is-better when useful.
            scalar_features = array([1.0 - norm_losses[idx],        # lower loss -> better
                                           1.0 - norm_times[idx],         # lower time -> better
                                           1.0 - norm_energies[idx],      # lower energy -> better
                                           norm_batteries[idx],           # larger battery -> better
                                           norm_bandwidths[idx],          # larger bandwidth -> better
                                           1.0 - norm_latencies[idx],     # lower latency -> better
                                           norm_max_capacities[idx],      # larger capacity available
                                           ], dtype=float)
            # The final representation concatenates data-composition information with systems/profile features.
            repr_vec = concatenate([class_capacity_vec, scalar_features], axis=0)
            raw_feature_rows.append(repr_vec)
        representations = array(raw_feature_rows, dtype=float)
        return candidate_client_ids, representations, client_aux_info, sorted_classes

    def _build_gradient_based_client_representations(self,
                                                     current_round: int,
                                                     candidate_clients: dict,
                                                     client_gradients_history: dict,
                                                     selected_clients_metrics_history: dict,
                                                     clients_profiles: dict,
                                                     data_privacy_approach: str) -> tuple[list[str], ndarray, list[dict], list]:
        candidate_client_ids = list(candidate_clients.keys())
        class_capacity_vectors_list, sorted_classes = build_class_capacity_vectors_list(candidate_clients,
                                                                                        data_privacy_approach,
                                                                                        "train")
        client_aux_info = []
        raw_feature_rows = []
        latest_round_with_updates = None
        valid_rounds = sorted(client_gradients_history.keys(), reverse=True)
        for round_idx in valid_rounds:
            round_updates = client_gradients_history[round_idx]
            if any(client_id in round_updates for client_id in candidate_client_ids):
                latest_round_with_updates = round_idx
                break
        if latest_round_with_updates is None:
            return self._build_client_representations("train",
                                                      candidate_clients,
                                                      selected_clients_metrics_history,
                                                      clients_profiles,
                                                      data_privacy_approach)
        round_updates = client_gradients_history[latest_round_with_updates]
        available_update_vectors = [round_updates[client_id]
                                    for client_id in candidate_client_ids
                                    if client_id in round_updates]
        if not available_update_vectors:
            return self._build_client_representations("train",
                                                      candidate_clients,
                                                      selected_clients_metrics_history,
                                                      clients_profiles,
                                                      data_privacy_approach)
        target_dim = len(available_update_vectors[0])
        for client_id in candidate_client_ids:
            client_map = candidate_clients[client_id]
            capacities = client_map["client_task_assignment_capacities_train"]
            max_capacity = max(capacities)
            min_positive_capacity = min([c for c in capacities if c > 0], default=max_capacity)
            client_aux_info.append({"client_id": client_id,
                                    "max_capacity": int(max_capacity),
                                    "min_positive_capacity": int(min_positive_capacity),
                                    "capacities": capacities})
            if client_id in round_updates:
                update_vec = round_updates[client_id]
                if len(update_vec) != target_dim:
                    if len(update_vec) > target_dim:
                        update_vec = update_vec[:target_dim]
                    else:
                        padding = zeros(target_dim - len(update_vec), dtype=float)
                        update_vec = concatenate((update_vec, padding))
            else:
                update_vec = zeros(target_dim, dtype=float)
            raw_feature_rows.append(update_vec)
        representations = array(raw_feature_rows, dtype=float)
        return candidate_client_ids, representations, client_aux_info, sorted_classes

    @staticmethod
    def _facility_location_value(selected_indices: set[int],
                                 similarity_matrix: ndarray) -> float:
        if not selected_indices:
            return 0.0
        selected_list = sorted(list(selected_indices))
        total_value = 0.0
        num_clients = similarity_matrix.shape[0]
        for i in range(num_clients):
            best_similarity = 0.0
            for j in selected_list:
                if similarity_matrix[i, j] > best_similarity:
                    best_similarity = similarity_matrix[i, j]
            total_value += best_similarity
        return float(total_value)

    @staticmethod
    def _marginal_gain(candidate_idx: int,
                       selected_indices: set[int],
                       current_cover: ndarray,
                       similarity_matrix: ndarray) -> float:
        new_cover = maximum(current_cover, similarity_matrix[:, candidate_idx])
        return float(sum(new_cover - current_cover))

    def _select_diverse_clients_under_capacity(self,
                                               similarity_matrix: ndarray,
                                               client_aux_info: list[dict],
                                               target_num_tasks: int,
                                               minimum_num_clients: int = 1) -> list[int]:
        num_clients = similarity_matrix.shape[0]
        if num_clients == 0:
            return []
        selected_indices = []
        selected_set = set()
        current_cover = zeros(num_clients, dtype=float)
        accumulated_max_capacity = 0
        # Greedy facility-location selection with a mild bias toward capacity efficiency.
        while True:
            stop_by_capacity = accumulated_max_capacity >= target_num_tasks
            stop_by_min_clients = len(selected_indices) >= minimum_num_clients
            if stop_by_capacity and stop_by_min_clients:
                break
            best_idx = None
            best_score = -float("inf")
            for candidate_idx in range(num_clients):
                if candidate_idx in selected_set:
                    continue
                marginal_gain = self._marginal_gain(candidate_idx,
                                                    selected_set,
                                                    current_cover,
                                                    similarity_matrix)
                candidate_capacity = client_aux_info[candidate_idx]["max_capacity"]
                score = marginal_gain / max(1.0, sqrt(float(candidate_capacity)))
                # Small tie-break in favor of larger capacity.
                score += 1e-9 * float(candidate_capacity)
                if score > best_score:
                    best_score = score
                    best_idx = candidate_idx
            if best_idx is None:
                break
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            current_cover = maximum(current_cover, similarity_matrix[:, best_idx])
            accumulated_max_capacity += client_aux_info[best_idx]["max_capacity"]
            if len(selected_indices) == num_clients:
                break
        return selected_indices

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
        # Dynamic programming for one-choice-per-client exact sum.
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
        # Fallback: closest reachable below target.
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
                        client_gradients_history: dict,
                        logger: Logger) -> dict:
        client_selection_settings = self.get_attribute("_client_selection_settings")
        if current_phase == "train":
            candidate_client_ids, representations, client_aux_info, sorted_classes = \
                self._build_gradient_based_client_representations(current_round,
                                                                  candidate_clients,
                                                                  client_gradients_history,
                                                                  selected_clients_metrics_history,
                                                                  clients_profiles,
                                                                  data_privacy_approach)
        else:
            candidate_client_ids, representations, client_aux_info, sorted_classes = \
                self._build_client_representations(current_phase,
                                                   candidate_clients,
                                                   selected_clients_metrics_history,
                                                   clients_profiles,
                                                   data_privacy_approach)
        similarity_matrix = self._cosine_similarity_matrix(representations)
        minimum_num_clients = client_selection_settings.get("minimum_num_clients", 1)
        selected_indices = self._select_diverse_clients_under_capacity(similarity_matrix,
                                                                       client_aux_info,
                                                                       num_tasks,
                                                                       minimum_num_clients)
        if not selected_indices:
            raise ValueError("DivFL could not select any client.")
        # Second stage: exact one-capacity-per-selected-client assignment.
        X_selected = self._solve_exact_task_assignment_for_selected_clients(selected_indices,
                                                                            client_aux_info,
                                                                            num_tasks)
        # Build full X over all candidate clients.
        X = [0 for _ in candidate_client_ids]
        for pos, selected_idx in enumerate(selected_indices):
            X[selected_idx] = int(X_selected[pos])
        # If exact assignment under selected subset is still below target due to feasibility,
        # try augmenting with more clients until the target becomes representable.
        selected_set = set(selected_indices)
        while sum(X) < num_tasks and len(selected_set) < len(candidate_client_ids):
            remaining_indices = [idx for idx in range(len(candidate_client_ids)) if idx not in selected_set]
            if not remaining_indices:
                break
            best_extra_idx = None
            best_extra_score = -float("inf")
            current_cover = zeros(len(candidate_client_ids), dtype=float)
            if selected_set:
                selected_list = sorted(list(selected_set))
                current_cover = similarity_matrix[:, selected_list].max(axis=1)
            for candidate_idx in remaining_indices:
                marginal_gain = self._marginal_gain(candidate_idx,
                                                    selected_set,
                                                    current_cover,
                                                    similarity_matrix)
                candidate_capacity = client_aux_info[candidate_idx]["max_capacity"]
                score = marginal_gain / max(1.0, sqrt(float(candidate_capacity)))
                if score > best_extra_score:
                    best_extra_score = score
                    best_extra_idx = candidate_idx
            if best_extra_idx is None:
                break
            selected_indices.append(best_extra_idx)
            selected_set.add(best_extra_idx)
            X_selected = self._solve_exact_task_assignment_for_selected_clients(selected_indices,
                                                                                client_aux_info,
                                                                                num_tasks)
            X = [0 for _ in candidate_client_ids]
            for pos, selected_idx in enumerate(selected_indices):
                X[selected_idx] = int(X_selected[pos])
            if sum(X) >= num_tasks:
                break
        # Distribute the selected workload per class, but now in samples.
        X_scaled = [x_i * samples_per_task for x_i in X]
        class_capacity_vectors_list, _ = build_class_capacity_vectors_list(candidate_clients,
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
        # Logging.
        selected_client_ids = [candidate_client_ids[idx] for idx in selected_indices]
        facility_value = self._facility_location_value(set(selected_indices), similarity_matrix)
        message = "[DivFL | Round {0}] Selected clients: {1}".format(current_round, selected_client_ids)
        log_message(logger, message, "INFO")
        message = "[DivFL | Round {0}] Facility-location value of selected subset: {1}".format(current_round,
                                                                                                round(facility_value, 6))
        log_message(logger, message, "INFO")
        message = "[DivFL | Round {0}] Assigned total tasks: {1} / requested: {2}".format(current_round,
                                                                                           sum(X),
                                                                                           num_tasks)
        log_message(logger, message, "INFO")
        # Build final selected_clients dict in the same shape used by MetaCS-FL.
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
        client_gradients_history = kwargs.get("client_gradients_history", {})
        data_privacy_approach = kwargs["data_privacy_approach"]
        logger = kwargs["logger"]
        task_assignment_capacities_list = [client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                                           for _, client_map in candidate_clients.items()]
        self._set_attribute("_task_assignment_capacities_list", task_assignment_capacities_list)
        # num_tasks in your code path may be a float fraction.
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
                                                client_gradients_history,
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
