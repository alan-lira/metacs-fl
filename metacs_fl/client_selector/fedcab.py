from logging import Logger
from math import log
from typing import Any

from numpy.random import default_rng, SeedSequence

from metacs_fl.utils.client_selector_util import get_all_possible_sums, take_closest
from metacs_fl.utils.logger_util import log_message
from metacs_fl.utils.task_scheduler_util import build_class_capacity_vectors_list, \
    distribute_tasks_with_globally_balanced_approach, distribute_tasks_with_locally_balanced_approach, \
    distribute_tasks_with_random_approach, organize_tasks_distribution


class FedCAB:
    """
    Workload-aware FedCAB adaptation for the MetaCS-FL pipeline.

    Behavior preserved:
      - ranks currently available clients;
      - uses a KL-divergence heterogeneity signal;
      - uses the original FedCAB quadratic-to-linear rank weighting;
      - compensates less frequently selected clients through availability/participation rank;
      - applies per-client late-joining compensation beta with decay;
      - applies gamma compensation to below-average participants.

    Adaptation needed for MetaCS-FL:
      - original FedCAB computes KL(local updated model || global model) after a
        pre-selection local update. Here, to avoid training every available client
        before selection, KL is computed from the available class-distribution
        vector against the available/global class-distribution vector.
      - original FedCAB selects K clients. Here, the selected set is extended only
        as needed to cover the server-defined workload, then tasks are assigned
        using the existing feasible-capacity mechanism.
    """

    def __init__(self,
                 client_selection_settings: dict,
                 seed: None | int | SeedSequence) -> None:
        self._client_selection_settings = client_selection_settings
        self._internal_candidate_clients_history = {}
        self._internal_selected_clients_history = {}
        self._task_assignment_capacities_list = []
        self._first_seen_round = {}
        self._selection_count = {}
        self._beta_by_client = {}
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

    @staticmethod
    def _kl_divergence(p: list,
                       q: list,
                       epsilon: float = 1e-12) -> float:
        size = max(len(p), len(q))
        p = p + [0.0 for _ in range(size - len(p))]
        q = q + [0.0 for _ in range(size - len(q))]
        divergence = 0.0
        for p_i, q_i in zip(p, q):
            p_i = max(float(p_i), epsilon)
            q_i = max(float(q_i), epsilon)
            divergence += p_i * log(p_i / q_i)
        return divergence

    @staticmethod
    def _rank_positions_desc(values_dict: dict) -> dict:
        """
        Return 1-based ranks after sorting values from high to low.

        Highest value receives rank 1. Lowest value receives rank m_t.
        This matches the FedCAB notation where clients are sorted by a signal
        and then the rank position is used inside the weighting function.
        """
        ranked_client_ids = sorted(values_dict.keys(),
                                   key=lambda client_id: (values_dict[client_id], str(client_id)),
                                   reverse=True)
        return {client_id: rank + 1 for rank, client_id in enumerate(ranked_client_ids)}

    def _get_client_distribution_vector(self,
                                        client_id: str,
                                        candidate_clients: dict,
                                        class_capacity_vectors_list: list,
                                        candidate_client_ids: list) -> list:
        client_idx = candidate_client_ids.index(client_id)
        vector = class_capacity_vectors_list[client_idx]
        if isinstance(vector, dict):
            values = list(vector.values())
        else:
            values = list(vector)
        values = [self._safe_float(value, 0.0) for value in values]
        total = sum(values)
        if total <= 0:
            if not values:
                return [1.0]
            return [1.0 / len(values) for _ in values]
        return [value / total for value in values]

    def _build_global_distribution(self,
                                   candidate_client_ids: list,
                                   candidate_clients: dict,
                                   class_capacity_vectors_list: list) -> list:
        vectors = []
        for client_id in candidate_client_ids:
            vectors.append(self._get_client_distribution_vector(client_id,
                                                                candidate_clients,
                                                                class_capacity_vectors_list,
                                                                candidate_client_ids))
        size = max(len(vector) for vector in vectors) if vectors else 1
        global_vector = [0.0 for _ in range(size)]
        for vector in vectors:
            for idx, value in enumerate(vector):
                global_vector[idx] += value
        total = sum(global_vector)
        if total <= 0:
            return [1.0 / size for _ in range(size)]
        return [value / total for value in global_vector]

    def _update_first_seen_round(self,
                                 current_round: int,
                                 candidate_client_ids: list) -> None:
        first_seen_round = self.get_attribute("_first_seen_round")
        for client_id in candidate_client_ids:
            if client_id not in first_seen_round:
                first_seen_round[client_id] = current_round
        self._set_attribute("_first_seen_round", first_seen_round)

    def _get_or_initialize_beta(self,
                                client_id: str) -> float:
        settings = self.get_attribute("_client_selection_settings")
        beta_init = self._safe_float(settings.get("beta", 2.0), 2.0)
        beta_by_client = self.get_attribute("_beta_by_client")
        if client_id not in beta_by_client:
            beta_by_client[client_id] = beta_init
            self._set_attribute("_beta_by_client", beta_by_client)
        return beta_by_client[client_id]

    def _decay_beta(self,
                    client_id: str) -> None:
        settings = self.get_attribute("_client_selection_settings")
        delta_beta = self._safe_float(settings.get("delta_beta", 0.01), 0.01)
        beta_by_client = self.get_attribute("_beta_by_client")
        beta_k = self._get_or_initialize_beta(client_id)
        beta_by_client[client_id] = max(beta_k - delta_beta, 1.0)
        self._set_attribute("_beta_by_client", beta_by_client)

    def _build_fedcab_scores(self,
                             current_round: int,
                             candidate_clients: dict,
                             class_capacity_vectors_list: list,
                             candidate_client_ids: list) -> dict:
        """
        Build FedCAB-like ranking scores.

        Original FedCAB score, adapted:
          if alpha_t > 1:
              R_t^k = (b0 * P_L^2 + b1 * P_L + b2) * P_A / m_t * beta_k
          else:
              R_t^k = P_L * P_A / m_t^2 * beta_k

        where P_L is the rank by heterogeneity and P_A is the rank by the
        degree of client availability/participation.
        """
        settings = self.get_attribute("_client_selection_settings")
        selection_count = self.get_attribute("_selection_count")
        alpha_init = self._safe_float(settings.get("alpha", 3.0), 3.0)
        delta_alpha = self._safe_float(settings.get("delta_alpha", 0.2), 0.2)
        gamma = self._safe_float(settings.get("gamma", 1.1), 1.1)
        m_t = len(candidate_client_ids)
        if m_t == 0:
            return {}
        alpha_t = alpha_init - delta_alpha * max(0, current_round - 1)
        global_distribution = self._build_global_distribution(candidate_client_ids,
                                                              candidate_clients,
                                                              class_capacity_vectors_list)
        kl_scores = {}
        for client_id in candidate_client_ids:
            client_distribution = self._get_client_distribution_vector(client_id,
                                                                       candidate_clients,
                                                                       class_capacity_vectors_list,
                                                                       candidate_client_ids)
            kl_scores[client_id] = self._kl_divergence(client_distribution,
                                                       global_distribution)
        # Higher KL means more heterogeneous.
        # High-KL clients receive lower rank numbers and are favored during the quadratic phase.
        p_l = self._rank_positions_desc(kl_scores)
        # A_k = n_k / t in FedCAB. Higher A_k means more frequent historical participation.
        # Since ranks are sorted descending, under-participating clients get larger P_A and therefore larger P_A / m_t factors.
        availability_degree = {}
        for client_id in candidate_client_ids:
            n_k = selection_count.get(client_id, 0)
            availability_degree[client_id] = n_k / max(current_round, 1)
        p_a = self._rank_positions_desc(availability_degree)
        if alpha_t > 1.0:
            b0 = (alpha_t - 1.0) / (m_t ** 2)
            b1 = (-2.0 * m_t * (alpha_t - 1.0)) / (m_t ** 2)
            b2 = alpha_t
        else:
            b0 = b1 = 0.0
            b2 = 1.0
        total_updates = sum(selection_count.values())
        average_updates_among_candidates = total_updates / max(m_t, 1)
        scores = {}
        for client_id in candidate_client_ids:
            beta_k = self._get_or_initialize_beta(client_id)
            if alpha_t > 1.0:
                data_weight = (b0 * (p_l[client_id] ** 2) + b1 * p_l[client_id] + b2)
                score = data_weight * (p_a[client_id] / m_t) * beta_k
            else:
                # Linear stage: larger P_L now favors lower-KL clients,
                # matching the FedCAB transition toward convergence acceleration.
                score = (p_l[client_id] * p_a[client_id] / (m_t ** 2) * beta_k)
            if selection_count.get(client_id, 0) < average_updates_among_candidates:
                score *= gamma
            scores[client_id] = score
            # FedCAB decays beta for clients after they participate in ranking.
            self._decay_beta(client_id)
        return scores

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
        message = "[FedCAB | Round {0}] Assigned total tasks: {1} / requested: {2}".format(current_round,
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
                        data_privacy_approach: str,
                        logger: Logger) -> dict:
        candidate_client_ids = list(candidate_clients.keys())
        if not candidate_client_ids:
            raise ValueError("FedCAB received an empty candidate client set.")
        self._update_first_seen_round(current_round, candidate_client_ids)
        class_capacity_vectors_list, _ = build_class_capacity_vectors_list(candidate_clients,
                                                                           data_privacy_approach,
                                                                           current_phase)
        fedcab_scores = self._build_fedcab_scores(current_round,
                                                  candidate_clients,
                                                  class_capacity_vectors_list,
                                                  candidate_client_ids)
        ranked_client_ids = sorted(candidate_client_ids,
                                   key=lambda client_id: (fedcab_scores[client_id], str(client_id)),
                                   reverse=True)
        min_clients_required = self._minimum_clients_required_to_cover_workload(candidate_clients,
                                                                                current_phase,
                                                                                num_tasks)
        selected_client_ids = ranked_client_ids[:min_clients_required]
        selected_capacity = 0
        capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
        for client_id in selected_client_ids:
            selected_capacity += int(max(candidate_clients[client_id][capacity_key]))
        for client_id in ranked_client_ids[min_clients_required:]:
            if selected_capacity >= num_tasks:
                break
            selected_client_ids.append(client_id)
            selected_capacity += int(max(candidate_clients[client_id][capacity_key]))
        if not selected_client_ids:
            raise ValueError("FedCAB could not select any client.")
        message = "[FedCAB | Round {0}] Selected clients: {1}".format(current_round,
                                                                      selected_client_ids)
        log_message(logger, message, "INFO")
        message = "[FedCAB | Round {0}] Top scores: {1}".format(current_round,
                                                                {client_id: round(fedcab_scores[client_id], 6)
                                                                 for client_id in selected_client_ids[: min(10, len(selected_client_ids))]})
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
        selection_count = self.get_attribute("_selection_count")
        for client_id in selected_clients:
            selection_count[client_id] = selection_count.get(client_id, 0) + 1
        self._set_attribute("_selection_count", selection_count)
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
        data_privacy_approach = kwargs["data_privacy_approach"]
        logger = kwargs["logger"]
        capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
        task_assignment_capacities_list = [client_map[capacity_key] for _, client_map in candidate_clients.items()]
        self._set_attribute("_task_assignment_capacities_list", task_assignment_capacities_list)
        scaled_task_assignment_capacities_list = [sorted(set([capacity for capacity in capacities]))
                                                  for capacities in task_assignment_capacities_list]
        max_num_tasks = sum(max(capacities)
                            for capacities in scaled_task_assignment_capacities_list if capacities)
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
