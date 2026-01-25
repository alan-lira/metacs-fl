from copy import deepcopy
from logging import Logger
from math import log
from numpy.random import default_rng, SeedSequence

from metacs_fl.task_scheduler.random import random_selection
from metacs_fl.utils.client_selector_util import get_all_possible_sums, take_closest
from metacs_fl.utils.logger_util import log_message
from metacs_fl.utils.system_modeler_util import calculate_computation_time, calculate_download_time, calculate_upload_time
from metacs_fl.utils.task_scheduler_util import build_class_capacity_vectors_list, \
    distribute_tasks_with_random_approach, organize_tasks_distribution


class Oort:

    def __init__(self,
                 client_selection_settings: dict,
                 seed: None | int | SeedSequence) -> None:
        self._client_selection_settings = client_selection_settings
        self._initial_solution_generation_history = {}
        self._internal_candidate_clients_history = {}
        self._internal_selected_clients_history = {}
        self._task_assignment_capacities_list = []
        self._oort_last_selected_round = {}
        self._oort_round_util_hist = {}
        self._oort_time_limit = None
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
    def _oort_statistical_utility(candidate_clients: dict,
                                  metrics_history: dict,
                                  phase: str,
                                  stat_window: int = 5,
                                  default: float = 1.0) -> list:
        utils = {cid: [] for cid in candidate_clients}
        last_round = max(metrics_history.keys()) if metrics_history else 0
        rounds = range(max(1, last_round - stat_window + 1), last_round + 1)
        for r in rounds:
            if r not in metrics_history or phase not in metrics_history[r]:
                continue
            for d in metrics_history[r][phase].get("clients_metrics_dicts", []):
                cid = next(iter(d))
                if cid not in utils:
                    continue
                m = d[cid]
                loss = None
                for k, v in m.items():
                    if "loss" in k:
                        loss = v
                        break
                n = m.get("num_examples", 0)
                if loss is not None and n > 0:
                    utils[cid].append((loss, n))
        stat_utils = []
        for cid in candidate_clients:
            vals = utils[cid]
            if vals:
                num = sum((l ** 2) * n for l, n in vals)
                den = sum(n for _, n in vals)
                stat_utils.append(den * (num / max(1e-9, den)) ** 0.5)
            else:
                stat_utils.append(default)
        mx, mn = max(stat_utils), min(stat_utils)
        rng = max(1e-9, mx - mn)
        stat_utils = [(u - mn) / rng for u in stat_utils]
        return stat_utils

    @staticmethod
    def _oort_ucb_bonus(current_round: int,
                        last_round: int,
                        c: float=0.1) -> float:
        if last_round <= 0:
            return 1.0
        return (c * log(max(2, current_round)) / last_round) ** 0.5

    @staticmethod
    def _oort_client_utility(U_i: float,
                             t_i: float,
                             T: float,
                             alpha: float,
                             ucb: float) -> float:
        penalty = 1.0
        if t_i > T:
            penalty = (T / max(1e-6, t_i)) ** alpha
        return (U_i * penalty) + ucb

    @staticmethod
    def _generate_cost_matrices(current_phase: str,
                                candidate_clients: dict,
                                selected_clients_metrics_history: dict,
                                clients_profiles: dict) -> dict:
        time_costs = []
        for client_id, client_map in candidate_clients.items():
            task_assignment_capacities_i = client_map[f"client_task_assignment_capacities_{current_phase}"]
            bw_down = client_map["client_current_download_bandwidth_in_bytes_per_second"]
            bw_up = client_map["client_current_upload_bandwidth_in_bytes_per_second"]
            latency = client_map["client_current_latency_in_milliseconds"]
            latest_metrics = {}
            for round_key in reversed(selected_clients_metrics_history):
                if current_phase in selected_clients_metrics_history[round_key]:
                    for d in selected_clients_metrics_history[round_key][current_phase].get("clients_metrics_dicts", []):
                        if client_id in d:
                            latest_metrics = d[client_id]
                            break
            if not latest_metrics:
                latest_metrics = clients_profiles[client_id][current_phase]
            m = deepcopy(latest_metrics)
            m["bw_down_i"] = bw_down
            m["bw_up_i"] = bw_up
            m["rtt_down_i"] = latency
            m["rtt_up_i"] = latency
            d_time, _ = calculate_download_time(m, current_phase)
            u_time, _ = calculate_upload_time(m, current_phase)
            time_costs_i = []
            for ac in task_assignment_capacities_i:
                m[f"ds_{current_phase}_i"] = ac
                comp_time = calculate_computation_time(m, current_phase)
                time_costs_i.append(d_time + comp_time + u_time)
            time_costs.append(time_costs_i)
        return {"time_costs": time_costs}

    def weighted_sample(self,
                        utilities: dict,
                        cands: list,
                        k: int) -> list:
        if not cands or k <= 0:
            return []
        ws = [max(0.0, utilities[c]) for c in cands]
        s = sum(ws)
        if s <= 0:
            return list(self._rng.choice(cands, size=min(k, len(cands)), replace=False))
        ps = [w / s for w in ws]
        return list(self._rng.choice(cands, size=min(k, len(cands)), replace=False, p=ps))

    @staticmethod
    def fastest_positive_time(time_costs: list,
                              scaled_caps_list: list,
                              i: int) -> float:
        pts = [t for k, t in enumerate(time_costs[i]) if scaled_caps_list[i][k] > 0]
        return min(pts) if pts else time_costs[i][-1]

    @staticmethod
    def max_capacity(scaled_caps_list: list,
                     indices: list) -> int:
        return sum(max(scaled_caps_list[i]) for i in indices)

    def _select_clients(self,
                        current_round: int,
                        current_phase: str,
                        candidate_clients: dict,
                        num_tasks: int,
                        samples_per_task: int,
                        selected_clients_metrics_history: dict,
                        clients_profiles: dict,
                        time_limit: float,
                        data_privacy_approach: str,
                        logger: Logger) -> dict:
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        client_selector_phase = client_selection_settings["client_selector_{0}ing".format(current_phase)]
        client_selector_name = client_selector_phase["name"]
        # Generate the cost matrices.
        cost_matrices = self._generate_cost_matrices(current_phase,
                                                     candidate_clients,
                                                     selected_clients_metrics_history,
                                                     clients_profiles)
        # Get the necessary properties of the candidate clients.
        task_assignment_capacities_list = self.get_attribute("_task_assignment_capacities_list")
        scaled_caps_list = self.get_attribute("_scaled_caps_list")
        # Initialize the solution.
        X = []
        X_dist = []
        # Build the class capacity vectors list.
        class_capacity_vectors_list, sorted_classes = build_class_capacity_vectors_list(candidate_clients,
                                                                                        data_privacy_approach,
                                                                                        current_phase)
        # Log a 'selecting clients' message.
        message = "[Oort | Round {0}] Selecting {1}ing clients for round {0} using '{2}'..." \
                  .format(current_round, current_phase, client_selector_name)
        log_message(logger, message, "INFO")
        match client_selector_name:
            case "Random":
                # Select clients using the 'Random' algorithm.
                fraction_clients = client_selector_phase["fraction_clients_{0}ing".format(current_phase)]
                X = random_selection(current_phase,
                                     num_tasks,
                                     candidate_clients,
                                     fraction_clients)
                # Scale the assigned tasks back to sample-level counts (since distribution is based on samples per class).
                X_scaled = [x_i * samples_per_task for x_i in X]
                X_dist = distribute_tasks_with_random_approach(X_scaled, class_capacity_vectors_list)
            case "Oort":
                # Select clients using the 'OORT' algorithm.
                time_costs = cost_matrices["time_costs"]
                oort_mode = client_selector_phase["oort_mode"]
                alpha = client_selector_phase["alpha"]
                explore_frac = client_selector_phase["explore_frac"]
                pacer_step = client_selector_phase["pacer_step"]
                pacer_delta = client_selector_phase["pacer_delta"]
                # Initialize persistent Oort time limit (pacer state)
                if self._oort_time_limit is None:
                    self._oort_time_limit = time_limit
                X_scaled = []
                match oort_mode:
                    case "binary":
                        stat_utils = self._oort_statistical_utility(candidate_clients,
                                                                    selected_clients_metrics_history,
                                                                    current_phase)
                        utilities = {}
                        explored, unexplored = [], []
                        client_ids = list(candidate_clients.keys())
                        for i, cid in enumerate(client_ids):
                            last = self._oort_last_selected_round.get(cid, 0)
                            ucb = self._oort_ucb_bonus(current_round, max(1, current_round - last))
                            positive_times = [t for k, t in enumerate(time_costs[i]) if scaled_caps_list[i][k] > 0]
                            t_i = min(positive_times) if positive_times else time_costs[i][-1]
                            u = self._oort_client_utility(stat_utils[i], t_i, self._oort_time_limit, alpha, ucb)
                            utilities[cid] = u
                            if last > 0:
                                explored.append(cid)
                            else:
                                unexplored.append(cid)
                        explored_sorted = sorted(explored, key=lambda c: utilities[c], reverse=True)
                        cutoff = max(1, int(0.95 * len(explored_sorted)))
                        pool = explored_sorted[:cutoff]
                        K = len(candidate_clients)
                        exploit_k = int((1 - explore_frac) * K)
                        exploit_k = min(exploit_k, len(pool))
                        explore_k = K - exploit_k
                        exploit_sel = self.weighted_sample(utilities, pool, exploit_k)
                        unexplored_sorted = sorted(unexplored,
                                                   key=lambda c: self.fastest_positive_time(time_costs, scaled_caps_list, client_ids.index(c)))
                        explore_sel = unexplored_sorted[:explore_k]
                        selected = exploit_sel + explore_sel
                        selected_idx = [client_ids.index(cid) for cid in selected]
                        if self.max_capacity(scaled_caps_list, selected_idx) < num_tasks:
                            remaining_idx = [i for i in range(len(client_ids)) if i not in selected_idx]
                            remaining_sorted = sorted(remaining_idx,
                                                      key=lambda i: utilities[client_ids[i]],
                                                      reverse=True)
                            for i in remaining_sorted:
                                selected_idx.append(i)
                                selected.append(client_ids[i])
                                if self.max_capacity(scaled_caps_list, selected_idx) >= num_tasks:
                                    break
                        for cid in selected:
                            self._oort_last_selected_round[cid] = current_round
                        round_util = sum(utilities[c] for c in selected)
                        self._oort_round_util_hist[current_round] = round_util
                        if len(self._oort_round_util_hist) >= 2 * pacer_step:
                            recent_rounds = sorted(self._oort_round_util_hist.keys())[-2 * pacer_step:]
                            prev = sum(self._oort_round_util_hist[r] for r in recent_rounds[:pacer_step])
                            curr = sum(self._oort_round_util_hist[r] for r in recent_rounds[pacer_step:])
                            if curr < prev:
                                self._oort_time_limit *= (1 + pacer_delta)
                        dp = {0: (0.0, {})}
                        for idx in selected_idx:
                            new_dp = {}
                            for t_prev, (time_prev, assign_prev) in dp.items():
                                for k, cap in enumerate(scaled_caps_list[idx]):
                                    t_new = t_prev + cap
                                    if t_new > num_tasks:
                                        continue
                                    time_new = max(time_prev, time_costs[idx][k])
                                    if (t_new not in new_dp) or (time_new < new_dp[t_new][0]):
                                        new_assign = assign_prev.copy()
                                        new_assign[idx] = cap
                                        new_dp[t_new] = (time_new, new_assign)
                            dp = new_dp
                        if not dp:
                            idx = selected_idx[0]
                            cap = min(scaled_caps_list[idx])
                            X_scaled = [0] * len(task_assignment_capacities_list)
                            X_scaled[idx] = cap
                        else:
                            best_t = num_tasks if num_tasks in dp else max(dp.keys())
                            best_assign = dp[best_t][1]
                            X_scaled = [0] * len(task_assignment_capacities_list)
                            for idx, cap_scaled in best_assign.items():
                                X_scaled[idx] = cap_scaled
                X = [x_scaled // samples_per_task for x_scaled in X_scaled]
                X_dist = distribute_tasks_with_random_approach(X_scaled, class_capacity_vectors_list)
        # Organize the tasks' distribution.
        X_dist = organize_tasks_distribution(X_dist, sorted_classes)
        # Initialize the set of selected clients.
        selected_clients = {}
        for i, _ in enumerate(X):
            # Get the number of tasks assigned to client i (in tasks/minibatches).
            x_i = int(X[i])
            # Get the tasks-per-class distribution (already in samples).
            x_dist_i = X_dist[i]
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
                selected_clients.update({client_id_str: client_info})
        # Return the set of selected clients.
        return selected_clients

    def run_client_selection_procedure(self,
                                       **kwargs) -> dict:
        # Get the necessary parameters.
        current_round = kwargs["current_round"]
        current_phase = kwargs["current_phase"]
        candidate_clients = kwargs["candidate_clients"]
        num_tasks = kwargs["num_tasks"]
        samples_per_task = kwargs["samples_per_task"]
        selected_clients_metrics_history = kwargs["selected_clients_metrics_history"]
        clients_profiles = kwargs["clients_profiles"]
        time_limit = kwargs["time_limit"]
        data_privacy_approach = kwargs["data_privacy_approach"]
        logger = kwargs["logger"]
        # Get the necessary properties of the candidate clients.
        task_assignment_capacities_list = [client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                                           for _, client_map in candidate_clients.items()]
        self._set_attribute("_task_assignment_capacities_list", task_assignment_capacities_list)
        scaled_task_assignment_capacities_list = [sorted(set(c * samples_per_task for c in capacities))
                                                  for capacities in task_assignment_capacities_list]
        self._set_attribute("_scaled_caps_list", scaled_task_assignment_capacities_list)
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
        # Select clients.
        selected_clients = self._select_clients(current_round,
                                                current_phase,
                                                candidate_clients,
                                                num_tasks,
                                                samples_per_task,
                                                selected_clients_metrics_history,
                                                clients_profiles,
                                                time_limit,
                                                data_privacy_approach,
                                                logger)
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
