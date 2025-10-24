from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from numpy import array
from numpy.random import default_rng, SeedSequence
from statistics import mean
from time import process_time

from metacs_fl.task_scheduler.ecmtc import ecmtc
from metacs_fl.task_scheduler.mec import mec
from metacs_fl.task_scheduler.random import random_selection
from metacs_fl.utils.client_selector_util import calculate_linear_interpolation_or_extrapolation, \
    calculate_quadratic_interpolation_or_extrapolation, select_all_available_clients, schedule_tasks_to_selected_clients
from metacs_fl.utils.logger_util import log_message
from metacs_fl.utils.task_scheduler_util import distribute_tasks_with_random_approach, \
    build_class_capacity_vectors_list, distribute_tasks_with_locally_balanced_approach, organize_tasks_distribution


class SBACPAD2024:

    def __init__(self,
                 client_selection_settings: dict,
                 seed: None | int | SeedSequence) -> None:
        self._client_selection_settings = client_selection_settings
        self._initial_solution_generation_history = {}
        self._internal_candidate_clients_history = {}
        self._internal_selected_clients_history = {}
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
    def _generate_cost_matrices(current_phase: str,
                                candidate_clients: dict,
                                selected_clients_metrics_history: dict,
                                history_checker: str) -> dict:
        # Get the necessary properties of the candidate clients.
        task_assignment_capacities_list = [client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                                           for _, client_map in candidate_clients.items()]
        # Initialize the cost matrices.
        time_costs = []
        energy_costs = []
        for i in range(0, len(candidate_clients)):
            time_costs_i = [0] * len(task_assignment_capacities_list[i])
            energy_costs_i = [0] * len(task_assignment_capacities_list[i])
            # Set the costs of zero tasks scheduled, if needed, allowing the data point (x=0, y=0) to be used
            # during the estimation of his costs for the unused task assignment capacities.
            if 0 not in task_assignment_capacities_list[i]:
                time_costs_i.insert(0, 0)
                energy_costs_i.insert(0, 0)
            time_costs.append(time_costs_i)
            energy_costs.append(energy_costs_i)
        # Retrieve the historical costs per number of tasks per client.
        candidate_clients_costs_hist = {}
        round_keys = []
        match history_checker:
            case "All_Previous_Rounds":
                round_keys = selected_clients_metrics_history.keys()
        for round_key in round_keys:
            if current_phase in selected_clients_metrics_history[round_key] and \
                "clients_metrics_dicts" in selected_clients_metrics_history[round_key][current_phase]:
                clients_metrics = selected_clients_metrics_history[round_key][current_phase]["clients_metrics_dicts"]
                for client_dict in clients_metrics:
                    for client_id, client_map in candidate_clients.items():
                        if client_id in client_dict:
                            phase_metrics_i = client_dict[client_id]
                            x_i = phase_metrics_i["num_examples"]
                            time_i = 0
                            energy_i = 0
                            if "{0}ing_time_in_seconds".format(current_phase) in phase_metrics_i:
                                time_i = phase_metrics_i["{0}ing_time_in_seconds".format(current_phase)]
                            if "{0}ing_energy_in_joules".format(current_phase) in phase_metrics_i:
                                energy_i = phase_metrics_i["{0}ing_energy_in_joules".format(current_phase)]
                            # Create the dictionary of costs for the client i.
                            client_costs = {"time_costs": [time_i], "energy_costs": [energy_i]}
                            if client_id not in candidate_clients_costs_hist:
                                candidate_clients_costs_hist.update({client_id: {x_i: client_costs}})
                            else:
                                if x_i in candidate_clients_costs_hist[client_id]:
                                    for cost_key, cost_value in client_costs.items():
                                        if cost_key not in candidate_clients_costs_hist[client_id][x_i]:
                                            candidate_clients_costs_hist[client_id][x_i].update({cost_key: cost_value})
                                        else:
                                            candidate_clients_costs_hist[client_id][x_i][cost_key].extend(cost_value)
                                else:
                                    candidate_clients_costs_hist[client_id].update({x_i: client_costs})
        # Calculate the averages of the historical costs per number of tasks per client.
        for client_id, _ in candidate_clients_costs_hist.items():
            client_costs_dict = candidate_clients_costs_hist[client_id]
            for x_i, _ in client_costs_dict.items():
                time_costs_x_i = client_costs_dict[x_i]["time_costs"]
                del client_costs_dict[x_i]["time_costs"]
                client_costs_dict[x_i].update({"time_costs_mean": mean(time_costs_x_i)})
                energy_costs_x_i = client_costs_dict[x_i]["energy_costs"]
                del client_costs_dict[x_i]["energy_costs"]
                client_costs_dict[x_i].update({"energy_costs_mean": mean(energy_costs_x_i)})
        # Fill the cost matrices with the known values (averages of the historical costs).
        i = 0
        for client_id, _ in candidate_clients.items():
            if client_id in candidate_clients_costs_hist:
                client_costs_dict = candidate_clients_costs_hist[client_id]
                task_assignment_capacities_i = list(task_assignment_capacities_list[i])
                time_costs_i = time_costs[i]
                energy_costs_i = energy_costs[i]
                for x_i, client_costs in client_costs_dict.items():
                    x_i_idx = abs(array(task_assignment_capacities_i) - x_i).argmin()
                    time_costs_mean = client_costs["time_costs_mean"]
                    energy_costs_mean = client_costs["energy_costs_mean"]
                    time_costs_i[x_i_idx] = time_costs_mean
                    energy_costs_i[x_i_idx] = energy_costs_mean
                time_costs[i] = time_costs_i
                energy_costs[i] = energy_costs_i
            i = i + 1
        # Fill the cost matrices with estimations via interpolation/extrapolation of the known costs
        # for the indices whose costs are still unknown.
        i = 0
        for client_id, _ in candidate_clients.items():
            task_assignment_capacities_i = list(task_assignment_capacities_list[i])
            time_costs_i = time_costs[i]
            energy_costs_i = energy_costs[i]
            # Fill the time cost matrix of the client i.
            unknown_time_costs_indices = [idx for idx in range(1, len(time_costs_i)) if time_costs_i[idx] == 0]
            known_time_costs_indices = [idx for idx in range(0, len(time_costs_i)) if idx not in unknown_time_costs_indices]
            for _, unknown_idx in enumerate(unknown_time_costs_indices):
                prev_known_indices = [known_idx for known_idx in known_time_costs_indices if
                                      known_idx < unknown_idx]
                next_known_indices = [known_idx for known_idx in known_time_costs_indices if
                                      known_idx > unknown_idx]
                x = task_assignment_capacities_i[unknown_idx]
                if len(prev_known_indices) + len(next_known_indices) == 2:
                    # Linear estimation.
                    x1_idx = None
                    x2_idx = None
                    if len(next_known_indices) == 0:
                        # Extrapolation.
                        x1_idx = prev_known_indices[-2]
                        x2_idx = prev_known_indices[-1]
                    else:
                        # Interpolation.
                        x1_idx = prev_known_indices[-1]
                        x2_idx = next_known_indices[0]
                    if x1_idx is not None and x2_idx is not None:
                        x1 = task_assignment_capacities_i[x1_idx]
                        x2 = task_assignment_capacities_i[x2_idx]
                        y1 = time_costs_i[x1_idx]
                        y2 = time_costs_i[x2_idx]
                        if x1 < x2:
                            time_costs_i[unknown_idx] = calculate_linear_interpolation_or_extrapolation(x1,
                                                                                                        x2,
                                                                                                        y1,
                                                                                                        y2,
                                                                                                        x)
                elif len(prev_known_indices) + len(next_known_indices) > 2:
                    # Quadratic estimation.
                    x1_idx = None
                    x2_idx = None
                    x3_idx = None
                    if len(next_known_indices) == 0:
                        # Extrapolation.
                        x1_idx = prev_known_indices[-3]
                        x2_idx = prev_known_indices[-2]
                        x3_idx = prev_known_indices[-1]
                    else:
                        # Interpolation.
                        if len(prev_known_indices) == 0:
                            x1_idx = next_known_indices[0]
                            x2_idx = next_known_indices[1]
                            x3_idx = next_known_indices[2]
                        elif len(prev_known_indices) == 1 and len(next_known_indices) >= 2:
                            x1_idx = prev_known_indices[-1]
                            x2_idx = next_known_indices[0]
                            x3_idx = next_known_indices[1]
                        elif len(prev_known_indices) >= 2 and len(next_known_indices) == 1:
                            x1_idx = prev_known_indices[-2]
                            x2_idx = prev_known_indices[-1]
                            x3_idx = next_known_indices[0]
                        elif len(prev_known_indices) >= 2 and len(next_known_indices) >= 2:
                            x1_idx = prev_known_indices[-1]
                            x2_idx = next_known_indices[0]
                            x3_idx = next_known_indices[1]
                    if x1_idx is not None and x2_idx is not None and x3_idx is not None:
                        x1 = task_assignment_capacities_i[x1_idx]
                        x2 = task_assignment_capacities_i[x2_idx]
                        x3 = task_assignment_capacities_i[x3_idx]
                        y1 = time_costs_i[x1_idx]
                        y2 = time_costs_i[x2_idx]
                        y3 = time_costs_i[x3_idx]
                        if x1 < x2 < x3:
                            time_costs_i[unknown_idx] = calculate_quadratic_interpolation_or_extrapolation(x1,
                                                                                                           x2,
                                                                                                           x3,
                                                                                                           y1,
                                                                                                           y2,
                                                                                                           y3,
                                                                                                           x)
            # Fill the energy cost matrix of the client i.
            unknown_energy_costs_indices = [idx for idx in range(1, len(energy_costs_i)) if energy_costs_i[idx] == 0]
            known_energy_costs_indices = [idx for idx in range(0, len(energy_costs_i)) if
                                          idx not in unknown_energy_costs_indices]
            for _, unknown_idx in enumerate(unknown_energy_costs_indices):
                prev_known_indices = [known_idx for known_idx in known_energy_costs_indices if
                                      known_idx < unknown_idx]
                next_known_indices = [known_idx for known_idx in known_energy_costs_indices if
                                      known_idx > unknown_idx]
                x = task_assignment_capacities_i[unknown_idx]
                if len(prev_known_indices) + len(next_known_indices) == 2:
                    # Linear estimation.
                    x1_idx = None
                    x2_idx = None
                    if len(next_known_indices) == 0:
                        # Extrapolation.
                        x1_idx = prev_known_indices[-2]
                        x2_idx = prev_known_indices[-1]
                    else:
                        # Interpolation.
                        x1_idx = prev_known_indices[-1]
                        x2_idx = next_known_indices[0]
                    if x1_idx is not None and x2_idx is not None:
                        x1 = task_assignment_capacities_i[x1_idx]
                        x2 = task_assignment_capacities_i[x2_idx]
                        y1 = energy_costs_i[x1_idx]
                        y2 = energy_costs_i[x2_idx]
                        if x1 < x2:
                            energy_costs_i[unknown_idx] = calculate_linear_interpolation_or_extrapolation(x1,
                                                                                                          x2,
                                                                                                          y1,
                                                                                                          y2,
                                                                                                          x)
                elif len(prev_known_indices) + len(next_known_indices) > 2:
                    # Quadratic estimation.
                    x1_idx = None
                    x2_idx = None
                    x3_idx = None
                    if len(next_known_indices) == 0:
                        # Extrapolation.
                        x1_idx = prev_known_indices[-3]
                        x2_idx = prev_known_indices[-2]
                        x3_idx = prev_known_indices[-1]
                    else:
                        # Interpolation.
                        if len(prev_known_indices) == 0:
                            x1_idx = next_known_indices[0]
                            x2_idx = next_known_indices[1]
                            x3_idx = next_known_indices[2]
                        elif len(prev_known_indices) == 1 and len(next_known_indices) >= 2:
                            x1_idx = prev_known_indices[-1]
                            x2_idx = next_known_indices[0]
                            x3_idx = next_known_indices[1]
                        elif len(prev_known_indices) >= 2 and len(next_known_indices) == 1:
                            x1_idx = prev_known_indices[-2]
                            x2_idx = prev_known_indices[-1]
                            x3_idx = next_known_indices[0]
                        elif len(prev_known_indices) >= 2 and len(next_known_indices) >= 2:
                            x1_idx = prev_known_indices[-1]
                            x2_idx = next_known_indices[0]
                            x3_idx = next_known_indices[1]
                    if x1_idx is not None and x2_idx is not None and x3_idx is not None:
                        x1 = task_assignment_capacities_i[x1_idx]
                        x2 = task_assignment_capacities_i[x2_idx]
                        x3 = task_assignment_capacities_i[x3_idx]
                        y1 = energy_costs_i[x1_idx]
                        y2 = energy_costs_i[x2_idx]
                        y3 = energy_costs_i[x3_idx]
                        if x1 < x2 < x3:
                            energy_costs_i[unknown_idx] = calculate_quadratic_interpolation_or_extrapolation(x1,
                                                                                                             x2,
                                                                                                             x3,
                                                                                                             y1,
                                                                                                             y2,
                                                                                                             y3,
                                                                                                             x)
            time_costs[i] = time_costs_i
            energy_costs[i] = energy_costs_i
            i = i + 1
        cost_matrices = {"time_costs": time_costs,
                         "energy_costs": energy_costs}
        return cost_matrices

    def _select_clients_task(self,
                             fl_round: int,
                             current_phase: str,
                             candidate_clients: dict,
                             num_tasks: int,
                             samples_per_task: int,
                             cost_matrices: dict,
                             time_limit: float,
                             data_privacy_approach: str,
                             logger: Logger) -> dict:
        # Start the clients' selection duration timer.
        selection_duration_start = process_time()
        # Initialize the set of selected clients.
        selected_clients = {}
        # Initialize the solution.
        X = []
        X_dist = []
        # Build the class capacity vectors list.
        class_capacity_vectors_list, sorted_classes = build_class_capacity_vectors_list(candidate_clients,
                                                                                        data_privacy_approach,
                                                                                        current_phase)
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        client_selector_phase = client_selection_settings["client_selector_{0}ing".format(current_phase)]
        client_selector_name = client_selector_phase["name"]
        # Log a 'selecting clients' message.
        message = "[SBAC-PAD_2024 | Round {0}] Selecting {1}ing clients for round {0} using '{2}'..." \
                  .format(fl_round, current_phase, client_selector_name)
        log_message(logger, message, "INFO")
        match client_selector_name:
            case "Random":
                # Select clients using the 'Random' algorithm.
                fraction_clients = client_selector_phase["fraction_clients_{0}ing".format(current_phase)]
                X = random_selection(current_phase,
                                     num_tasks,
                                     candidate_clients,
                                     fraction_clients)
                X_dist = distribute_tasks_with_random_approach(X, class_capacity_vectors_list)
            case "MEC":
                if fl_round == 1:
                    # Log a 'selecting all available clients' message.
                    message = "[SBAC-PAD_2024 | Round {0}] Selecting all available clients ({1}) for {2}ing (profiling round)..." \
                        .format(fl_round, len(candidate_clients), current_phase)
                    log_message(logger, message, "INFO")
                    # Schedule tasks to all available (candidate) clients, respecting assignment capacities.
                    selected_clients = select_all_available_clients(candidate_clients, current_phase)
                    selected_clients = schedule_tasks_to_selected_clients(num_tasks,
                                                                          selected_clients,
                                                                          current_phase,
                                                                          profiling_round=True,
                                                                          schedule_to_all_clients=True)
                    for client_id, client_info in selected_clients.items():
                        num_tasks_i = client_info.get("client_num_tasks_scheduled", 0)
                        client_info["client_num_samples_scheduled"] = num_tasks_i * samples_per_task
                else:
                    # Select clients using the 'MEC' algorithm.
                    task_assignment_capacities_list = [client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                                                       for _, client_map in candidate_clients.items()]
                    time_costs = cost_matrices["time_costs"]
                    energy_costs = cost_matrices["energy_costs"]
                    X, _, _ = mec(len(candidate_clients),
                                  num_tasks,
                                  task_assignment_capacities_list,
                                  time_costs,
                                  energy_costs)
                    X_dist = distribute_tasks_with_locally_balanced_approach(X, class_capacity_vectors_list)
            case "ECMTC":
                if fl_round == 1:
                    # Log a 'selecting all available clients' message.
                    message = "[SBAC-PAD_2024 | Round {0}] Selecting all available clients ({1}) for {2}ing (profiling round)..." \
                        .format(fl_round, len(candidate_clients), current_phase)
                    log_message(logger, message, "INFO")
                    # Schedule tasks to all available (candidate) clients, respecting assignment capacities.
                    selected_clients = select_all_available_clients(candidate_clients, current_phase)
                    selected_clients = schedule_tasks_to_selected_clients(num_tasks,
                                                                          selected_clients,
                                                                          current_phase,
                                                                          profiling_round=True,
                                                                          schedule_to_all_clients=True)
                    for client_id, client_info in selected_clients.items():
                        num_tasks_i = client_info.get("client_num_tasks_scheduled", 0)
                        client_info["client_num_samples_scheduled"] = num_tasks_i * samples_per_task
                else:
                    # Select clients using the 'ECMTC' algorithm.
                    task_assignment_capacities_list = [client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                                                       for _, client_map in candidate_clients.items()]
                    time_costs = cost_matrices["time_costs"]
                    energy_costs = cost_matrices["energy_costs"]
                    X, _, _ = ecmtc(len(candidate_clients),
                                    num_tasks,
                                    task_assignment_capacities_list,
                                    time_costs,
                                    energy_costs,
                                    time_limit)
                    X_dist = distribute_tasks_with_locally_balanced_approach(X, class_capacity_vectors_list)
        if not selected_clients:
            # Organize the tasks' distribution.
            X_dist = organize_tasks_distribution(X_dist, sorted_classes)
            for i, _ in enumerate(X):
                # Get the tasks per class assigned to client i.
                x_i = int(X[i])
                x_dist_i = X_dist[i]
                total_samples_i = x_i * samples_per_task
                x_dist_i_str = "|".join(["{0}={1}".format(k, v) for k, v in x_dist_i.items()])
                if x_i > 0:
                    # Update the set of selected clients.
                    client_id_str = "client_{0}".format(i)
                    client_proxy = candidate_clients[client_id_str]["client_proxy"]
                    capacity_key = "client_task_assignment_capacities_{0}".format(current_phase)
                    client_task_assignment_capacities_phase = candidate_clients[client_id_str][capacity_key]
                    client_max_task_capacity = max(client_task_assignment_capacities_phase)
                    client_info = {"client_proxy": client_proxy,
                                   capacity_key: client_task_assignment_capacities_phase,
                                   "client_max_task_capacity": client_max_task_capacity,
                                   "client_num_tasks_scheduled": x_i,
                                   "client_num_samples_scheduled": total_samples_i,
                                   "client_num_tasks_per_class_scheduled": x_dist_i_str}
                    selected_clients[client_id_str] = client_info
        # Log a 'number of tasks → number of samples' per-client message.
        for client_id, client_info in selected_clients.items():
            message = "{0}: {1} tasks → {2} samples".format(client_id,
                                                            client_info['client_num_tasks_scheduled'],
                                                            client_info['client_num_samples_scheduled'])
            log_message(logger, message, "INFO")
        # Get the clients' selection duration.
        selection_duration_in_seconds = process_time() - selection_duration_start
        # Log a 'clients' selection duration' message.
        message = "[SBAC-PAD_2024 | Round {0}] The client selection procedure took {1} seconds!" \
                  .format(fl_round, round(selection_duration_in_seconds, 2))
        log_message(logger, message, "INFO")
        # Return the set of selected clients.
        selected_clients_future_result = {fl_round: {"selected_clients": selected_clients,
                                                     "selection_duration_in_seconds": selection_duration_in_seconds}}
        return selected_clients_future_result

    def _select_clients_async(self,
                              rounds_to_select_clients: list,
                              current_phase: str,
                              candidate_clients: dict,
                              num_tasks: int,
                              samples_per_task: int,
                              selected_clients_metrics_history: dict,
                              time_limit: float,
                              data_privacy_approach: str,
                              logger: Logger) -> list:
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        # Generate the cost matrices.
        history_checker = client_selection_settings["history_checker"]
        cost_matrices = self._generate_cost_matrices(current_phase,
                                                     candidate_clients,
                                                     selected_clients_metrics_history,
                                                     history_checker)
        # Initialize the list of selected clients' Future objects.
        selected_clients_futures = []
        target = self._select_clients_task
        with ThreadPoolExecutor(max_workers=len(rounds_to_select_clients)) as executor:
            for round_to_select_clients in rounds_to_select_clients:
                args = (round_to_select_clients,
                        current_phase,
                        candidate_clients,
                        num_tasks,
                        samples_per_task,
                        cost_matrices,
                        time_limit,
                        data_privacy_approach,
                        logger)
                selected_clients_future = executor.submit(target, *args)
                selected_clients_futures.append(selected_clients_future)
        # Return the list of selected clients' Future objects.
        return selected_clients_futures

    def run_client_selection_procedure(self,
                                       **kwargs) -> list:
        # Get the necessary parameters.
        current_round = kwargs["current_round"]
        current_phase = kwargs["current_phase"]
        candidate_clients = kwargs["candidate_clients"]
        num_rounds = kwargs["num_rounds"]
        num_tasks = kwargs["num_tasks"]
        samples_per_task = kwargs["samples_per_task"]
        selected_clients_metrics_history = kwargs["selected_clients_metrics_history"]
        time_limit = kwargs["time_limit"]
        data_privacy_approach = kwargs["data_privacy_approach"]
        logger = kwargs["logger"]
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        select_clients_for_immediate_next_round_while_executing_current_phase \
            = client_selection_settings["select_clients_for_immediate_next_round_while_executing_current_{0}ing".format(current_phase)]
        # Get the necessary properties of the candidate clients.
        task_assignment_capacities_list = [client_map["client_task_assignment_capacities_{0}".format(current_phase)]
                                           for _, client_map in candidate_clients.items()]
        scaled_task_assignment_capacities_list = [sorted(set(capacities))
                                                  for capacities in task_assignment_capacities_list]
        # Calculate the maximum number of tasks that can be scheduled.
        max_num_tasks = sum(max(capacities) for capacities in scaled_task_assignment_capacities_list)
        # Convert num_tasks → task count (fraction adjustment).
        if isinstance(num_tasks, float):
            # Always treat as fraction of total capacity.
            num_tasks = int(num_tasks * max_num_tasks)
        # Clamp to valid range.
        num_tasks = max(1, min(num_tasks, max_num_tasks))
        # Set the list of rounds to select clients.
        rounds_to_select_clients = []
        if not select_clients_for_immediate_next_round_while_executing_current_phase or current_round == 1:
            # Select clients only for the current round.
            rounds_to_select_clients = [current_round]
        if select_clients_for_immediate_next_round_while_executing_current_phase and current_round > 1:
            if current_round == 2:
                # Select clients for the current round and for its immediate next round, if applicable.
                rounds_to_select_clients = [current_round]
                immediate_next_round = current_round + 1
                if immediate_next_round <= num_rounds:
                    rounds_to_select_clients.append(immediate_next_round)
            else:
                # Select clients only for the immediate next round, if applicable.
                rounds_to_select_clients = []
                immediate_next_round = current_round + 1
                if immediate_next_round <= num_rounds:
                    rounds_to_select_clients.append(immediate_next_round)
        # Initialize the list of selected clients' Future objects.
        selected_clients_futures = []
        # If the list of rounds to select clients is not empty...
        if rounds_to_select_clients:
            # Select clients asynchronously, considering the list of rounds to select clients.
            selected_clients_futures = self._select_clients_async(rounds_to_select_clients,
                                                                  current_phase,
                                                                  candidate_clients,
                                                                  num_tasks,
                                                                  samples_per_task,
                                                                  selected_clients_metrics_history,
                                                                  time_limit,
                                                                  data_privacy_approach,
                                                                  logger)
        # Return the list of selected clients' Future objects.
        return selected_clients_futures
