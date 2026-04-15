from collections import Counter
from copy import deepcopy
from math import e, exp, log, sqrt
from numpy.random import default_rng
from statistics import mean, stdev


def estimate_makespan(X: list,
                      A: list,
                      G: list) -> float:
    # Initialize the estimated makespan of schedule X.
    M_X = 0
    for i, _ in enumerate(X):
        # Get the number of tasks assigned to candidate client i.
        x_i = X[i]
        x_i_index = list(A[i]).index(x_i)
        # Get the time spent by the candidate client i to complete x_i tasks.
        delta_i = G[i][x_i_index]
        # Update the makespan of the current candidate schedule.
        if M_X < delta_i:
            M_X = delta_i
    # Return the estimated makespan of schedule X.
    return M_X


def estimate_energy_consumption(X: list,
                                M_X: float,
                                A: list,
                                I: list,
                                B: list,
                                E: list) -> tuple:
    # Initialize the estimated energy consumption and remaining battery level of schedule X.
    E_X = []
    BL_X = []
    for i, _ in enumerate(X):
        # Get the number of tasks assigned to candidate client i.
        x_i = X[i]
        x_i_index = list(A[i]).index(x_i)
        # Get the energy consumed by the candidate client i to complete x_i tasks.
        if x_i == 0:
            # Client i was not selected, but has an energy consumption associated to its idle state.
            p_idle_i = I[i]
            epsilon_i = p_idle_i * M_X
        else:
            epsilon_i = E[i][x_i_index]
        # Append the energy consumed by the candidate client i to the list of energy consumptions of the current candidate schedule.
        E_X.append(epsilon_i)
    # Get the remaining battery level of all clients.
    for i, _ in enumerate(X):
        bl_i = 0 if E_X[i] >= B[i] else B[i] - E_X[i]
        BL_X.append(bl_i)
    # Calculate the total energy consumption of the current candidate schedule.
    E_X = sum(E_X)
    # Return the estimated energy consumption and remaining battery level of schedule X.
    return E_X, BL_X


def calculate_client_diversity_score(X: list,
                                     candidate_client_ids: list,
                                     current_round: int,
                                     q: int,
                                     candidate_clients_history_ids: dict,
                                     selected_clients_history_ids: dict) -> float:
    # Build a temporary copy of the selection history so we can evaluate X as the
    # current round without mutating the original state.
    selected_hist = deepcopy(selected_clients_history_ids or {})
    # Selected clients induced by schedule X in the current round.
    selected_client_ids = [candidate_client_ids[i] for i, x_i in enumerate(X) if x_i > 0]
    selected_hist[current_round] = selected_client_ids
    # History window: current round + previous q-1 rounds.
    first_round = max(1, current_round - q + 1)
    rounds_window = range(first_round, current_round + 1)
    # Count selections in the window.
    selection_counter = Counter()
    for r in rounds_window:
        S_r = selected_hist.get(r, [])
        selection_counter.update(S_r)
    # No selections at all.
    total_selected = sum(selection_counter.values())
    if total_selected == 0:
        return 0.0
    # Not enough candidate clients to define diversity.
    if len(candidate_client_ids) <= 1:
        return 0.0
    # Relative participation frequency p_{i,r}.
    P_sel = []
    for client_id in candidate_client_ids:
        p_i_r = selection_counter.get(client_id, 0) / total_selected
        if p_i_r > 0:
            P_sel.append(p_i_r)
    # Shannon Evenness Index.
    H = -sum(pi * log(pi, e) for pi in P_sel)
    H_max = log(len(candidate_client_ids), e)
    D = H / H_max if H_max > 0 else 0.0
    return D


def calculate_normalized_client_diversity_score_over_past_x_rounds(candidate_client_ids: list,
                                                                   candidate_clients_history_ids: dict,
                                                                   selected_clients_history_ids: dict,
                                                                   x: int) -> float:
    if not candidate_client_ids:
        return 0.0  # No candidate clients
    if len(candidate_client_ids) <= 1:
        return 0.0  # Not enough candidate clients to define diversity
    # Determine last round number from the available history.
    history_round_keys = set(candidate_clients_history_ids.keys()) | set(selected_clients_history_ids.keys())
    if not history_round_keys:
        return 0.0
    last_round = max(history_round_keys)
    # Get keys of past x rounds.
    past_rounds_keys = [r for r in range(last_round - x + 1, last_round + 1) if r > 0]
    # Count selections in the history window.
    selection_counter = Counter()
    for r in past_rounds_keys:
        if r not in selected_clients_history_ids:
            continue
        S_r = selected_clients_history_ids[r]
        selection_counter.update(S_r)
    # No selections at all.
    total_selected = sum(selection_counter.values())
    if total_selected == 0:
        return 0.0
    # Relative participation frequency p_i over the history window,
    # using the current candidate set as the support.
    P_sel = []
    for client_id in candidate_client_ids:
        p_i = selection_counter.get(client_id, 0) / total_selected
        if p_i > 0:
            P_sel.append(p_i)
    # Shannon Evenness Index.
    H = -sum(pi * log(pi, e) for pi in P_sel)
    H_max = log(len(candidate_client_ids), e)
    D = H / H_max if H_max > 0 else 0.0
    return D


def build_class_capacity_vectors_list(candidate_clients: dict,
                                      data_privacy_approach: str,
                                      current_phase: str) -> tuple:
    Y = []
    sorted_classes = []
    match data_privacy_approach:
        case "Non_Private":
            tasks_per_class_list = [client_map["client_tasks_per_class_{0}".format(current_phase)]
                                    for _, client_map in candidate_clients.items()]
            unique_classes = set()
            for d in tasks_per_class_list:
                unique_classes.update(int(k) for k in d.keys())
            sorted_classes = sorted(unique_classes)
            for d in tasks_per_class_list:
                gamma = []
                for k in sorted_classes:
                    gamma.append(d.get(str(k), 0))
                Y.append(gamma)
        case "Differentially_Private":
            # Get the class-index mapping (all clients have the same mapping, so we take from the first).
            _, first_client_map = next(iter(candidate_clients.items()))
            class_index_map = first_client_map["class_index_map"]
            idx_to_cls = {idx: cls for cls, idx in class_index_map.items()}
            sorted_classes = [idx_to_cls[idx] for idx in sorted(idx_to_cls.keys())]
            for _, client_map in candidate_clients.items():
                client_dp_histogram = client_map["client_dp_histogram_{0}".format(current_phase)]
                gamma = []
                for cls in sorted_classes:
                    idx = class_index_map[cls]
                    gamma.append(int(round(client_dp_histogram[idx])))
                Y.append(gamma)
    return Y, sorted_classes


def distribute_tasks_with_random_approach(X: list,
                                          Y: list) -> list:
    rng = default_rng()
    X_dist = []
    for i, x_i in enumerate(X):
        # Initialize with zeros for each class.
        x_dist_i = [0] * len(Y[i])
        if x_i <= 0:
            X_dist.append(x_dist_i)
            continue
        # Expand the available classes according to their counts.
        expanded_gamma_i = [k for k, count in enumerate(Y[i]) for _ in range(count)]
        if not expanded_gamma_i:
            X_dist.append(x_dist_i)
            continue
        # Allow duplication if local dataset is smaller than target.
        replace_mode = x_i > len(expanded_gamma_i)
        selected_classes = rng.choice(expanded_gamma_i, size=x_i, replace=replace_mode)
        class_counts = Counter(selected_classes)
        # Update x_dist_i with these counts.
        for k, count in class_counts.items():
            x_dist_i[k] = count
        # Enforce exact count.
        current_total = sum(x_dist_i)
        if current_total < x_i:
            available_classes = [k for k, c in enumerate(Y[i]) if c > 0] or [0]
            for _ in range(x_i - current_total):
                k = rng.choice(available_classes)
                x_dist_i[k] += 1
        elif current_total > x_i:
            all_assigned = [k for k, c in enumerate(x_dist_i) for _ in range(c)]
            remove_idx = rng.choice(all_assigned, size=current_total - x_i, replace=False)
            for k in remove_idx:
                x_dist_i[k] -= 1
        X_dist.append(x_dist_i)
    return X_dist


def distribute_tasks_with_locally_balanced_approach(X: list,
                                                    Y: list) -> list:
    rng = default_rng()
    X_dist = []
    for i, x_i in enumerate(X):
        # Initialize with zeros for each class.
        x_dist_i = [0] * len(Y[i])
        # If no tasks to distribute, skip this client.
        if x_i <= 0:
            X_dist.append(x_dist_i)
            continue
        # Get the available classes (non-zero counts) for this client.
        available_classes = [k for k, count in enumerate(Y[i]) if count > 0]
        num_classes = len(available_classes)
        if num_classes == 0:
            # No capacity, all tasks must be dropped.
            X_dist.append(x_dist_i)
            continue
        remaining_tasks = x_i
        # First pass: distribute evenly without exceeding local counts.
        for k in available_classes:
            # Calculate the number of tasks to assign to this class.
            tasks_for_class = min(remaining_tasks // num_classes, Y[i][k])
            x_dist_i[k] += tasks_for_class
            remaining_tasks -= tasks_for_class
        # Second pass: fill remaining tasks safely (with duplication if needed).
        class_idx = 0
        while remaining_tasks > 0:
            k = available_classes[class_idx % num_classes]
            x_dist_i[k] += 1
            remaining_tasks -= 1
            class_idx += 1
        # Enforce exact count.
        current_total = sum(x_dist_i)
        if current_total < x_i:
            for _ in range(x_i - current_total):
                k = rng.choice(available_classes)
                x_dist_i[k] += 1
        elif current_total > x_i:
            all_assigned = [k for k, c in enumerate(x_dist_i) for _ in range(c)]
            remove_idx = rng.choice(all_assigned, size=current_total - x_i, replace=False)
            for k in remove_idx:
                x_dist_i[k] -= 1
        X_dist.append(x_dist_i)
    return X_dist


def distribute_tasks_with_globally_balanced_approach(X: list,
                                                     Y: list) -> list:
    rng = default_rng()
    # Get the number of candidate clients.
    n = len(X)
    # Get the number of task classes from the first client
    # (all clients have the same length of class capacity vector).
    m = len(Y[0])
    # Get the number of tasks scheduled.
    T = sum(X)
    total_capacity_per_class = [sum(Y[i][k] for i in range(n)) for k in range(m)]
    total_capacity = sum(total_capacity_per_class)
    if total_capacity == 0:
        return [[0] * m for _ in range(n)]
    # Ideal class-level distribution.
    ideal_distribution_per_class = [int(T * total_capacity_per_class[k] / total_capacity)
                                    for k in range(m)]
    # Adjust rounding.
    allocated_tasks = sum(ideal_distribution_per_class)
    remaining_tasks = T - allocated_tasks
    # Distribute remaining tasks to the classes with the highest residual.
    residuals = [(T * total_capacity_per_class[k] / total_capacity) % 1 for k in range(m)]
    for k in sorted(range(m), key=lambda x: -residuals[x])[:remaining_tasks]:
        ideal_distribution_per_class[k] += 1
    # Allocate tasks globally based on ideal distribution (starting with the highest-capacity classes).
    X_dist = [[0] * m for _ in range(n)]
    remaining_X = deepcopy(X)
    remaining_Y = deepcopy(Y)
    # Global balanced allocation.
    for k in range(m):
        tasks_to_allocate = ideal_distribution_per_class[k]
        for i in range(n):
            if tasks_to_allocate <= 0:
                break
            if remaining_Y[i][k] > 0:
                allocated = min(tasks_to_allocate, remaining_X[i], remaining_Y[i][k])
                X_dist[i][k] += allocated
                remaining_X[i] -= allocated
                remaining_Y[i][k] -= allocated
                tasks_to_allocate -= allocated
    # Redistribute remaining tasks (duplicate if needed).
    remaining_tasks = sum(remaining_X)
    while remaining_tasks > 0:
        progress = False
        for i in range(n):
            if remaining_X[i] > 0:
                for k in range(m):
                    X_dist[i][k] += 1
                    remaining_X[i] -= 1
                    remaining_tasks -= 1
                    progress = True
                    if remaining_tasks == 0:
                        break
                if remaining_tasks == 0:
                    break
        if not progress:
            break
    # Enforce exact count per client.
    for i, x_i in enumerate(X):
        current_total = sum(X_dist[i])
        if current_total < x_i:
            available_classes = [k for k in range(m) if Y[i][k] > 0] or [0]
            for _ in range(x_i - current_total):
                k = rng.choice(available_classes)
                X_dist[i][k] += 1
        elif current_total > x_i:
            all_assigned = [k for k, c in enumerate(X_dist[i]) for _ in range(c)]
            remove_idx = rng.choice(all_assigned, size=current_total - x_i, replace=False)
            for k in remove_idx:
                X_dist[i][k] -= 1
    return X_dist


def organize_tasks_distribution(X_dist: list,
                                sorted_classes: list) -> list:
    X_dist_organized = []
    for gamma in X_dist:
        gamma_dict = {k: int(v) for k, v in zip(sorted_classes, gamma) if v != 0}
        X_dist_organized.append(gamma_dict)
    return X_dist_organized


def normalize_deviation_with_sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def adjust_learning_rates(X: list,
                          base_learning_rate: float,
                          min_learning_rate: float = 1e-5,
                          max_learning_rate: float = 1e-2) -> list:
    # Filter out zero tasks before calculating the statistics.
    X_filt = [int(X[i]) for i, _ in enumerate(X) if int(X[i]) > 0]
    # Calculate the average and standard deviation of number of tasks scheduled.
    avg = mean(X_filt)
    if len(X_filt) > 1:
        std_dev = stdev(X_filt)
    else:
        std_dev = 1
    # Calculate dynamic adjustment factor based on standard deviation and average.
    # Larger standard deviation -> larger adjustment factor.
    adjustment_factor = std_dev / avg if avg != 0 else 0
    # Initialize the list of adjusted learning rate per client.
    adjusted_learning_rates = []
    for i, _ in enumerate(X):
        # Get the number of tasks assigned to client i.
        x_i = int(X[i])
        if x_i == 0:
            # If the client i has zero tasks, skip the learning rate adjustment.
            adjusted_learning_rates.append(0)
        else:
            # Calculate how many standard deviations the client is away from the average.
            std_devs_away = abs(x_i - avg) / std_dev if std_dev != 0 else 0
            # Apply the sigmoid function to normalize the deviation.
            sigmoid_value = normalize_deviation_with_sigmoid(std_devs_away)
            if x_i > avg:
                # Clients with more tasks should have lower learning rate (subtract the scaled adjustment).
                scaled_adjustment = -(sigmoid_value - 0.5) * adjustment_factor
            else:
                # Clients with fewer tasks should have higher learning rate (add the scaled adjustment).
                scaled_adjustment = (sigmoid_value - 0.5) * adjustment_factor
            # Adjust the learning rate (clamp within safe range).
            learning_rate_i = base_learning_rate + base_learning_rate * scaled_adjustment
            learning_rate_i = max(min_learning_rate, min(max_learning_rate, learning_rate_i))
            # Append the adjusted learning rate of the client i into the list.
            adjusted_learning_rates.append(learning_rate_i)
    # Return the list of adjusted learning rate per client.
    return adjusted_learning_rates


def adjust_batch_sizes(X: list,
                       base_batch_size: int,
                       min_batch_size: int = 8,
                       max_batch_size: int = 128) -> list:
    # Filter out zero tasks before calculating the statistics.
    X_filt = [int(X[i]) for i in range(len(X)) if int(X[i]) > 0]
    # Calculate the average and standard deviation of number of tasks scheduled.
    avg = mean(X_filt)
    if len(X_filt) > 1:
        std_dev = stdev(X_filt)
    else:
        std_dev = 1
    # Calculate dynamic adjustment factor based on standard deviation and average.
    # Larger standard deviation -> larger adjustment factor.
    adjustment_factor = std_dev / avg if avg != 0 else 0
    # Initialize the list of adjusted batch_size per client.
    adjusted_batch_sizes = []
    for i, _ in enumerate(X):
        # Get the number of tasks assigned to client i.
        x_i = int(X[i])
        if x_i == 0:
            # If the client i has zero tasks, skip the batch_size adjustment.
            adjusted_batch_sizes.append(0)
        else:
            # Calculate how many standard deviations the client is away from the average.
            std_devs_away = abs(x_i - avg) / std_dev if std_dev != 0 else 0
            # Apply the sigmoid function to normalize the deviation.
            sigmoid_value = normalize_deviation_with_sigmoid(std_devs_away)
            if x_i > avg:
                # Clients with more tasks should have higher batch_size (subtract the scaled adjustment).
                scaled_adjustment = (sigmoid_value - 0.5) * adjustment_factor
            else:
                # Clients with fewer tasks should have lower batch_size (add the scaled adjustment).
                scaled_adjustment = -(sigmoid_value - 0.5) * adjustment_factor
            # Adjust the batch_size (clamp within safe range).
            batch_size_i = base_batch_size + int(round(base_batch_size * scaled_adjustment))
            batch_size_i = max(min_batch_size, min(max_batch_size, batch_size_i))
            # Append the adjusted batch_size of the client i into the list.
            adjusted_batch_sizes.append(batch_size_i)
    # Return the list of adjusted batch_size per client.
    return adjusted_batch_sizes


def adjust_num_epochs(X: list,
                      base_num_epochs: int) -> list:
    # Filter out zero tasks before calculating the statistics.
    X_filt = [int(X[i]) for i, _ in enumerate(X) if int(X[i]) > 0]
    # Calculate the average and standard deviation of number of tasks scheduled.
    avg = mean(X_filt)
    if len(X_filt) > 1:
        std_dev = stdev(X_filt)
    else:
        std_dev = 1
    # Calculate dynamic adjustment factor based on standard deviation and average.
    # Larger standard deviation -> larger adjustment factor.
    adjustment_factor = std_dev / avg if avg != 0 else 0
    # Initialize the list of adjusted number of epochs per client.
    adjusted_num_epochs = []
    for i, _ in enumerate(X):
        # Get the number of tasks assigned to client i.
        x_i = int(X[i])
        if x_i == 0:
            # If the client i has zero tasks, skip the epoch adjustment.
            adjusted_num_epochs.append(0)
        else:
            # Calculate how many standard deviations the client is away from the average.
            std_devs_away = abs(x_i - avg) / std_dev if std_dev != 0 else 0
            # Apply the sigmoid function to normalize the deviation.
            sigmoid_value = normalize_deviation_with_sigmoid(std_devs_away)
            # Compress the effect of the sigmoid output by adjusting the scaling factor.
            if x_i > avg:
                # Clients with more tasks should have fewer epochs (subtract the scaled adjustment).
                scaled_adjustment = -(sigmoid_value - 0.5) * adjustment_factor
            else:
                # Clients with fewer tasks should have more epochs (add the scaled adjustment).
                scaled_adjustment = (sigmoid_value - 0.5) * adjustment_factor
            # Adjust the number of epochs.
            num_epochs_i = base_num_epochs + int(round(base_num_epochs * scaled_adjustment))
            # Clamp the number of epochs to at least 1 and at most 5 times the base_num_epochs.
            max_num_epochs = 5 * base_num_epochs
            num_epochs_i = max(1, min(max_num_epochs, num_epochs_i))
            # Append the adjusted number of epochs of the client i into the list.
            adjusted_num_epochs.append(num_epochs_i)
    # Return the list of adjusted number of epochs per client.
    return adjusted_num_epochs


def calculate_class_distribution_score(X: list,
                                       X_dist: list,
                                       t: int,
                                       Y: list,
                                       beta: float = 0.5) -> tuple:
    # Get the number of task classes from the first client.
    m = len(Y[0])
    # Initialize the lists of available and assigned types of tasks across the clients.
    KC = [0] * m  # Available class counts.
    KS = [0] * m  # Assigned class counts.
    # Calculate KC (available class counts).
    for i in range(len(X)):
        for k, count in enumerate(Y[i]):
            KC[k] += count
    # Calculate KS (assigned class counts).
    for i in range(len(X_dist)):
        for k, count in enumerate(X_dist[i]):
            KS[k] += count
    # Calculate the class coverage score (KCov_X).
    KCov_X = (1 / m) * sum((KS[k] / KC[k]) if KC[k] != 0 else 0 for k in range(m))
    # Mean number of tasks per class.
    mean_tasks_per_class = t / m
    # Variance.
    variance = (1 / m) * sum((KS[k] - mean_tasks_per_class) ** 2 for k in range(m))
    # Standard deviation.
    std_dev = sqrt(variance)
    # Maximum possible standard deviation (normalization factor).
    max_std_dev = t / sqrt(m)
    # Normalized class standard deviation score (between 0 and 1).
    KStd_X = std_dev / max_std_dev if max_std_dev != 0 else 0
    # Calculate the class-distribution quality score (B_X).
    B_X = (beta * KCov_X) + ((1 - beta) * (1 - KStd_X))
    # Return the class-distribution quality score and intermediate scores.
    return B_X, KCov_X, KStd_X


def estimate_utility(X: list,
                     t: int,
                     phi_list: list,
                     psi_list: list,
                     alpha: float) -> float:
    # Initialize the utility score of schedule X.
    U_X = 0.0
    for i, _ in enumerate(X):
        # Get the number of tasks assigned to candidate client i.
        x_i = X[i]
        if x_i == 0:
            u_i = 0.0
        else:
            phi_i = phi_list[i]
            psi_i = psi_list[i]
            u_i = (x_i / t) * ((alpha * phi_i) + ((1 - alpha) * psi_i))
        # Update the utility score of the current candidate schedule.
        U_X += u_i
    # Return the utility score of schedule X.
    return U_X


def estimate_costs(n: int,
                   t: int,
                   A: list,
                   Y: list,
                   I: list,
                   B: list,
                   G: list,
                   E: list,
                   phi_list: list,
                   psi_list: list,
                   alpha: float,
                   X_init: list,
                   X_best: list,
                   X_rpr: list,
                   X_dist_approaches: dict,
                   candidate_client_ids: list,
                   current_round: int,
                   q: int,
                   candidate_clients_history_ids: dict,
                   selected_clients_history_ids: dict) -> dict:
    # Estimate the solution costs of the initial solution (X_init).
    X_init_costs = estimate_solution_costs(n, t, A, Y, I, B, G, E, phi_list, psi_list, alpha,
                                           X_init, X_dist_approaches["X_init"],
                                           candidate_client_ids, current_round, q,
                                           candidate_clients_history_ids, selected_clients_history_ids)
    # Estimate the solution costs of the best solution (X_best).
    X_best_costs = estimate_solution_costs(n, t, A, Y, I, B, G, E, phi_list, psi_list, alpha,
                                           X_best, X_dist_approaches["X_best"],
                                           candidate_client_ids, current_round, q,
                                           candidate_clients_history_ids, selected_clients_history_ids)
    # Estimate the solution costs of the repaired solution (X_rpr).
    X_rpr_costs = estimate_solution_costs(n, t, A, Y, I, B, G, E, phi_list, psi_list, alpha,
                                          X_rpr, X_dist_approaches["X_rpr"],
                                          candidate_client_ids, current_round, q,
                                          candidate_clients_history_ids, selected_clients_history_ids)
    # Set the dictionary of estimated costs.
    estimated_costs = {"X_init": X_init_costs, "X_best": X_best_costs, "X_rpr": X_rpr_costs}
    # Return the dictionary of estimated costs.
    return estimated_costs


def update_min_max_costs(sol_costs: dict,
                         min_max_costs: dict) -> dict:
    for k in sol_costs.keys():
        if "M_X" in sol_costs[k] and sol_costs[k]["M_X"] < min_max_costs["min_M_X"]:
            min_max_costs["min_M_X"] = sol_costs[k]["M_X"]
        if "M_X" in sol_costs[k] and sol_costs[k]["M_X"] > min_max_costs["max_M_X"]:
            min_max_costs["max_M_X"] = sol_costs[k]["M_X"]
        if "E_X" in sol_costs[k] and sol_costs[k]["E_X"] < min_max_costs["min_E_X"]:
            min_max_costs["min_E_X"] = sol_costs[k]["E_X"]
        if "E_X" in sol_costs[k] and sol_costs[k]["E_X"] > min_max_costs["max_E_X"]:
            min_max_costs["max_E_X"] = sol_costs[k]["E_X"]
    return min_max_costs


def calculate_percentage_change(old_value: int | float,
                                new_value: int | float) -> int | float:
    if old_value == 0:
        return 0
    percentage_change = ((new_value - old_value) / old_value) * 100
    return percentage_change


def normalize_cost(cost: float,
                   cost_min: float,
                   cost_max: float) -> float:
    if cost_min == cost_max:
        return 0.0
    # Avoid log(0).
    epsilon = 1e-12
    cost = max(cost, epsilon)
    cost_min = max(cost_min, epsilon)
    cost_max = max(cost_max, epsilon)
    cost_norm = (log(cost, e) - log(cost_min, e)) / (log(cost_max, e) - log(cost_min, e))
    return cost_norm


def normalize_costs(sol_costs: dict,
                    normalization_bounds: dict) -> dict:
    normalized_costs = deepcopy(sol_costs)
    for k in normalized_costs.keys():
        if "M_X" in normalized_costs[k]:
            M_X_norm = normalize_cost(normalized_costs[k]["M_X"], normalization_bounds["min_M_X"], normalization_bounds["max_M_X"])
            normalized_costs[k]["M_X"] = M_X_norm
        if "E_X" in normalized_costs[k]:
            E_X_norm = normalize_cost(normalized_costs[k]["E_X"], normalization_bounds["min_E_X"], normalization_bounds["max_E_X"])
            normalized_costs[k]["E_X"] = E_X_norm
    return normalized_costs


def estimate_solution_costs(n: int,
                            t: int,
                            A: list,
                            Y: list,
                            I: list,
                            B: list,
                            G: list,
                            E: list,
                            phi_list: list,
                            psi_list: list,
                            alpha: float,
                            X: list,
                            X_dist_approach: str,
                            candidate_client_ids: list,
                            current_round: int,
                            q: int,
                            candidate_clients_history_ids: dict,
                            selected_clients_history_ids: dict) -> dict:
    # Estimate the makespan.
    M_X = estimate_makespan(X, A, G)
    # Estimate the energy consumption and remaining battery levels.
    E_X, BL_X = estimate_energy_consumption(X, M_X, A, I, B, E)
    # Calculate the client diversity score.
    D_X = calculate_client_diversity_score(X, candidate_client_ids, current_round, q, candidate_clients_history_ids, selected_clients_history_ids)
    # Distribute the type of tasks scheduled per client.
    X_dist = []
    match X_dist_approach:
        case "random":
            X_dist = distribute_tasks_with_random_approach(X, Y)
        case "locally_balanced":
            X_dist = distribute_tasks_with_locally_balanced_approach(X, Y)
        case "globally_balanced":
            X_dist = distribute_tasks_with_globally_balanced_approach(X, Y)
    # Calculate the class-distribution quality score.
    B_X, KCov_X, KStd_X = calculate_class_distribution_score(X, X_dist, t, Y)
    # Estimate the utility score.
    U_X = estimate_utility(X, t, phi_list, psi_list, alpha)
    # Set the dictionary of estimated costs for the schedule X.
    X_costs = {"X": X,
               "X_dist": X_dist,
               "M_X": M_X,
               "E_X": E_X,
               "BL_X": BL_X,
               "D_X": D_X,
               "B_X": B_X,
               "KCov_X": KCov_X,
               "KStd_X": KStd_X,
               "U_X": U_X}
    # Return the dictionary of estimated costs for the schedule X.
    return X_costs
