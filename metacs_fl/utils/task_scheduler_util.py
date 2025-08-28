from collections import Counter
from copy import deepcopy
from math import e, exp, log, sqrt
from numpy import ndarray
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
    B_X = []
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
        B_X.append(bl_i)
    # Calculate the total energy consumption of the current candidate schedule.
    E_X = sum(E_X)
    # Return the estimated energy consumption and remaining battery level of schedule X.
    return E_X, B_X


def calculate_client_diversity_score(X: list,
                                     n: int) -> float:
    P = []
    for i, _ in enumerate(X):
        s_i = 1 if X[i] > 0 else 0
        p_i = s_i / n
        P.append(p_i)
    # Shannon Evenness Index.
    H = - sum(pi * log(pi, e) for pi in P if pi > 0)
    H_max = log(n, e)
    D = H / H_max if H_max > 0 else 0
    return D


def calculate_normalized_client_diversity_score_over_past_x_rounds(candidate_clients_history_ids: dict,
                                                                   selected_clients_history_ids: dict,
                                                                   x: int) -> float:
    if not selected_clients_history_ids:
        return 0.0  # No data
    # Determine last round number.
    last_round = max(selected_clients_history_ids.keys())
    # Get keys of past x rounds.
    past_rounds_keys = [r for r in range(last_round - x + 1, last_round + 1) if r > 0]
    candidate_counter = Counter()
    selection_counter = Counter()
    for r in past_rounds_keys:
        if r not in selected_clients_history_ids or r not in candidate_clients_history_ids:
            continue  # skip missing rounds safely
        C_r = candidate_clients_history_ids[r]
        S_r = selected_clients_history_ids[r]
        candidate_counter.update(C_r)
        selection_counter.update(S_r)
    # Build normalized selection frequencies.
    relative_freqs = []
    for client in candidate_counter:
        candidate_count = candidate_counter[client]
        selection_count = selection_counter.get(client, 0)
        # Avoid division by zero (shouldn't happen but safe).
        rel_freq = selection_count / candidate_count if candidate_count > 0 else 0
        relative_freqs.append(rel_freq)
    if len(relative_freqs) <= 1:
        return 0.0  # Not enough diversity to compute.
    # Normalize relative frequencies to sum to 1.
    total_rel_freq = sum(relative_freqs)
    if total_rel_freq == 0:
        return 0.0  # No selections at all.
    P_sel = [rf / total_rel_freq for rf in relative_freqs]
    # Shannon Evenness Index.
    H = - sum(pi * log(pi, e) for pi in P_sel if pi > 0)
    H_max = log(len(P_sel), e)
    D = H / H_max if H_max > 0 else 0
    return D


def build_class_capacity_vectors_list(tasks_per_class_list: list) -> tuple:
    unique_classes = set()
    for d in tasks_per_class_list:
        unique_classes.update(int(k) for k in d.keys())
    sorted_classes = sorted(unique_classes)
    Y = []
    for d in tasks_per_class_list:
        gamma = []
        for k in sorted_classes:
            gamma.append(d.get(str(k), 0))
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
        # If not enough available tasks, limit to what's possible.
        sample_size = min(x_i, len(expanded_gamma_i))
        # Randomly select classes without replacement.
        selected_classes = rng.choice(expanded_gamma_i, size=sample_size, replace=False)
        # Count how many times each class was selected.
        class_counts = Counter(selected_classes)
        # Update x_dist_i with these counts.
        for k, count in class_counts.items():
            x_dist_i[k] = count
        X_dist.append(x_dist_i)
    return X_dist


def distribute_tasks_with_locally_balanced_approach(X: list,
                                                    Y: list) -> list:
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
        # Distribute tasks as evenly as possible while respecting capacity.
        remaining_tasks = x_i
        for k in available_classes:
            # Calculate the number of tasks to assign to this class.
            tasks_for_class = min(remaining_tasks // num_classes, Y[i][k])
            x_dist_i[k] += tasks_for_class
            remaining_tasks -= tasks_for_class
        # Distribute any remaining tasks in a balanced way (round-robin) respecting capacity.
        class_idx = 0
        while remaining_tasks > 0:
            k = available_classes[class_idx % num_classes]
            # Check if capacity is not exceeded.
            if x_dist_i[k] < Y[i][k]:
                x_dist_i[k] += 1
                remaining_tasks -= 1
            class_idx += 1
        # Append the distribution of tasks of this client.
        X_dist.append(x_dist_i)
    return X_dist


def distribute_tasks_with_globally_balanced_approach(X: list,
                                                     Y: list) -> list:
    # Get the number of candidate clients.
    n = len(X)
    # Get the number of task classes from the first client
    # (all clients have the same length of class capacity vector).
    m = len(Y[0])
    # Get the number of tasks scheduled.
    T = sum(X)
    # Compute the total capacity per class globally (sum across all clients).
    total_capacity_per_class = [0] * m
    for i in range(n):
        for k in range(m):
            total_capacity_per_class[k] += Y[i][k]
    # Calculate the global ideal distribution of tasks per class.
    total_capacity = sum(total_capacity_per_class)
    ideal_distribution_per_class = [int(T * total_capacity_per_class[k] / total_capacity)
                                    for k in range(m)]
    # Adjust for rounding errors.
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
    for k in range(m):
        tasks_to_allocate = ideal_distribution_per_class[k]
        for i in range(n):
            if tasks_to_allocate <= 0:
                break
            if remaining_Y[i][k] > 0:
                allocated_tasks = min(tasks_to_allocate, remaining_X[i], remaining_Y[i][k])
                X_dist[i][k] += allocated_tasks
                remaining_X[i] -= allocated_tasks
                remaining_Y[i][k] -= allocated_tasks
                tasks_to_allocate -= allocated_tasks
    # Distribute any remaining tasks in a balanced way.
    remaining_tasks = sum(remaining_X)
    while remaining_tasks > 0:
        for i in range(n):
            if remaining_X[i] > 0:
                for k in range(m):
                    if remaining_Y[i][k] > 0 and remaining_X[i] > 0:
                        X_dist[i][k] += 1
                        remaining_X[i] -= 1
                        remaining_Y[i][k] -= 1
                        remaining_tasks -= 1
                        if remaining_tasks == 0:
                            break
                if remaining_tasks == 0:
                    break
    # Final check to verify if all tasks were distributed.
    if sum(remaining_X) != 0:
        print("Error: Not all tasks have been allocated. Remaining tasks: {0}".format(sum(remaining_X)))
    return X_dist


def organize_tasks_distribution(X_dist: list,
                                sorted_classes: list) -> list:
    X_dist_organized = []
    for gamma in X_dist:
        gamma_dict = {k: v for k, v in zip(sorted_classes, gamma) if v != 0}
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


def calculate_class_coverage_and_standard_deviation_scores(X: list,
                                                           X_dist: list,
                                                           t: int,
                                                           Y: list) -> tuple:
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
    # Return the class coverage and standard deviation scores.
    return KCov_X, KStd_X


def estimate_costs(n: int,
                   t: int,
                   A: list,
                   Y: list,
                   I: list,
                   B: list,
                   G: list,
                   E: list,
                   X_init: list,
                   X_best: list,
                   X_rpr: list,
                   X_dist_approaches: dict) -> dict:
    # Estimate the solution costs of the initial solution (X_init).
    X_init_costs = estimate_solution_costs(n, t, A, Y, I, B, G, E, X_init, X_dist_approaches["X_init"])
    # Estimate the solution costs of the best solution (X_best).
    X_best_costs = estimate_solution_costs(n, t, A, Y, I, B, G, E, X_best, X_dist_approaches["X_best"])
    # Estimate the solution costs of the repaired solution (X_rpr).
    X_rpr_costs = estimate_solution_costs(n, t, A, Y, I, B, G, E, X_rpr, X_dist_approaches["X_rpr"])
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
    cost_norm = (log(cost, e) - log(cost_min, e)) / (log(cost_max, e) - log(cost_min, e))
    return cost_norm


def normalize_costs(sol_costs: dict,
                    min_max_costs: dict) -> dict:
    normalized_costs = deepcopy(sol_costs)
    for k in normalized_costs.keys():
        if "M_X" in normalized_costs[k]:
            M_X_norm = normalize_cost(normalized_costs[k]["M_X"], min_max_costs["min_M_X"], min_max_costs["max_M_X"])
            normalized_costs[k]["M_X"] = M_X_norm
        if "E_X" in normalized_costs[k]:
            E_X_norm = normalize_cost(normalized_costs[k]["E_X"], min_max_costs["min_E_X"], min_max_costs["max_E_X"])
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
                            X: list,
                            X_dist_approach: str) -> dict:
    # Estimate the makespan.
    M_X = estimate_makespan(X, A, G)
    # Estimate the energy consumption and remaining battery levels.
    E_X, B_X = estimate_energy_consumption(X, M_X, A, I, B, E)
    # Calculate the client diversity score.
    D_X = calculate_client_diversity_score(X, n)
    # Distribute the type of tasks scheduled per client.
    X_dist = []
    match X_dist_approach:
        case "random":
            X_dist = distribute_tasks_with_random_approach(X, Y)
        case "locally_balanced":
            X_dist = distribute_tasks_with_locally_balanced_approach(X, Y)
        case "globally_balanced":
            X_dist = distribute_tasks_with_globally_balanced_approach(X, Y)
    # Calculate the class coverage and class standard deviation scores.
    KCov_X, KStd_X = calculate_class_coverage_and_standard_deviation_scores(X, X_dist, t, Y)
    # Set the dictionary of estimated costs for the schedule X.
    X_costs = {"X": X,
               "X_dist": X_dist,
               "M_X": M_X,
               "E_X": E_X,
               "B_X": B_X,
               "D_X": D_X,
               "KCov_X": KCov_X,
               "KStd_X": KStd_X}
    # Return the dictionary of estimated costs for the schedule X.
    return X_costs
