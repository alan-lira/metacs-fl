from bisect import bisect_left
from numpy import array
from numpy.linalg import linalg
from random import choice, sample
from time import time


def select_all_available_clients(available_clients_map: dict,
                                 phase: str) -> dict:
    selected_clients = {}
    for client_key, client_values in available_clients_map.items():
        client_proxy = client_values["client_proxy"]
        client_task_assignment_capacities_phase_key = "client_task_assignment_capacities_{0}".format(phase)
        client_task_assignment_capacities_phase = client_values[client_task_assignment_capacities_phase_key]
        client_max_task_capacity = max(client_values["client_task_assignment_capacities_{0}".format(phase)])
        selected_clients.update({client_key: {"client_proxy": client_proxy,
                                              client_task_assignment_capacities_phase_key: client_task_assignment_capacities_phase,
                                              "client_max_task_capacity": client_max_task_capacity,
                                              "client_num_tasks_scheduled": 0}})
    return selected_clients


def select_random_fraction_available_clients(available_clients_map: dict,
                                             phase: str,
                                             clients_fraction: float) -> dict:
    selected_clients = {}
    if 0 < clients_fraction <= 1:
        num_available_clients = len(available_clients_map)
        num_clients_to_select = max(1, int(num_available_clients * clients_fraction))
        sampled_clients_keys = sample(sorted(available_clients_map), num_clients_to_select)
        for client_key in sampled_clients_keys:
            client_map = available_clients_map[client_key]
            client_proxy = client_map["client_proxy"]
            client_task_assignment_capacities_phase_key = "client_task_assignment_capacities_{0}".format(phase)
            client_task_assignment_capacities_phase = client_map[client_task_assignment_capacities_phase_key]
            client_max_task_capacity = max(client_map["client_task_assignment_capacities_{0}".format(phase)])
            selected_clients.update({client_key: {"client_proxy": client_proxy,
                                                  client_task_assignment_capacities_phase_key: client_task_assignment_capacities_phase,
                                                  "client_max_task_capacity": client_max_task_capacity,
                                                  "client_num_tasks_scheduled": 0}})
    return selected_clients


def take_closest(values: list,
                 value: int) -> int:
    value_idx = bisect_left(values, value)
    if value_idx == 0:
        return values[0]
    if value_idx == len(values):
        return values[-1]
    prev_idx = values[value_idx - 1]
    next_idx = values[value_idx]
    closest = next_idx if (next_idx - value < value - prev_idx) else prev_idx
    return closest


def find_combination_dp(lists: list,
                        target: int) -> list | None:
    # Memoization table to store results for sub-problems (index, current_sum).
    memo = {}
    def dp(index: int,
           current_sum: int) -> list | None:
        # If all lists were considered...
        if index == len(lists):
            # If the sum equals the target, return an empty list (indicating a valid path).
            return [] if current_sum == target else None
        # If this sub-problem have already been computed, return the result.
        if (index, current_sum) in memo:
            return memo[(index, current_sum)]
        # Try selecting each number from the current list.
        for num in lists[index]:
            # Recursively try to solve the next sub-problem with the updated sum.
            result = dp(index + 1, current_sum + num)
            if result is not None:
                # If a valid combination is found, append the current number to it.
                memo[(index, current_sum)] = [num] + result
                return memo[(index, current_sum)]
        # If no valid combination is found, return None.
        memo[(index, current_sum)] = None
        return None
    # Start from the first list with sum 0.
    return dp(0, 0)


def find_combination(lists: list,
                     target: int,
                     index: int = 0,
                     current_sum: int = 0,
                     current_combination = None) -> list | None:
    if current_combination is None:
        current_combination = []
    # If all lists were considered...
    if index == len(lists):
        # Check if the current combination sum matches the target.
        if current_sum == target:
            return current_combination
        else:
            return None
    # Try each element in the current list...
    for num in lists[index]:
        # Call recursively for the next list with the updated sum and combination.
        result = find_combination(lists, target, index + 1, current_sum + num, current_combination + [num])
        if result:
            return result
    # If no combination found, return None.
    return None


def get_all_possible_sums(lists: list) -> list:
    # Initialize with the sum of an empty combination (0).
    possible_sums = {0}
    for lst in lists:
        # Create a new set to store the updated sums.
        new_sums = set()
        # For each sum already in possible_sums, add each element of the current list.
        for value in lst:
            for s in possible_sums:
                new_sums.add(s + value)
        # Update possible_sums to include the new sums.
        possible_sums = new_sums
    # Return the sorted list of all possible sums.
    return sorted(possible_sums)


def schedule_minimum_tasks_to_all_clients(selected_clients: dict,
                                          phase: str,
                                          start_time: float | None = None,
                                          max_duration_seconds: float = 120.0) -> None:
    # Initialize the timer.
    if start_time is None:
        start_time = time()
    while True:
        # If any stopping condition was met...
        if time() - start_time > max_duration_seconds:
            break  # return the current incomplete schedule.
        # Get the client indices that have no tasks scheduled.
        clients_indices = [client_id
                           for client_id, client_info in selected_clients.items()
                           if client_info["client_num_tasks_scheduled"] == 0]
        if not clients_indices:
            # All clients received a minimum number of tasks.
            break
        # Randomly sample a client index.
        client_id_sampled = sample(clients_indices, 1)[0]
        # Get the index of the current capacity used.
        client_task_assignment_capacities \
            = selected_clients[client_id_sampled]["client_task_assignment_capacities_{0}".format(phase)]
        client_current_num_tasks_scheduled = selected_clients[client_id_sampled]["client_num_tasks_scheduled"]
        cap_idx = list(client_task_assignment_capacities).index(client_current_num_tasks_scheduled)
        if cap_idx != len(client_task_assignment_capacities) - 1:
            # Get the next valid capacity of the client i.
            cap_next = client_task_assignment_capacities[cap_idx + 1]
            # Set the assignment of client i to its next valid capacity.
            selected_clients[client_id_sampled]["client_num_tasks_scheduled"] = cap_next


def safely_copy_selected_clients(selected_clients: dict,
                                 phase: str) -> dict:
    selected_clients_copy = {}
    for client_id, client_info in selected_clients.items():
        client_task_assignment_capacities = client_info["client_task_assignment_capacities_{0}".format(phase)]
        client_num_tasks_scheduled = client_info["client_num_tasks_scheduled"]
        client_max_task_capacity = client_info["client_max_task_capacity"]
        # Only keep copyable fields.
        copyable_client_info = {"client_task_assignment_capacities_{0}".format(phase): client_task_assignment_capacities,
                                "client_num_tasks_scheduled": client_num_tasks_scheduled,
                                "client_max_task_capacity": client_max_task_capacity}
        selected_clients_copy[client_id] = copyable_client_info
    return selected_clients_copy


def restore_noncopyable_attributes(selected_clients: dict,
                                   selected_clients_copy: dict) -> None:
    for client_id in selected_clients_copy:
        original_client_info = selected_clients[client_id]
        selected_clients_copy[client_id]["client_proxy"] = original_client_info["client_proxy"]


def find_different_schedule(selected_clients: dict,
                            phase: str,
                            previous_schedules_to_avoid: list) -> dict:
    # Deep copy the current list of selected clients.
    selected_clients_copy = safely_copy_selected_clients(selected_clients, phase)
    # For each selected client i ...
    for client_i, client_info_i in selected_clients_copy.items():
        tasks_i = client_info_i["client_num_tasks_scheduled"]
        capacities_i = client_info_i["client_task_assignment_capacities_{0}".format(phase)]
        # Search for clients j that can swap their scheduled number of tasks with client i.
        swap_clients_indices = [client_j for client_j, client_info_j in selected_clients_copy.items()
                                if client_j != client_i
                                and client_info_i["client_num_tasks_scheduled"] in client_info_j["client_task_assignment_capacities_{0}".format(phase)]
                                and client_info_j["client_num_tasks_scheduled"] in capacities_i
                                and client_info_j["client_num_tasks_scheduled"] != tasks_i]
        if swap_clients_indices:
            # Randomly sample a client index.
            client_j_sampled = sample(swap_clients_indices, 1)[0]
            tasks_j = selected_clients_copy[client_j_sampled]["client_num_tasks_scheduled"]
            # Perform the swap of tasks.
            selected_clients_copy[client_i]["client_num_tasks_scheduled"] = tasks_j
            selected_clients_copy[client_j_sampled]["client_num_tasks_scheduled"] = tasks_i
            # Break after a successful swap
            break
    # Get the generated schedule using consistent client order
    different_schedule = [selected_clients_copy[client]["client_num_tasks_scheduled"]
                          for client in sorted(selected_clients_copy)]
    # Check if the schedule is really different...
    if different_schedule not in previous_schedules_to_avoid:
        restore_noncopyable_attributes(selected_clients, selected_clients_copy)
        return selected_clients_copy
    # Return the original if no valid different schedule found.
    return selected_clients


def remove_tasks_from_a_random_client(selected_clients: dict,
                                      phase: str,
                                      num_tasks_to_remove: int = 0,
                                      schedule_to_all_clients: bool = False) -> int:
    # Get the client indices that have at least one task scheduled.
    clients_indices = [client_id
                       for client_id, client_info in selected_clients.items()
                       if client_info["client_num_tasks_scheduled"] > 0]
    if clients_indices:
        # Randomly sample a client index.
        client_id_sampled = sample(clients_indices, 1)[0]
        # Get the index of the current capacity used.
        client_task_assignment_capacities \
            = selected_clients[client_id_sampled]["client_task_assignment_capacities_{0}".format(phase)]
        client_previous_num_tasks_scheduled = selected_clients[client_id_sampled]["client_num_tasks_scheduled"]
        cap_idx = list(client_task_assignment_capacities).index(client_previous_num_tasks_scheduled)
        if num_tasks_to_remove > 0:
            # Calculate the new number of tasks after the removal.
            num_tasks_after_removal = client_previous_num_tasks_scheduled - num_tasks_to_remove
            if (num_tasks_after_removal in client_task_assignment_capacities) and \
               ((not schedule_to_all_clients) or (schedule_to_all_clients and num_tasks_after_removal > 0)):
                # Set the assignment of client i to a previous valid capacity.
                selected_clients[client_id_sampled]["client_num_tasks_scheduled"] = num_tasks_after_removal
        else:
            if (not schedule_to_all_clients and cap_idx != 0) or (schedule_to_all_clients and cap_idx > 1):
                # Get the previous valid capacity of the client i.
                cap_prev = client_task_assignment_capacities[cap_idx - 1]
                # Set the assignment of client i to its previous valid capacity.
                selected_clients[client_id_sampled]["client_num_tasks_scheduled"] = cap_prev
        client_current_num_tasks_scheduled = selected_clients[client_id_sampled]["client_num_tasks_scheduled"]
        num_tasks_removed = client_previous_num_tasks_scheduled - client_current_num_tasks_scheduled
        return num_tasks_removed
    return 0


def add_tasks_to_a_random_client(selected_clients: dict,
                                 phase: str,
                                 num_tasks_to_add: int = 0) -> int:
    # Get the client indices that have any number of tasks scheduled.
    clients_indices = [client_id for client_id, _ in selected_clients.items()]
    if clients_indices:
        # Randomly sample a client index.
        client_id_sampled = sample(clients_indices, 1)[0]
        # Get the index of the current capacity used.
        client_task_assignment_capacities \
            = selected_clients[client_id_sampled]["client_task_assignment_capacities_{0}".format(phase)]
        client_previous_num_tasks_scheduled = selected_clients[client_id_sampled]["client_num_tasks_scheduled"]
        cap_idx = list(client_task_assignment_capacities).index(client_previous_num_tasks_scheduled)
        if num_tasks_to_add > 0:
            # Calculate the new number of tasks after the addition.
            num_tasks_after_addition = client_previous_num_tasks_scheduled + num_tasks_to_add
            if num_tasks_after_addition in client_task_assignment_capacities:
                # Set the assignment of client i to a next valid capacity.
                selected_clients[client_id_sampled]["client_num_tasks_scheduled"] = num_tasks_after_addition
        else:
            if cap_idx != len(client_task_assignment_capacities) - 1:
                # Get the next valid capacity of the client i.
                cap_next = client_task_assignment_capacities[cap_idx + 1]
                # Set the assignment of client i to its next valid capacity.
                selected_clients[client_id_sampled]["client_num_tasks_scheduled"] = cap_next
        client_current_num_tasks_scheduled = selected_clients[client_id_sampled]["client_num_tasks_scheduled"]
        num_tasks_added = client_current_num_tasks_scheduled - client_previous_num_tasks_scheduled
        return num_tasks_added
    return 0


def schedule_tasks_to_selected_clients(num_tasks_to_schedule: int,
                                       selected_clients: dict,
                                       phase: str,
                                       profiling_round: bool,
                                       previous_schedules_to_avoid: list | None = None,
                                       schedule_to_all_clients: bool = False,
                                       max_duration_seconds: float = 120.0) -> dict:
    # Initialize the timer.
    start_time = time()
    # If there are no tasks to schedule or selected clients, end.
    if num_tasks_to_schedule == 0 or not selected_clients:
        return selected_clients
    # Initialize the list of task assignment capacities per client.
    task_assignment_capacities_list = []
    # Set the key for the task assignment capacities per client (current phase).
    client_task_assignment_capacities_key = "client_task_assignment_capacities_{0}".format(phase)
    for _, client_info in selected_clients.items():
        # Get the task assignment capacities of client i.
        client_task_assignment_capacities_phase = client_info[client_task_assignment_capacities_key]
        # Append the task assignment capacities of client i to the list of task assignment capacities.
        task_assignment_capacities_list.append(client_task_assignment_capacities_phase)
    # Compute all the possible sums of task assignments, considering one assignment per client.
    all_possible_task_assignment_sums = get_all_possible_sums(task_assignment_capacities_list)
    # If the number of tasks to schedule is infeasible...
    if num_tasks_to_schedule not in all_possible_task_assignment_sums:
        # Set a new valid number of tasks to schedule.
        num_tasks_to_schedule = take_closest(all_possible_task_assignment_sums, num_tasks_to_schedule)
    # If is a profile round or all clients must have tasks, initially schedule a minimum number of tasks to all clients.
    if profiling_round or schedule_to_all_clients:
        schedule_minimum_tasks_to_all_clients(selected_clients, phase, start_time)
    # While there are tasks left to schedule...
    remove_action = True
    while True:
        # If any stopping condition was met...
        if time() - start_time > max_duration_seconds:
            if previous_schedules_to_avoid:
                # Fallback: return a random previous schedule (originally supposed to be avoided).
                return choice(list(previous_schedules_to_avoid))
            else:
                # Fallback: return the current incomplete schedule.
                return selected_clients
        # Initialize the variable used to invalidate the current schedule.
        invalidate_current_schedule = False
        # If is a profile round and there are previous schedules to be avoided...
        if profiling_round and previous_schedules_to_avoid:
            # Find a different schedule (clients should ideally have different schedules during profiling).
            new_schedule = find_different_schedule(selected_clients,
                                                   phase,
                                                   previous_schedules_to_avoid)
            if new_schedule is not None:
                selected_clients = new_schedule
        # If all clients must receive tasks, verify if this constraint was met.
        if schedule_to_all_clients:
            all_clients_have_tasks = all(client_info["client_num_tasks_scheduled"] > 0
                                         for _, client_info in selected_clients.items())
            if not all_clients_have_tasks:
                invalidate_current_schedule = True
        all_clients_have_a_valid_num_tasks = all(client_info["client_num_tasks_scheduled"]
                                                 in client_info[client_task_assignment_capacities_key]
                                                 for _, client_info in selected_clients.items())
        if not all_clients_have_a_valid_num_tasks:
            invalidate_current_schedule = True
        # Invalidate the current schedule, if needed.
        if invalidate_current_schedule:
            if remove_action:
                # Remove tasks from a random client.
                remove_tasks_from_a_random_client(selected_clients, phase, schedule_to_all_clients=schedule_to_all_clients)
            else:
                # Add tasks to a random client.
                add_tasks_to_a_random_client(selected_clients, phase)
            remove_action = not remove_action
        # Get the current schedule.
        current_schedule = [client_info["client_num_tasks_scheduled"] for _, client_info in selected_clients.items()]
        # Get the current number of tasks assigned.
        num_tasks_scheduled = sum(current_schedule)
        # Verify if all the tasks have been scheduled...
        if not invalidate_current_schedule and num_tasks_scheduled == num_tasks_to_schedule:
            # If so, filter out the clients with no tasks scheduled, if any.
            selected_clients_filtered = {client_id: client_info
                                         for client_id, client_info in selected_clients.items()
                                         if client_info["client_num_tasks_scheduled"] > 0}
            return selected_clients_filtered
        if num_tasks_scheduled > num_tasks_to_schedule:
            # Remove tasks from a random client.
            remove_tasks_from_a_random_client(selected_clients, phase, schedule_to_all_clients=schedule_to_all_clients)
        else:
            # Add tasks to a random client.
            add_tasks_to_a_random_client(selected_clients, phase)


def calculate_linear_interpolation_or_extrapolation(x1: int | float,
                                                    x2: int | float,
                                                    y1: int | float,
                                                    y2: int | float,
                                                    x: int | float) -> int | float:
    # Calculate the slope m of the line.
    m = (y2 - y1) / (x2 - x1)
    # Calculate the value of y using the line equation.
    y = y1 + m * (x - x1)
    # Return the y value.
    return y


def calculate_quadratic_interpolation_or_extrapolation(x1: int | float,
                                                       x2: int | float,
                                                       x3: int | float,
                                                       y1: int | float,
                                                       y2: int | float,
                                                       y3: int | float,
                                                       x: int | float) -> int | float:
    # A: the matrix of the x-values and powers.
    A = array([[x1 ** 2, x1, 1], [x2 ** 2, x2, 1], [x3 ** 2, x3, 1]])
    # B: the vector of the y-values.
    B = array([y1, y2, y3])
    # Solve the system of equations (matrix form) to get the coefficients of the quadratic equation (a, b, and c).
    a, b, c = linalg.solve(A, B)
    # Use the quadratic equation to estimate the value of y.
    y = a * x ** 2 + b * x + c
    # If the quadratic result is negative, switch to linear.
    if y < 0:
        y = calculate_linear_interpolation_or_extrapolation(x1, x2, y1, y2, x)
    # Return the y value.
    return y
