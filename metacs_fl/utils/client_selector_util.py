from bisect import bisect_left
from numpy import array, ceil
from numpy.linalg import linalg
from random import sample, shuffle
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
                                             clients_fraction: float,
                                             num_tasks_to_schedule: int) -> dict:
    phase_key = "client_task_assignment_capacities_{0}".format(phase)
    # Build (cid, capacity) list.
    items = []
    for cid, cmap in available_clients_map.items():
        cap = max(cmap[phase_key])
        items.append((cid, cap))
    num_available_clients = len(available_clients_map)
    num_selected_clients = max(1, int(ceil(clients_fraction * num_available_clients)))
    shuffle(items)
    selected = []
    cumulative = 0
    # Pick target fraction.
    for cid, cap in items[:num_selected_clients]:
        selected.append((cid, cap))
        cumulative += cap
    # Repair if capacity is insufficient.
    if cumulative < num_tasks_to_schedule:
        for cid, cap in items[num_selected_clients:]:
            selected.append((cid, cap))
            cumulative += cap
            if cumulative >= num_tasks_to_schedule:
                break
    # Build return structure.
    selected_clients = {}
    for cid, _ in selected:
        cmap = available_clients_map[cid]
        caps = cmap[phase_key]
        selected_clients[cid] = {"client_proxy": cmap["client_proxy"],
                                 phase_key: caps,
                                 "client_max_task_capacity": max(caps),
                                 "client_num_tasks_scheduled": 0}
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


def find_multichoice_combination_closest(lists: list,
                                         target: int,
                                         prefer_lower: bool = False) -> tuple:
    if not lists:
        return [], 0
    # Parents[stage]: dict mapping sum_after_stage -> (prev_sum_before_stage, chosen_value).
    parents = []
    reachable = {0}
    # Iterate over stages (clients).
    for choices in lists:
        # Deterministic sorted unique choices.
        choices_list = sorted(set(int(c) for c in choices))
        new_reachable = set()
        parent = {}
        # Extend previous reachable sums with each choice.
        for s in reachable:
            for v in choices_list:
                ns = s + v
                # Record first parent seen for determinism.
                if ns not in new_reachable:
                    new_reachable.add(ns)
                    parent[ns] = (s, v)
        # If no sums reachable after including this client's choices,
        # it means no valid combination exists that picks one per client up to this point.
        if not new_reachable:
            # Fallback: pick best among keys of parent.
            if not parent:
                return [], 0
            best_sum = min(parent.keys(), key=lambda x: (abs(x - target), x if prefer_lower else -x))
            # Backtrack across available parents (parents list may be shorter than lists).
            stages = len(parents) + 1
            combination = [0] * stages
            cur_sum = best_sum
            # Reconstruct last stage from parent dict.
            prev_sum, val = parent[cur_sum]
            combination[stages - 1] = val
            cur_sum = prev_sum
            # Reconstruct earlier stages.
            for stage in range(len(parents) - 1, -1, -1):
                p = parents[stage]
                prev_sum, val = p[cur_sum]
                combination[stage] = val
                cur_sum = prev_sum
            # Pad with zeros if somehow lists longer.
            if len(combination) < len(lists):
                combination += [0] * (len(lists) - len(combination))
            return combination, best_sum
        parents.append(parent)
        reachable = new_reachable
    # Choose reachable sum closest to target (tie-breaker prefers larger unless prefer_lower True).
    best_sum = min(reachable, key=lambda x: (abs(x - target), x if prefer_lower else -x))
    # Backtrack to build combination.
    n = len(lists)
    combination = [0] * n
    cur_sum = best_sum
    for stage in range(n - 1, -1, -1):
        parent = parents[stage]
        prev_sum, val = parent[cur_sum]
        combination[stage] = val
        cur_sum = prev_sum
    return combination, best_sum


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
    if previous_schedules_to_avoid is None or different_schedule not in previous_schedules_to_avoid:
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


def distribute_remaining_tasks_balanced(selected_clients: dict,
                                        client_capacities_map: dict,
                                        remaining_tasks: int,
                                        schedule_to_all_clients: bool) -> None:
    while remaining_tasks != 0:
        if remaining_tasks > 0:
            # Add tasks to the most under-utilized client that can accept more.
            candidates = []
            for client_id, client_info in selected_clients.items():
                capacities = client_capacities_map[client_id]
                current = client_info["client_num_tasks_scheduled"]
                max_capacity = capacities[-1]
                if current < max_capacity:
                    # Calculate utilization ratio (current/max).
                    utilization = current / max_capacity if max_capacity > 0 else 0
                    # Find next capacity.
                    next_capacity = find_next_capacity(capacities, current)
                    if next_capacity is not None:
                        task_increase = next_capacity - current
                        candidates.append((client_id, utilization, task_increase))
            if not candidates:
                break
            # Prefer clients with the lowest utilization (most under-utilized).
            candidates.sort(key=lambda x: (x[1], -x[2]))  # Low utilization, high capacity increases.
            best_client_id, _, task_increase = candidates[0]
            # Apply the increase.
            current = selected_clients[best_client_id]["client_num_tasks_scheduled"]
            next_capacity = find_next_capacity(client_capacities_map[best_client_id], current)
            selected_clients[best_client_id]["client_num_tasks_scheduled"] = next_capacity
            remaining_tasks -= task_increase
        else:  # remaining_tasks < 0
            # Remove tasks from the most over-utilized client that can reduce.
            candidates = []
            for client_id, client_info in selected_clients.items():
                capacities = client_capacities_map[client_id]
                current = client_info["client_num_tasks_scheduled"]
                min_capacity = capacities[0]
                can_reduce = current > min_capacity
                if schedule_to_all_clients:
                    can_reduce = can_reduce and current > 1
                if can_reduce:
                    # Calculate utilization ratio.
                    max_capacity = capacities[-1]
                    utilization = current / max_capacity if max_capacity > 0 else 1.0
                    # Find previous capacity.
                    prev_capacity = find_previous_capacity(capacities, current)
                    if prev_capacity is not None:
                        task_decrease = current - prev_capacity
                        candidates.append((client_id, utilization, task_decrease))
            if not candidates:
                break
            # Prefer clients with the highest utilization (most over-utilized).
            candidates.sort(key=lambda x: (-x[1], -x[2]))  # High utilization, high capacity decreases
            best_client_id, _, task_decrease = candidates[0]
            # Apply the decrease.
            current = selected_clients[best_client_id]["client_num_tasks_scheduled"]
            prev_capacity = find_previous_capacity(client_capacities_map[best_client_id], current)
            selected_clients[best_client_id]["client_num_tasks_scheduled"] = prev_capacity
            remaining_tasks += task_decrease


def adjust_capacity_weighted_distribution(provisional_schedule: dict,
                                          selected_clients: dict,
                                          client_capacities_map: dict,
                                          remaining_tasks: int) -> None:
    if remaining_tasks > 0:
        # Distribute extra tasks to clients that can accept more.
        clients_by_spare_capacity = []
        for client_id, scheduled_tasks in provisional_schedule.items():
            capacities = client_capacities_map[client_id]
            max_capacity = max(capacities)
            spare_capacity = max_capacity - scheduled_tasks
            if spare_capacity > 0:
                # Find the actual next valid capacity.
                next_capacity = find_next_capacity(capacities, scheduled_tasks)
                if next_capacity is not None:
                    actual_increase = next_capacity - scheduled_tasks
                    clients_by_spare_capacity.append((client_id, spare_capacity, actual_increase, next_capacity))
        # Sort by spare capacity (descending) and then by actual increase needed (ascending).
        clients_by_spare_capacity.sort(key=lambda x: (-x[1], x[2]))
        for client_id, _, actual_increase, next_capacity in clients_by_spare_capacity:
            if actual_increase <= remaining_tasks:
                provisional_schedule[client_id] = next_capacity
                remaining_tasks -= actual_increase
                if remaining_tasks <= 0:
                    break
    else:  # remaining_tasks < 0.
        # Remove excess tasks from clients that can reduce.
        tasks_to_remove = -remaining_tasks
        clients_by_current_load = []
        for client_id, scheduled_tasks in provisional_schedule.items():
            capacities = client_capacities_map[client_id]
            min_capacity = min(capacities)
            if scheduled_tasks > min_capacity:
                # Find the actual previous valid capacity.
                prev_capacity = find_previous_capacity(capacities, scheduled_tasks)
                if prev_capacity is not None:
                    actual_decrease = scheduled_tasks - prev_capacity
                    # Calculate current utilization for prioritization.
                    max_capacity = max(capacities)
                    utilization = scheduled_tasks / max_capacity if max_capacity > 0 else 1.0
                    clients_by_current_load.append((client_id, utilization, actual_decrease, prev_capacity))
        # Sort by utilization (descending) - remove from most utilized first.
        clients_by_current_load.sort(key=lambda x: (-x[1], x[2]))
        for client_id, _, actual_decrease, prev_capacity in clients_by_current_load:
            if actual_decrease <= tasks_to_remove:
                provisional_schedule[client_id] = prev_capacity
                tasks_to_remove -= actual_decrease
                if tasks_to_remove <= 0:
                    break


def find_closest_capacity(capacities: list,
                          target: int) -> int:
    return min(capacities, key=lambda x: abs(x - target))


def distribute_tasks_capacity_weighted(selected_clients: dict,
                                       client_capacities_map: dict,
                                       num_tasks_to_schedule: int) -> None:
    if not selected_clients:
        return
    # Calculate total capacity weight (sum of max capacities).
    total_max_capacity = sum(max(capacities) for capacities in client_capacities_map.values())
    if total_max_capacity == 0:
        # Fallback: assign minimum tasks to all clients.
        for client_id, client_info in selected_clients.items():
            capacities = client_capacities_map[client_id]
            client_info["client_num_tasks_scheduled"] = min(capacities)
        return
    # First pass: assign tasks proportionally to max capacity.
    remaining_tasks = num_tasks_to_schedule
    provisional_schedule = {}
    for client_id, client_info in selected_clients.items():
        capacities = client_capacities_map[client_id]
        max_capacity = max(capacities)
        # Calculate proportional allocation.
        proportional_share = (max_capacity / total_max_capacity) * num_tasks_to_schedule
        target_tasks = max(min(capacities), int(proportional_share))
        # Find the closest valid capacity.
        closest_capacity = find_closest_capacity(capacities, target_tasks)
        provisional_schedule[client_id] = closest_capacity
        remaining_tasks -= closest_capacity
    # Second pass: adjust for remaining tasks.
    if remaining_tasks != 0:
        adjust_capacity_weighted_distribution(provisional_schedule,
                                              selected_clients,
                                              client_capacities_map,
                                              remaining_tasks)
    # Apply the final schedule.
    for client_id, task_count in provisional_schedule.items():
        selected_clients[client_id]["client_num_tasks_scheduled"] = task_count


def balanced_initial_schedule(selected_clients: dict,
                              client_capacities_map: dict,
                              num_tasks_to_schedule: int,
                              schedule_to_all_clients: bool) -> None:
    if schedule_to_all_clients:
        # Use balanced distribution for all clients.
        base_tasks_per_client = num_tasks_to_schedule // len(selected_clients)
        remaining_tasks = num_tasks_to_schedule
        # First pass: assign base tasks to all clients.
        for client_id, client_info in selected_clients.items():
            capacities = client_capacities_map[client_id]
            min_capacity = capacities[0]
            # Find the closest valid capacity to base_tasks_per_client.
            target = max(min_capacity, base_tasks_per_client)
            closest_capacity = find_closest_capacity(capacities, target)
            client_info["client_num_tasks_scheduled"] = closest_capacity
            remaining_tasks -= closest_capacity
        # Second pass: distribute remaining tasks while maintaining balance.
        if remaining_tasks != 0:
            distribute_remaining_tasks_balanced(selected_clients,
                                                client_capacities_map,
                                                remaining_tasks,
                                                schedule_to_all_clients)
    else:
        # For non-all scheduling, use capacity-weighted distribution.
        distribute_tasks_capacity_weighted(selected_clients, client_capacities_map, num_tasks_to_schedule)


def schedule_tasks_to_selected_clients(num_tasks_to_schedule: int,
                                       selected_clients: dict,
                                       phase: str,
                                       schedule_to_all_clients: bool = False,
                                       max_duration_seconds: float = 120.0) -> dict:
    start_time = time()
    if num_tasks_to_schedule == 0 or not selected_clients:
        return selected_clients
    # Initialize the list of task assignment capacities per client.
    task_assignment_capacities_list = []
    client_ids = list(selected_clients.keys())  # Preserve order.
    # Set the key for the task assignment capacities per client (current phase).
    client_task_assignment_capacities_key = "client_task_assignment_capacities_{0}".format(phase)
    for _, client_info in selected_clients.items():
        # Get the task assignment capacities of client i.
        client_task_assignment_capacities_phase = client_info[client_task_assignment_capacities_key]
        # Append the task assignment capacities of client i to the list of task assignment capacities.
        task_assignment_capacities_list.append(client_task_assignment_capacities_phase)
    # If scheduling to all clients: use multiple-choice DP to get explicit per-client assignment.
    if schedule_to_all_clients:
        cleaned_capacities = []
        for caps in task_assignment_capacities_list:
            cleaned_capacities.append([c for c in caps if c > 0])
        combination, achieved_sum = find_multichoice_combination_closest(cleaned_capacities, num_tasks_to_schedule)
        # Apply combination to selected_clients (aligned with client_ids).
        for cid, chosen in zip(client_ids, combination):
            selected_clients[cid]["client_num_tasks_scheduled"] = chosen
        # Return the DP-produced schedule (best-effort, explicit per-client assignment).
        return selected_clients
    # Otherwise (not scheduling to all clients)...
    # Compute all the possible sums of task assignments, considering one assignment per client.
    all_possible_task_assignment_sums = get_all_possible_sums(task_assignment_capacities_list)
    # If the number of tasks to schedule is infeasible...
    if num_tasks_to_schedule not in all_possible_task_assignment_sums:
        # Set a new valid number of tasks to schedule.
        num_tasks_to_schedule = take_closest(all_possible_task_assignment_sums, num_tasks_to_schedule)
    # Calculate min/max possible sums.
    min_possible_sum = 0
    max_possible_sum = 0
    client_capacities_map = {}
    for client_id, client_info in selected_clients.items():
        capacities = client_info[client_task_assignment_capacities_key]
        client_capacities_map[client_id] = sorted(capacities)  # Keep sorted for efficient lookup
        min_possible_sum += min(capacities)
        max_possible_sum += max(capacities)
    # Adjust target if infeasible.
    if num_tasks_to_schedule < min_possible_sum:
        num_tasks_to_schedule = min_possible_sum
    elif num_tasks_to_schedule > max_possible_sum:
        num_tasks_to_schedule = max_possible_sum
    # Use balanced adjustment strategy.
    return balanced_adjustment_strategy(selected_clients,
                                        client_capacities_map,
                                        num_tasks_to_schedule,
                                        phase,
                                        schedule_to_all_clients,
                                        start_time,
                                        max_duration_seconds)


def calculate_balance_metric(utilizations: list) -> float:
    if not utilizations:
        return 0.0
    mean = sum(utilizations) / len(utilizations)
    variance = sum((u - mean) ** 2 for u in utilizations) / len(utilizations)
    return variance ** 0.5


def get_schedule_tuple(selected_clients: dict) -> tuple:
    return tuple(sorted((cid, info["client_num_tasks_scheduled"]) for cid, info in selected_clients.items()))


def find_balanced_alternative(selected_clients: dict,
                              client_capacities_map: dict,
                              previous_schedules_to_avoid: list) -> bool:
    # Try small swaps between clients.
    client_ids = list(selected_clients.keys())
    for i in range(len(client_ids)):
        for j in range(i + 1, len(client_ids)):
            client_i, client_j = client_ids[i], client_ids[j]
            cap_i = client_capacities_map[client_i]
            cap_j = client_capacities_map[client_j]
            current_i = selected_clients[client_i]["client_num_tasks_scheduled"]
            current_j = selected_clients[client_j]["client_num_tasks_scheduled"]
            # Check if swap is possible and valid.
            if current_j in cap_i and current_i in cap_j:
                # Try the swap.
                selected_clients[client_i]["client_num_tasks_scheduled"] = current_j
                selected_clients[client_j]["client_num_tasks_scheduled"] = current_i
                new_schedule = get_schedule_tuple(selected_clients)
                if previous_schedules_to_avoid is None or new_schedule not in previous_schedules_to_avoid:
                    return True
                # Swap back if not valid.
                selected_clients[client_i]["client_num_tasks_scheduled"] = current_i
                selected_clients[client_j]["client_num_tasks_scheduled"] = current_j

    return False


def balanced_adjustment_strategy(selected_clients: dict,
                                 client_capacities_map: dict,
                                 num_tasks_to_schedule: int,
                                 phase: str,
                                 schedule_to_all_clients: bool,
                                 start_time: float,
                                 max_duration_seconds: float) -> dict:
    max_iterations = len(selected_clients) * 20
    iteration = 0
    while iteration < max_iterations and (time() - start_time) < max_duration_seconds:
        iteration += 1
        current_total = sum(client_info["client_num_tasks_scheduled"]
                            for client_info in selected_clients.values())
        difference = num_tasks_to_schedule - current_total
        if difference == 0:
            break
        # Calculate current balance metric (standard deviation of utilization).
        utilizations = []
        for client_id, client_info in selected_clients.items():
            capacities = client_capacities_map[client_id]
            current = client_info["client_num_tasks_scheduled"]
            max_cap = capacities[-1]
            utilization = current / max_cap if max_cap > 0 else 0
            utilizations.append(utilization)
        current_balance = calculate_balance_metric(utilizations)
        if difference > 0:
            # Try to add tasks while maintaining balance.
            if not balanced_task_addition(selected_clients,
                                          client_capacities_map,
                                          difference,
                                          current_balance):
                break
        else:
            # Try to remove tasks while maintaining balance.
            if not balanced_task_removal(selected_clients,
                                         client_capacities_map,
                                         abs(difference),
                                         current_balance,
                                         schedule_to_all_clients):
                break
    # Final validation.
    if schedule_to_all_clients:
        for client_info in selected_clients.values():
            if client_info["client_num_tasks_scheduled"] == 0:
                capacities = client_info["client_task_assignment_capacities_{0}".format(phase)]
                client_info["client_num_tasks_scheduled"] = min(c for c in capacities if c > 0)
    if not schedule_to_all_clients:
        selected_clients = {cid: info for cid, info in selected_clients.items()
                            if info["client_num_tasks_scheduled"] > 0}
    return selected_clients


def find_next_capacity(capacities: list,
                       current: int) -> int | None:
    for cap in capacities:
        if cap > current:
            return cap
    return None


def balanced_task_addition(selected_clients: dict,
                           client_capacities_map: dict,
                           tasks_to_add: int,
                           current_balance: float) -> bool:
    candidates = []
    for client_id, client_info in selected_clients.items():
        capacities = client_capacities_map[client_id]
        current = client_info["client_num_tasks_scheduled"]
        max_capacity = capacities[-1]
        if current < max_capacity:
            next_capacity = find_next_capacity(capacities, current)
            if next_capacity is not None:
                increase = next_capacity - current
                new_utilization = next_capacity / max_capacity
                # Calculate potential new balance.
                utilizations = []
                for other_id, other_info in selected_clients.items():
                    if other_id == client_id:
                        utilizations.append(new_utilization)
                    else:
                        other_capacities = client_capacities_map[other_id]
                        other_current = other_info["client_num_tasks_scheduled"]
                        other_max = other_capacities[-1]
                        utilizations.append(other_current / other_max if other_max > 0 else 0)
                new_balance = calculate_balance_metric(utilizations)
                balance_improvement = current_balance - new_balance  # Lower is better.
                candidates.append((client_id, balance_improvement, increase, next_capacity))
    if not candidates:
        return False
    # Prefer moves that improve balance the most.
    candidates.sort(key=lambda x: (-x[1], x[2]))  # High balance improvement, small increases.
    for client_id, _, increase, new_capacity in candidates:
        if increase <= tasks_to_add:
            selected_clients[client_id]["client_num_tasks_scheduled"] = new_capacity
            return True
    return False


def find_previous_capacity(capacities: list,
                           current: int) -> int | None:
    for cap in reversed(capacities):
        if cap < current:
            return cap
    return None


def balanced_task_removal(selected_clients: dict,
                          client_capacities_map: dict,
                          tasks_to_remove: int,
                          current_balance: float,
                          schedule_to_all_clients: bool) -> bool:
    candidates = []
    for client_id, client_info in selected_clients.items():
        capacities = client_capacities_map[client_id]
        current = client_info["client_num_tasks_scheduled"]
        min_capacity = capacities[0]
        can_reduce = current > min_capacity
        if schedule_to_all_clients:
            can_reduce = can_reduce and current > 1
        if can_reduce:
            prev_capacity = find_previous_capacity(capacities, current)
            if prev_capacity is not None:
                decrease = current - prev_capacity
                max_capacity = capacities[-1]
                new_utilization = prev_capacity / max_capacity
                # Calculate potential new balance.
                utilizations = []
                for other_id, other_info in selected_clients.items():
                    if other_id == client_id:
                        utilizations.append(new_utilization)
                    else:
                        other_capacities = client_capacities_map[other_id]
                        other_current = other_info["client_num_tasks_scheduled"]
                        other_max = other_capacities[-1]
                        utilizations.append(other_current / other_max if other_max > 0 else 0)
                new_balance = calculate_balance_metric(utilizations)
                balance_improvement = current_balance - new_balance  # Lower is better.
                candidates.append((client_id, balance_improvement, decrease, prev_capacity))
    if not candidates:
        return False
    # Prefer moves that improve balance the most.
    candidates.sort(key=lambda x: (-x[1], x[2]))  # High balance improvement, small decreases.
    for client_id, _, decrease, new_capacity in candidates:
        if decrease <= tasks_to_remove:
            selected_clients[client_id]["client_num_tasks_scheduled"] = new_capacity
            return True
    return False


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
