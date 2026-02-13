from metacs_fl.utils.client_selector_util import can_schedule_to_all_clients, \
    select_random_fraction_available_clients, schedule_tasks_to_selected_clients


def random_selection(current_phase: str,
                     num_tasks: int,
                     candidate_clients: dict,
                     fraction_clients: float) -> list:
    # Select a random fraction of the candidate clients.
    selected_clients = select_random_fraction_available_clients(candidate_clients, current_phase, fraction_clients, num_tasks)
    # Verify if it's possible to schedule to all clients.
    schedule_to_all_clients = can_schedule_to_all_clients(num_tasks, selected_clients, current_phase)
    # Schedule tasks to all selected clients, respecting assignment capacities.
    selected_clients = schedule_tasks_to_selected_clients(num_tasks,
                                                          selected_clients,
                                                          current_phase,
                                                          schedule_to_all_clients=schedule_to_all_clients)
    # Build a client index map.
    candidate_idx = {client_id: idx for idx, client_id in enumerate(candidate_clients)}
    # Get the schedule provided by the 'Random' selection algorithm.
    random_schedule = [0] * len(candidate_clients)
    for client_id, client_info in selected_clients.items():
        idx = candidate_idx[client_id]
        random_schedule[idx] = client_info["client_num_tasks_scheduled"]
    return random_schedule
