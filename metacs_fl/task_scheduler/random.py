from metacs_fl.utils.client_selector_util import select_random_fraction_available_clients, \
    schedule_tasks_to_selected_clients


def random_selection(current_phase: str,
                     num_tasks: int,
                     candidate_clients: dict,
                     fraction_clients: float) -> list:
    # Select a random fraction of the candidate clients.
    selected_clients = select_random_fraction_available_clients(candidate_clients, current_phase, fraction_clients, num_tasks)
    # Schedule tasks to all selected clients, respecting assignment capacities.
    selected_clients = schedule_tasks_to_selected_clients(num_tasks,
                                                          selected_clients,
                                                          current_phase,
                                                          profiling_round=False,
                                                          schedule_to_all_clients=True)
    # Get the schedule provided by the 'Random' selection algorithm.
    random_schedule = []
    for client_id, client_info in selected_clients.items():
        client_num_tasks_scheduled = client_info["client_num_tasks_scheduled"]
        random_schedule.append(client_num_tasks_scheduled)
    return random_schedule
