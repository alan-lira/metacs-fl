from copy import deepcopy
from math import floor, inf
from numpy import ndarray
from numpy.random import Generator
from time import perf_counter

from metacs_fl.utils.task_scheduler_util import calculate_percentage_change, estimate_costs, normalize_costs, \
    update_min_max_costs


def to_stop_lns(it: int,
                t_it: float,
                stop_criteria: dict) -> bool:
    stop_lns = False
    sa_name = stop_criteria["name"]
    match sa_name:
        case "LNS_Stop_Max_Iterations":
            # Check if the maximum number of iterations was met.
            it_max = stop_criteria["it_max"]
            stop_lns = (it > it_max)
        case "LNS_Stop_Elapsed_Time":
            # Check if the maximum elapsed time was met.
            t_max = stop_criteria["t_max"]
            stop_lns = (t_it > t_max)
    return stop_lns


def lns_destroy(rng: Generator,
                A: ndarray,
                X_curr: list,
                destroy_approach: dict) -> tuple:
    # Get the destruction configuration.
    sf = destroy_approach["sf"]
    rf_min = destroy_approach["rf_min"]
    rf_max = destroy_approach["rf_max"]
    # Initialize the destroyed solution (copy of X_curr).
    X_dest = deepcopy(X_curr)
    # Initialize the set of destroyed indices.
    dest_idx = []
    # Get client indices with tasks.
    I = [i for i in range(0, len(X_dest)) if X_dest[i] > 0]
    # Randomly sample a subset of client indices.
    I_spl = rng.choice(I, size=max(1, floor(sf * len(I))), replace=False)
    for i in I_spl:
        # Get the current number of tasks of the client i.
        x_i = X_dest[i]
        # Get a random number of tasks to remove from the client i.
        rmv_i = rng.integers(low=floor(rf_min * x_i), high=floor(rf_max * x_i) + 1)
        # Get the candidate new number of scheduled tasks to the client i.
        x_bar_i = x_i - min(rmv_i, x_i)
        # Set the new number of scheduled tasks to the client i, considering its previous largest valid assignment.
        X_dest[i] = max([a for a in A[i] if a <= x_bar_i])
        # Append the current index to the list of destroyed indices.
        dest_idx.append(i)
    # Return the destroyed solution and the list of destroyed indices.
    return X_dest, dest_idx


def safe_index(A_i: list,
               x_i: int) -> int | None:
    for idx, v in enumerate(A_i):
        if v == x_i:
            return idx
    return None


def prev_capacity(A_i: list,
                  x_i: int) -> int:
    pos = safe_index(A_i, x_i)
    if pos is None:
        prevs = [v for v in A_i if v < x_i]
        return max(prevs) if prevs else A_i[0]
    if pos > 0:
        return A_i[pos - 1]
    return A_i[0]


def next_capacity(A_i: list,
                  x_i: int) -> int:
    pos = safe_index(A_i, x_i)
    if pos is None:
        nexts = [v for v in A_i if v > x_i]
        return min(nexts) if nexts else A_i[-1]
    if pos < len(A_i) - 1:
        return A_i[pos + 1]
    return A_i[-1]


def lns_repair(rng: Generator,
               t: int,
               A: ndarray,
               X_dest: list,
               dest_indices: list,
               max_inner_iters: int = 10000) -> list:
    X_rpr = deepcopy(X_dest)
    # Get all client indices.
    I = list(range(len(X_rpr)))
    prev_sum = None
    stalled = 0
    stall_limit = 200
    for _ in range(max_inner_iters):
        t_asg = sum(X_rpr)
        if t_asg == t:
            return X_rpr
        # Detect stalling.
        if prev_sum is not None and t_asg == prev_sum:
            stalled += 1
        else:
            stalled = 0
        prev_sum = t_asg
        # Case 1: Too many tasks → remove.
        if t_asg > t:
            # Preferred: non-destroyed with removal capacity.
            cand = [i for i in I
                    if i not in dest_indices and
                    safe_index(A[i], X_rpr[i]) not in (None, 0)]
            # Fallback: any with removal capacity.
            if not cand:
                cand = [i for i in I
                        if safe_index(A[i], X_rpr[i]) not in (None, 0)]
            # Final fallback (stalled): any client with X>0.
            if not cand and stalled > stall_limit:
                cand = [i for i in I if X_rpr[i] > 0]
            if cand:
                i = rng.choice(cand, size=1, replace=False)[0]
                X_rpr[i] = prev_capacity(A[i], X_rpr[i])
            else:
                # No removal possible → break for best-effort return.
                break
        # Case 2: Too few tasks → add.
        else:
            # Preferred: non-destroyed with tasks and addition capacity.
            cand = [i for i in I
                    if i not in dest_indices and X_rpr[i] > 0 and
                    safe_index(A[i], X_rpr[i]) not in (None, len(A[i]) - 1)]
            # Fallback: non-destroyed with no tasks.
            if not cand:
                cand = [i for i in I
                        if i not in dest_indices and X_rpr[i] == 0 and
                        safe_index(A[i], X_rpr[i]) not in (None, len(A[i]) - 1)]
            # Fallback: destroyed with addition capacity.
            if not cand:
                cand = [i for i in I
                        if i in dest_indices and
                        safe_index(A[i], X_rpr[i]) not in (None, len(A[i]) - 1)]
            # Global fallback: any addition capacity.
            if not cand:
                cand = [i for i in I
                        if safe_index(A[i], X_rpr[i]) not in (None, len(A[i]) - 1)]
            # Final fallback if stalled: any client (may already be max, no-op).
            if not cand and stalled > stall_limit:
                cand = I[:]
            if cand:
                i = rng.choice(cand, size=1, replace=False)[0]
                X_rpr[i] = next_capacity(A[i], X_rpr[i])
            else:
                break
    # Exact repair failed... Return the best-effort solution (closest sum).
    t_asg = sum(X_rpr)
    diff = t_asg - t
    if diff > 0:
        # Remove diff tasks if possible.
        for _ in range(abs(diff)):
            removable = [i for i in I if safe_index(A[i], X_rpr[i]) not in (None, 0)]
            if not removable:
                break
            i = rng.choice(removable, size=1, replace=False)[0]
            X_rpr[i] = prev_capacity(A[i], X_rpr[i])
    elif diff < 0:
        # Add -diff tasks if possible.
        for _ in range(abs(diff)):
            addable = [i for i in I if safe_index(A[i], X_rpr[i]) not in (None, len(A[i]) - 1)]
            if not addable:
                break
            i = rng.choice(addable, size=1, replace=False)[0]
            X_rpr[i] = next_capacity(A[i], X_rpr[i])
    return X_rpr


def lns_accept(sol_costs: dict,
               accept_criteria: dict,
               tau: float,
               t: int,
               A: list) -> bool:
    acpt_res = []
    # Validate the 'client diversity rate percentage change' constraint.
    D_X_init = sol_costs["X_init"]["D_X"]
    D_X_rpr = sol_costs["X_rpr"]["D_X"]
    D_pc_min = accept_criteria["D_pc_min"]
    if 0 <= D_pc_min <= 1:
        D_pc_min *= 100
    D_pc = calculate_percentage_change(D_X_init, D_X_rpr)
    acpt_res.append(D_pc >= D_pc_min)
    # Validate the 'makespan percentage change' constraint.
    M_X_init = sol_costs["X_init"]["M_X"]
    M_X_rpr = sol_costs["X_rpr"]["M_X"]
    M_pc_max = accept_criteria["M_pc_max"]
    if 0 <= M_pc_max <= 1:
        M_pc_max *= 100
    M_pc = calculate_percentage_change(M_X_init, M_X_rpr)
    acpt_res.append(M_pc <= M_pc_max)
    # Validate the 'energy consumption percentage change' constraint.
    E_X_init = sol_costs["X_init"]["E_X"]
    E_X_rpr = sol_costs["X_rpr"]["E_X"]
    E_pc_max = accept_criteria["E_pc_max"]
    if 0 <= E_pc_max <= 1:
        E_pc_max *= 100
    E_pc = calculate_percentage_change(E_X_init, E_X_rpr)
    acpt_res.append(E_pc <= E_pc_max)
    # Validate the 'maximum makespan allowed' constraint.
    acpt_res.append(M_X_rpr <= tau)
    # Validate the 'remaining battery energy per client' constraint.
    B_X_rpr = sol_costs["X_rpr"]["B_X"]
    acpt_res.append(all(i >= accept_criteria["bl_min"] for i in B_X_rpr))
    # Validate the 'total number of tasks scheduled' constraint.
    X_rpr = sol_costs["X_rpr"]["X"]
    t_X_rpr = sum(X_rpr[i] for i in range(0, len(X_rpr)))
    acpt_res.append(t_X_rpr == t)
    # Validate the 'number of tasks scheduled per client' constraint.
    acpt_res.append(all(value in check_list for value, check_list in zip(X_rpr, A)))
    # Accept the solution only if all the constraints were met.
    to_accept = all(res for res in acpt_res)
    return to_accept


def lns_F(obj_func_costs: dict,
          obj_func_weights: dict) -> float:
    F_X = (obj_func_weights["M_weight"] * obj_func_costs["M_X"]) \
          + (obj_func_weights["E_weight"] * obj_func_costs["E_X"]) \
          - (obj_func_weights["D_weight"] * obj_func_costs["D_X"]) \
          - (obj_func_weights["KCov_weight"] * obj_func_costs["KCov_X"]) \
          + (obj_func_weights["KStd_weight"] * obj_func_costs["KStd_X"])
    return F_X


def run_lns(X_init: list,
            rng: Generator,
            destroy_approach: dict,
            n: int,
            tau: float,
            t: int,
            A: list,
            Y: list,
            I: list,
            B: list,
            G: list,
            E: list,
            stop_criteria: dict,
            accept_criteria: dict,
            obj_func_weights: dict,
            X_dist_approaches: dict)-> tuple:
    # Initialize the dict of min-max costs.
    min_max_costs = {"min_M_X": inf,
                     "max_M_X": 0,
                     "min_E_X": inf,
                     "max_E_X": 0}
    # Initialize the current and best solutions.
    X_curr = deepcopy(X_init)
    X_best = deepcopy(X_init)
    # Initialize the best solution costs.
    X_best_costs = {}
    # Initialize the iteration counter.
    it = 1
    # Get the start time.
    t_0 = perf_counter()
    # Initialize the elapsed time.
    t_it = 0
    while True:
        # Check the stopping criteria for the metaheuristic.
        to_stop_lns_execution = to_stop_lns(it, t_it, stop_criteria)
        if to_stop_lns_execution:
            # Stop the metaheuristic execution.
            break
        # LNS Block (Begin).
        # Destroy the current solution.
        X_dest, dest_indices = lns_destroy(rng, A, X_curr, destroy_approach)
        # Repair the destroyed solution.
        X_rpr = lns_repair(rng, t, A, X_dest, dest_indices)
        # Estimate costs.
        sol_costs = estimate_costs(n, t, A, Y, I, B, G, E, X_init, X_best, X_rpr, X_dist_approaches)
        # Update the min-max costs.
        min_max_costs = update_min_max_costs(sol_costs, min_max_costs)
        # Check if the repaired solution is acceptable.
        to_accept = lns_accept(sol_costs, accept_criteria, tau, t, A)
        if to_accept:
            # Update the current solution.
            X_curr = deepcopy(X_rpr)
            # Normalize the solution costs (particularly, M and E).
            norm_sol_costs = normalize_costs(sol_costs, min_max_costs)
            # Calculate the objective function value for X_rpr.
            F_X_rpr = lns_F(norm_sol_costs["X_rpr"], obj_func_weights)
            # Calculate the objective function value for X_best.
            F_X_best = lns_F(norm_sol_costs["X_best"], obj_func_weights)
            # Verify if the repaired solution is better than the best solution.
            if F_X_rpr < F_X_best:
                # Update the best solution.
                X_best = deepcopy(X_rpr)
                X_best_costs = deepcopy(sol_costs["X_rpr"])
        # LNS Block (End).
        # Update the iteration counter.
        it = it + 1
        # Update the elapsed time.
        t_it = perf_counter() - t_0
    # Set the LNS execution statistics.
    lns_statistics = {"time_lapsed": t_it, "num_iterations": it}
    # Return the best solution, the best solution costs, and the LNS execution statistics.
    return X_best, X_best_costs, lns_statistics
