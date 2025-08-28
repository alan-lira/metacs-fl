from numpy import zeros_like
from typing import List, Tuple

from flwr.common import FitRes, ndarrays_to_parameters, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def aggregate_parameters_with_fed_avg_e(fit_results: List[Tuple[ClientProxy, FitRes]]) -> Parameters:
    # Extract (parameters, weight) tuples using epoch-aware weights.
    weighted_results = []
    for _, fit_result in fit_results:
        local_parameters = parameters_to_ndarrays(fit_result.parameters)
        num_examples = fit_result.metrics["num_examples"]
        local_epochs = fit_result.metrics["epochs"]
        effort_weight = num_examples * local_epochs  # Reflects individual training effort.
        weighted_results.append((local_parameters, effort_weight))
    # Initialize aggregation with zeros.
    num_layers = len(weighted_results[0][0])
    aggregated_parameters_ndarrays = [zeros_like(layer) for layer in weighted_results[0][0]]
    total_weight = 0.0
    # Aggregate each layer using the effort weights.
    for local_parameters, effort_weight in weighted_results:
        for i in range(num_layers):
            aggregated_parameters_ndarrays[i] += effort_weight * local_parameters[i]
        total_weight += effort_weight
    # Normalize by the total weight to get the global model.
    for i in range(num_layers):
        aggregated_parameters_ndarrays[i] /= total_weight
    # Convert NumPy ndarrays to parameters object.
    aggregated_parameters = ndarrays_to_parameters(aggregated_parameters_ndarrays)
    return aggregated_parameters
