from typing import List, Tuple

from flwr.common import FitRes, ndarrays_to_parameters, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate


def aggregate_parameters_with_fed_prox(global_parameters: Parameters,
                                       fit_results: List[Tuple[ClientProxy, FitRes]],
                                       mu: float) -> Parameters:
    """
    Aggregates client models using FedProx, which includes a proximal term to regularize
    the client updates towards the global model from the previous round.

    Args:
        global_parameters: The global model parameters from the previous round.
        fit_results: List of tuples (ClientProxy, FitRes), containing the client proxy and the corresponding
                     fit result (model parameters and number of examples for each client).
        mu: Proximal term regularization strength (FedProx hyperparameter).

    Returns:
        Parameters: The aggregated global model parameters after applying FedProx.
    """
    # Aggregation with FedProx.
    weighted_results = []
    # For each client, calculate the FedProx update...
    for _, fit_result in fit_results:
        # Get the client's model parameters and the number of training samples.
        client_parameters = parameters_to_ndarrays(fit_result.parameters)
        num_examples = fit_result.num_examples
        # Calculate the FedProx update: client update minus the global model parameters.
        client_update = [client_param - global_param for client_param, global_param in
                         zip(client_parameters, parameters_to_ndarrays(global_parameters))]
        # Apply the proximal term: client update + mu * (client update - global model).
        prox_update = [client_param - mu * global_param for client_param, global_param in
                       zip(client_update, parameters_to_ndarrays(global_parameters))]
        # Weight by the number of examples each client has.
        weighted_results.append((prox_update, num_examples))
    # Perform aggregation (weighted averaging).
    aggregated_ndarrays = aggregate(weighted_results)
    # Convert aggregated ndarray back to Parameters format.
    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
    return parameters_aggregated
