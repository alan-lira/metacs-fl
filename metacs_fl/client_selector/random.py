from numpy.random import SeedSequence, default_rng
from random import sample

from metacs_fl.utils.logger_util import log_message


class Random:

    def __init__(self,
                 client_selection_settings: dict,
                 seed: None | int | SeedSequence) -> None:
        self._client_selection_settings = client_selection_settings
        # Initialize the random number generator with a fixed seed to allow replicable results.
        self._rng = default_rng(seed=seed)

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def run_client_selection_procedure(self,
                                       **kwargs) -> dict:
        # Get the necessary parameters.
        current_round = kwargs["current_round"]
        current_phase = kwargs["current_phase"]
        candidate_clients = kwargs["candidate_clients"]
        logger = kwargs["logger"]
        # Get the necessary attributes.
        client_selection_settings = self.get_attribute("_client_selection_settings")
        # Log a 'selecting clients' message.
        message = "[Random | Round {0}] Selecting {1}ing clients for round {0}..." \
                  .format(current_round, current_phase)
        log_message(logger, message, "INFO")
        # Select a random fraction of the candidate clients.
        fraction_clients = client_selection_settings["fraction_clients_{0}ing".format(current_phase)]
        selected_clients = {}
        if 0 < fraction_clients <= 1:
            num_available_clients = len(candidate_clients)
            num_clients_to_select = max(1, int(num_available_clients * fraction_clients))
            sampled_clients_keys = sample(sorted(candidate_clients), num_clients_to_select)
            for client_key in sampled_clients_keys:
                client_map = candidate_clients[client_key]
                client_proxy = client_map["client_proxy"]
                selected_clients.update({client_key: {"client_proxy": client_proxy}})
        # Return the set of selected clients.
        return selected_clients
