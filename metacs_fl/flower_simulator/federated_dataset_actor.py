from ray import remote

from metacs_fl.utils.dataset_loader_util import instantiate_fds, load_dataset


@remote
class FederatedDatasetActor:

    def __init__(self) -> None:
        self._dataset = {}

    def load_dataset_for_client(self,
                                client_id: int,
                                loading_approach: str,
                                local_dataset_settings: dict,
                                federated_dataset_settings: dict,
                                model_settings: dict) -> tuple:
        # If already loaded, return the cached dataset of this particular client.
        if client_id in self._dataset:
            return self._dataset[client_id]
        # Otherwise, load and cache the dataset for this particular client.
        fds = instantiate_fds(federated_dataset_settings)
        dataset_loading_dict = load_dataset(client_id,
                                            loading_approach,
                                            local_dataset_settings,
                                            federated_dataset_settings,
                                            model_settings,
                                            fds)
        x_train = dataset_loading_dict["x_train"]
        y_train = dataset_loading_dict["y_train"]
        x_test = dataset_loading_dict["x_test"]
        y_test = dataset_loading_dict["y_test"]
        dataset_loading_duration = dataset_loading_dict["dataset_loading_duration"]
        self._dataset[client_id] = (x_train, y_train, x_test, y_test)
        # Return the dataset for this particular client.
        return x_train, y_train, x_test, y_test, dataset_loading_duration

    @staticmethod
    def ping() -> str:
        return "pong"
