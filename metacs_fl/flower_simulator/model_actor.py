from copy import deepcopy
from ray import remote

from metacs_fl.utils.model_loader_util import load_model


@remote
class ModelActor:

    def __init__(self) -> None:
        self._model = None
        self._metrics_names = None

    def load_model_for_client(self,
                              model_settings: dict,
                              learning_rate_schedule_settings: dict) -> tuple:
        if self._model is not None and self._metrics_names is not None:
            return deepcopy(self._model), self._metrics_names
        self._model, self._metrics_names = load_model(model_settings, learning_rate_schedule_settings)
        return deepcopy(self._model), self._metrics_names

    @staticmethod
    def ping() -> str:
        return "pong"
