from pathlib import Path
from ray import remote

from flwr.server import ServerAppComponents

from metacs_fl.server_launcher.flower_server_launcher import FlowerServerLauncher
from metacs_fl.utils.system_modeler_util import set_cpu_cores_affinity


@remote
class ServerActor:

    def __init__(self) -> None:
        self._server_strategy = None
        self._server_config = None

    def initialize(self,
                   server_id: int,
                   server_config_file: Path,
                   server_personalized_settings: dict | None,
                   ray_node_shared_actors: dict | None) -> None:
        fsl = FlowerServerLauncher(server_id,
                                   server_config_file,
                                   server_personalized_settings,
                                   ray_node_shared_actors,
                                   instantiate_server=False)
        self._server_strategy = fsl.get_attribute("_server_strategy")
        self._server_config = fsl.get_attribute("_server_config")

    @staticmethod
    def set_affinity(acquired_cpu_cores: list) -> None:
        set_cpu_cores_affinity(acquired_cpu_cores)

    def get_server_app_components(self) -> any:
        return ServerAppComponents(strategy=self._server_strategy, config=self._server_config)

    @staticmethod
    def ping() -> str:
        return "pong"
