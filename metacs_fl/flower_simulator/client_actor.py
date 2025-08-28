from pathlib import Path
from ray import remote

from metacs_fl.client_launcher.flower_client_launcher import FlowerClientLauncher
from metacs_fl.utils.system_modeler_util import set_cpu_cores_affinity


@remote
class ClientActor:

    def __init__(self) -> None:
        self._client = None

    def initialize(self,
                   client_id: int,
                   client_config_file: Path,
                   client_personalized_settings: dict | None,
                   ray_node_shared_actors: dict | None) -> None:
        fcl = FlowerClientLauncher(client_id,
                                   client_config_file,
                                   client_personalized_settings,
                                   ray_node_shared_actors,
                                   instantiate_client=True)
        self._client = fcl.get_attribute("_client")

    def set_affinity(self,
                     acquired_cpu_cores: list) -> None:
        set_cpu_cores_affinity(acquired_cpu_cores)
        self._client.__dict__["numpy_client"].__dict__["_client_acquired_cpu_cores"] = acquired_cpu_cores

    def get_client(self) -> any:
        return self._client

    @staticmethod
    def ping() -> str:
        return "pong"
