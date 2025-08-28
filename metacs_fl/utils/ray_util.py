from pathlib import Path
from ray import get, get_actor, init, is_initialized, kill, nodes
from ray.exceptions import RayActorError, RaySystemError
from ray.util import list_named_actors
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_actors
from ray.util.state.exception import ServerUnavailable
from ray._private.services import get_node_ip_address
from time import perf_counter, sleep

from metacs_fl.flower_simulator.client_actor import ClientActor
from metacs_fl.flower_simulator.cpu_cores_allocator_actor import CPUCoresAllocatorActor
from metacs_fl.flower_simulator.federated_dataset_actor import FederatedDatasetActor
from metacs_fl.flower_simulator.lock_actor import LockActor
from metacs_fl.flower_simulator.logger_actor import LoggerActor
from metacs_fl.flower_simulator.model_actor import ModelActor
from metacs_fl.flower_simulator.server_actor import ServerActor
from metacs_fl.utils.config_parser_util import parse_config_section
from metacs_fl.utils.system_modeler_util import get_cpu_cores_available


def get_ray_node_ip_address() -> str:
    ray_node_ip_address = get_node_ip_address()
    return ray_node_ip_address


def get_alive_ray_nodes(namespace: str | None) -> list:
    # Connect to an existing Ray cluster or start one and connect to it, if needed.
    if not is_initialized():
        try:
            init(address="auto", namespace=namespace)
            print("Connected to an existing Ray cluster!")
        except Exception as e:
            print(e)
            print("Starting a local Ray cluster instead...".format(e))
            init(namespace=namespace)
    try:
        ray_nodes = nodes()
    except RaySystemError as e:
        raise RaySystemError(e)
    alive_ray_nodes = [{"node_id": node["NodeID"],
                        "node_manager_address": node["NodeManagerAddress"],
                        "node_manager_hostname": node["NodeManagerHostname"],
                        "node_manager_port": node["NodeManagerPort"],
                        "object_manager_port": node["ObjectManagerPort"],
                        "object_store_socket_name": node["ObjectStoreSocketName"],
                        "raylet_socket_name": node["RayletSocketName"],
                        "metrics_export_port": node["MetricsExportPort"],
                        "node_name": node["NodeName"],
                        "runtime_env_agent_port": node["RuntimeEnvAgentPort"],
                        "resources": node["Resources"],
                        "labels": node["Labels"],
                        "num_cpus": int(node["Resources"].get("CPU", 0))}
                       for node in ray_nodes if node["Alive"] and int(node["Resources"].get("CPU", 0)) > 0]
    return alive_ray_nodes


def get_ray_node_id(namespace: str | None) -> str:
    current_ray_node_ip_address = get_ray_node_ip_address()
    # Get the alive ray nodes.
    alive_ray_nodes = get_alive_ray_nodes(namespace)
    ray_node_id = None
    for node in alive_ray_nodes:
        if node["node_manager_address"] == current_ray_node_ip_address:
            ray_node_id = node["node_id"]
            break
    return ray_node_id


def is_ray_head_node(ray_head_node_ip_address: str) -> bool:
    current_ray_node_ip_address = get_ray_node_ip_address()
    return current_ray_node_ip_address == ray_head_node_ip_address


def get_ray_head_node_address(ray_port: int = 6379) -> str:
    # Build the Ray head node's address string.
    ray_head_node_ip_address = get_node_ip_address()
    ray_head_node_address = "{0}:{1}".format(ray_head_node_ip_address, ray_port)
    # Return the Ray head node's address string.
    return ray_head_node_address


def assign_nodes_to_actors(head_node_ip_address: str,
                           namespace: str | None,
                           all_actors_cpu_demands: list,
                           allow_oversubscription: bool = False,
                           assignment_strategy: str = "greedy_max_available") -> list:
    node_assignment_per_actor = []
    # Get the alive ray nodes.
    alive_ray_nodes = get_alive_ray_nodes(namespace)
    if not alive_ray_nodes:
        raise RuntimeError("No alive Ray nodes with available CPU resources detected.")
    # Identify the head node and worker nodes.
    head_node = None
    worker_nodes = []
    for node in alive_ray_nodes:
        if node["node_manager_address"] == head_node_ip_address:
            head_node = node
        else:
            worker_nodes.append(node)
    if head_node is None:
        raise RuntimeError("Head node with IP {0} not found among alive nodes.".format(head_node_ip_address))
    # Initialize remaining CPUs.
    for node in worker_nodes + [head_node]:
        node["remaining_cpus"] = node["num_cpus"]
    server_cpu_demand = all_actors_cpu_demands[-1]
    if not allow_oversubscription and head_node["remaining_cpus"] < server_cpu_demand:
        raise RuntimeError("Insufficient resources on the head node for the server requiring {0} CPUs.".format(server_cpu_demand))
    # Prepare for round-robin index.
    rr_index = 0
    num_workers = len(worker_nodes)
    # Assign all clients (all except the last element).
    for idx, actor_cpu_demand in enumerate(all_actors_cpu_demands[:-1]):
        node = None
        # If there are no worker nodes, assign client actors to the head node (standalone execution).
        if num_workers == 0:
            node = head_node
            if not allow_oversubscription and node["remaining_cpus"] < actor_cpu_demand:
                raise RuntimeError("Insufficient resources on head node for client actor {0} requiring {1} CPUs."
                                   .format(idx, actor_cpu_demand))
        else:
            match assignment_strategy:
                case "greedy_max_available":
                    suitable_nodes = [n for n in worker_nodes
                                      if allow_oversubscription or n["remaining_cpus"] >= actor_cpu_demand]
                    if not suitable_nodes:
                        if allow_oversubscription:
                            node = max(worker_nodes, key=lambda n: n["remaining_cpus"])
                        else:
                            raise RuntimeError("Insufficient resources to schedule client actor {0} requiring {1} CPUs."
                                               .format(idx, actor_cpu_demand))
                    else:
                        node = max(suitable_nodes, key=lambda n: n["remaining_cpus"])
                case "round_robin":
                    node = worker_nodes[rr_index % num_workers]
                    rr_index += 1
                    if not allow_oversubscription and node["remaining_cpus"] < actor_cpu_demand:
                        raise RuntimeError("Insufficient resources on node {0} for client actor {1} requiring {2} CPUs."
                                           .format(node["node_manager_address"], idx, actor_cpu_demand))
                case "least_loaded":
                    suitable_nodes = [n for n in worker_nodes
                                      if allow_oversubscription or n["remaining_cpus"] >= actor_cpu_demand]
                    if not suitable_nodes:
                        if allow_oversubscription:
                            node = min(worker_nodes, key=lambda n: n["remaining_cpus"])
                        else:
                            raise RuntimeError("Insufficient resources to schedule client actor {0} requiring {1} CPUs."
                                               .format(idx, actor_cpu_demand))
                    else:
                        node = min(suitable_nodes, key=lambda n: n["remaining_cpus"])
        node_assignment_per_actor.append(node)
        if not allow_oversubscription:
            node["remaining_cpus"] -= actor_cpu_demand
    # Assign server actor to the head node.
    node_assignment_per_actor.append(head_node)
    if not allow_oversubscription:
        head_node["remaining_cpus"] -= server_cpu_demand
    # Return the list of node assignments.
    return node_assignment_per_actor


def safely_kill_actor(actor_name: str,
                      actor_namespace: str,
                      wait: bool = True,
                      timeout: float = 5.0) -> None:
    try:
        actor = get_actor(name=actor_name, namespace=actor_namespace)
        kill(actor=actor, no_restart=True)
    except ValueError:
        return  # Actor doesn't exist (already gone).
    if not wait:
        return
    try:
        # list_actors: only available for the head node.
        start = perf_counter()
        while perf_counter() - start < timeout:
            actors = list_actors(address=None, filters=[("name", "=", actor_name)])
            if not actors:
                return
            sleep(0.25)
    except ServerUnavailable:
        # On non-head nodes, fallback to a simple sleep.
        sleep(min(timeout, 2.0))


def get_or_create_actor(actor_name: str,
                        actor_namespace: str,
                        ray_head_node_ip_address: str = None,
                        actor_num_cpus: int = None,
                        actor_class: any = None,
                        actor_lifetime: str = None,
                        scheduling_strategy: any = None,
                        post_create_initialization: bool = False,
                        initialization_args: tuple = None,
                        initialization_kwargs: dict = None,
                        use_lock_actor: bool = True,
                        lock_timeout: float = None,
                        retries: int = 5,
                        initial_delay: float = 1.0,
                        backoff_factor: float = 2.0) -> any:
    lock_actor = None
    if use_lock_actor:
        # Get or create the LockActor (shared actor).
        lock_actor_name = "lock_{0}".format(actor_name)
        try:
            lock_actor = get_actor(name=lock_actor_name, namespace=actor_namespace)
        except ValueError:
            try:
                lock_actor = LockActor.options(name=lock_actor_name,
                                               namespace=actor_namespace,
                                               lifetime=actor_lifetime,
                                               scheduling_strategy=scheduling_strategy).remote()
            except ValueError:
                # Race condition: try to get the lock actor again if already created.
                lock_actor = get_actor(name=lock_actor_name, namespace=actor_namespace)
        while True:
            lock_acquired = get(lock_actor.acquire.remote(timeout=lock_timeout))
            if lock_acquired:
                break
            # Optional: log or monitor that we are still waiting.
            print("Waiting to acquire lock for actor '{0}'...".format(actor_name))
            sleep(0.1)
    # Check whether the current node is the head node.
    if ray_head_node_ip_address and not is_ray_head_node(ray_head_node_ip_address):
        # Non-head nodes can only get actors, not create them.
        actor = None
        delay = initial_delay
        for attempt in range(retries):
            try:
                actor = get_actor(name=actor_name, namespace=actor_namespace)
            except ValueError as e:
                if attempt < retries - 1:
                    sleep(delay)
                    # Exponential backoff.
                    delay *= backoff_factor
                else:
                    raise RuntimeError("Actor '{0}' not found in namespace '{1}' after {2} attempts."
                                       .format(actor_name, actor_namespace, retries)) from e
        return actor
    # If this is the head node or no address is provided, proceed with actor creation.
    try:
        # Attempt to get the actor.
        actor = None
        delay = initial_delay
        for attempt in range(retries):
            try:
                actor = get_actor(name=actor_name, namespace=actor_namespace)
            except ValueError as _:
                if attempt < retries - 1:
                    sleep(delay)
                    # Exponential backoff.
                    delay *= backoff_factor
                else:
                    # Safely kill the actor if it exists but isn't alive.
                    safely_kill_actor(actor_name, actor_namespace)
                    # Create a new actor.
                    actor = actor_class.options(name=actor_name,
                                                namespace=actor_namespace,
                                                lifetime=actor_lifetime,
                                                scheduling_strategy=scheduling_strategy).remote()
                    # Initialize the actor after creation if requested.
                    if post_create_initialization:
                        init_args = initialization_args or ()
                        init_kwargs = initialization_kwargs or {}
                        get(actor.initialize.remote(*init_args, **init_kwargs))
        # Return the actor object.
        return actor
    finally:
        if use_lock_actor:
            # Release the lock after actor creation.
            get(lock_actor.release.remote())


def load_ray_shared_actors_per_node(ray_head_node_ip_address: str,
                                    namespace: str | None) -> dict:
    if not is_ray_head_node(ray_head_node_ip_address):
        print("Not the head node, skipping the creation of the shared actors.")
        return {}
    # Initialize the dictionary of ray shared actors per node.
    ray_shared_actors_per_node = {}
    # Get the alive ray nodes.
    alive_ray_nodes = get_alive_ray_nodes(namespace)
    # Load the ray shared actors per node.
    for ray_node in alive_ray_nodes:
        # Get the ray node properties.
        node_id = ray_node["node_id"]
        node_manager_address = ray_node["node_manager_address"]
        # Set the scheduling strategy (ray node affinity).
        lifetime = "detached"
        soft_scheduling = False  # Place this actor on the specified ray node (node_id).
        same_node_scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=soft_scheduling)
        # CPUCoresAllocatorActor (shared actor).
        cca_actor_name = "cca_actor_{0}".format(node_manager_address)
        cca_actor_num_cpus = 1
        cpu_cores_available = get_cpu_cores_available()
        cca_actor_init_args = (cpu_cores_available,
                               "MHz",
                               "performance",
                               True,
                               namespace,
                               cca_actor_name,
                               lifetime,
                               same_node_scheduling_strategy)
        cca_actor = get_or_create_actor(cca_actor_name,
                                        namespace,
                                        ray_head_node_ip_address,
                                        cca_actor_num_cpus,
                                        CPUCoresAllocatorActor,
                                        lifetime,
                                        same_node_scheduling_strategy,
                                        post_create_initialization=True,
                                        initialization_args=cca_actor_init_args,
                                        use_lock_actor=True)
        # FederatedDatasetActor (shared actor).
        fds_actor_name = "fds_actor_{0}".format(node_manager_address)
        fds_actor_num_cpus = 1
        fds_actor = get_or_create_actor(fds_actor_name,
                                        namespace,
                                        ray_head_node_ip_address,
                                        fds_actor_num_cpus,
                                        FederatedDatasetActor,
                                        lifetime,
                                        same_node_scheduling_strategy,
                                        use_lock_actor=True)
        # ModelActor (shared actor, supposing that all clients use the same model architecture).
        model_actor_name = "model_actor_{0}".format(node_manager_address)
        model_actor_num_cpus = 1
        model_actor = get_or_create_actor(model_actor_name,
                                          namespace,
                                          ray_head_node_ip_address,
                                          model_actor_num_cpus,
                                          ModelActor,
                                          lifetime,
                                          same_node_scheduling_strategy,
                                          use_lock_actor=True)
        # Update the dictionary of ray shared actors per node.
        ray_shared_actors_per_node.update({node_id: {"cca_actor": cca_actor,
                                                     "fds_actor": fds_actor,
                                                     "model_actor": model_actor}})
    # Return the dictionary of ray shared actors per node.
    return ray_shared_actors_per_node


def get_personalized_settings_for_client(client_id: int,
                                         base_client_config_file: Path,
                                         simulation_dict: dict | None = None,
                                         current_simulation_devices: list | None = None) -> dict:
    # Initialize the personalized_settings dictionary.
    personalized_settings = {}
    if simulation_dict:
        # Get the simulation name and settings.
        simulation_name = next(iter(simulation_dict))
        simulation_settings = simulation_dict[simulation_name]
        # Set the simulation resources settings.
        simulation_num_cpus = simulation_settings["backend_config"]["client_resources"]["num_cpus"]
        simulation_num_gpus = simulation_settings["backend_config"]["client_resources"]["num_gpus"]
        simulation_resources_settings = {"simulation_num_cpus": simulation_num_cpus,
                                         "simulation_num_gpus": simulation_num_gpus}
        personalized_settings.update({"_simulation_resources_settings": simulation_resources_settings})
        # Get the simulation output folder.
        simulation_output_folder = simulation_settings["simulation_output_folder"]
        # Set the root output folder.
        personalized_settings.update({"_root_output_folder": simulation_output_folder})
        # Set the simulation output folder as the parent folder for the logging file.
        logging_settings = parse_config_section(base_client_config_file, "Logging Settings")
        logging_settings["file_name"] = simulation_output_folder + "/" + logging_settings["file_name"]
        personalized_settings.update({"_logging_settings": logging_settings})
        # Set the number of partitions based on the number of supernodes (same value for all clients).
        num_partitions = simulation_settings["num_supernodes"]
        personalized_settings.update({"_federated_dataset_settings": {"num_partitions": num_partitions}})
    # Set the device emulation settings.
    if current_simulation_devices:
        device_emulation_settings = current_simulation_devices[client_id][1]
        personalized_settings.update({"_device_emulation_settings": device_emulation_settings})
    # Return the personalized_settings dictionary.
    return personalized_settings


def create_client_actors(current_simulation: dict,
                         current_simulation_devices: list,
                         ray_head_node_ip_address: str,
                         client_actors_cpu_demands: list,
                         namespace: str | None,
                         node_assignment_for_client_actors: list,
                         ray_shared_actors_per_node: dict,
                         base_client_config_file: Path) -> None:
    if not is_ray_head_node(ray_head_node_ip_address):
        print("Not the head node, skipping the creation of the client actors.")
    for client_id, ray_node in enumerate(node_assignment_for_client_actors):
        # Load the personalized_settings dictionary for this ClientActor.
        client_personalized_settings = get_personalized_settings_for_client(client_id,
                                                                            base_client_config_file,
                                                                            current_simulation,
                                                                            current_simulation_devices)
        # Get the ray node properties.
        node_id = ray_node["node_id"]
        ray_node_shared_actors = ray_shared_actors_per_node[node_id]
        # Set the scheduling strategy (ray node affinity).
        lifetime = "detached"
        soft_scheduling = False  # Place this actor on the specified ray node (node_id).
        same_node_scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=soft_scheduling)
        # Get or create the LoggerActor for this ClientActor.
        logging_settings = client_personalized_settings["_logging_settings"]
        # Append the client's id to the output file name.
        file_name = Path(logging_settings["file_name"]).absolute()
        file_name = str(file_name.parent.joinpath(file_name.stem + "_{0}".format(client_id) + file_name.suffix))
        logging_settings["file_name"] = file_name
        logger_name = "Client_{0}".format(client_id) + "_Logger"
        logger_actor_name = "logger_actor_client_{0}".format(client_id)
        logger_actor_num_cpus = 1
        logger_actor_init_args = (logging_settings, logger_name)
        client_logger_actor = get_or_create_actor(logger_actor_name,
                                                  namespace,
                                                  ray_head_node_ip_address,
                                                  logger_actor_num_cpus,
                                                  LoggerActor,
                                                  lifetime,
                                                  same_node_scheduling_strategy,
                                                  post_create_initialization=True,
                                                  initialization_args=logger_actor_init_args,
                                                  use_lock_actor=False)
        # Set the ray node shared actors for this ClientActor.
        ray_node_shared_actors.update({"client_logger_actor": client_logger_actor})
        # Get or create the ClientActor.
        client_actor_name = "client_actor_{0}".format(client_id)
        client_actor_num_cpus = client_actors_cpu_demands[client_id]
        client_actor_init_args = (client_id,
                                  base_client_config_file,
                                  client_personalized_settings,
                                  ray_node_shared_actors)
        _ = get_or_create_actor(client_actor_name,
                                namespace,
                                ray_head_node_ip_address,
                                client_actor_num_cpus,
                                ClientActor,
                                lifetime,
                                same_node_scheduling_strategy,
                                post_create_initialization=True,
                                initialization_args=client_actor_init_args,
                                use_lock_actor=False)


def get_personalized_settings_for_server(base_server_config_file: Path,
                                         simulation_dict: dict | None = None) -> dict:
    # Initialize the personalized_settings dictionary.
    personalized_settings = {}
    if simulation_dict:
        # Get the simulation name and settings.
        simulation_name = next(iter(simulation_dict))
        simulation_settings = simulation_dict[simulation_name]
        # Get the simulation output folder.
        simulation_output_folder = simulation_settings["simulation_output_folder"]
        # Set the root output folder.
        personalized_settings.update({"_root_output_folder": simulation_output_folder})
        # Set the simulation output folder as the parent folder for the logging file.
        logging_settings = parse_config_section(base_server_config_file, "Logging Settings")
        logging_settings["file_name"] = simulation_output_folder + "/" + logging_settings["file_name"]
        personalized_settings.update({"_logging_settings": logging_settings})
        # Set the simulation output folder as the parent folder for all output files.
        output_settings = parse_config_section(base_server_config_file, "Output Settings")
        output_file_keys_to_update = []
        for k, v in output_settings.items():
            if isinstance(v, str) and ".csv" in v:
                output_file_keys_to_update.append(k)
        for k in output_file_keys_to_update:
            output_settings[k] = simulation_output_folder + "/" + output_settings[k]
        personalized_settings.update({"_output_settings": output_settings})
        # Set the number of clients to wait based on the number of supernodes.
        num_clients_to_wait = simulation_settings["num_supernodes"]
        fl_settings = parse_config_section(base_server_config_file, "FL Settings")
        fl_settings["wait_for_initial_clients"]["num_clients_to_wait"] = num_clients_to_wait
        personalized_settings.update({"_fl_settings": fl_settings})
    # Return the personalized_settings dictionary.
    return personalized_settings


def create_server_actor(current_simulation: dict,
                        ray_head_node_ip_address: str,
                        server_actors_cpu_demands: list,
                        namespace: str | None,
                        ray_node_for_server_actor: any,
                        ray_shared_actors_per_node: dict,
                        base_server_config_file: Path) -> None:
    if not is_ray_head_node(ray_head_node_ip_address):
        print("Not the head node, skipping the creation of the server actor.")
    # Load the personalized_settings dictionary for this ServerActor.
    simulation_name = next(iter(current_simulation))
    simulation_settings = current_simulation[simulation_name]
    server_id = simulation_settings["server_id"]
    fictional_server_id = 999_999_999  # Fictional server ID (to avoid ID overlap with clients when acquiring CPU cores).
    server_personalized_settings = get_personalized_settings_for_server(base_server_config_file,
                                                                        current_simulation)
    # Get the ray node properties.
    node_id = ray_node_for_server_actor["node_id"]
    ray_node_shared_actors = ray_shared_actors_per_node[node_id]
    # Set the scheduling strategy (ray node affinity).
    lifetime = "detached"
    soft_scheduling = False  # Place this actor on the specified ray node (node_id).
    same_node_scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=soft_scheduling)
    # Get or create the LoggerActor for this ServerActor.
    logging_settings = server_personalized_settings["_logging_settings"]
    # Append the server's id to the output file name.
    file_name = Path(logging_settings["file_name"]).absolute()
    file_name = str(file_name.parent.joinpath(file_name.stem + "_{0}".format(server_id) + file_name.suffix))
    logging_settings["file_name"] = file_name
    logger_name = "Server_{0}".format(server_id) + "_Logger"
    logger_actor_name = "logger_actor_server_{0}".format(server_id)
    logger_actor_num_cpus = 1
    logger_actor_init_args = (logging_settings, logger_name)
    server_logger_actor = get_or_create_actor(logger_actor_name,
                                              namespace,
                                              ray_head_node_ip_address,
                                              logger_actor_num_cpus,
                                              LoggerActor,
                                              lifetime,
                                              same_node_scheduling_strategy,
                                              post_create_initialization=True,
                                              initialization_args=logger_actor_init_args,
                                              use_lock_actor=False)
    # Set the ray node shared actors for this ServerActor.
    ray_node_shared_actors.update({"server_logger_actor": server_logger_actor})
    # Get or create the ServerActor.
    server_actor_name = "server_actor_{0}".format(fictional_server_id)
    server_actor_num_cpus = server_actors_cpu_demands[server_id]
    server_actor_init_args = (server_id,
                              base_server_config_file,
                              server_personalized_settings,
                              ray_node_shared_actors)
    _ = get_or_create_actor(server_actor_name,
                            namespace,
                            ray_head_node_ip_address,
                            server_actor_num_cpus,
                            ServerActor,
                            lifetime,
                            same_node_scheduling_strategy,
                            post_create_initialization=True,
                            initialization_args=server_actor_init_args,
                            use_lock_actor=False)


def kill_all_named_detached_actors(namespace: str | None) -> None:
    named_actors = list_named_actors(all_namespaces=True)
    for actor_info in named_actors:
        actor_name = actor_info["name"]
        actor_namespace = actor_info["namespace"]
        if actor_namespace == namespace:
            try:
                actor = get_actor(name=actor_name, namespace=namespace)
                kill(actor)
            except (ValueError, RayActorError):
                pass
