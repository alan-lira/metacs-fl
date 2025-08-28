from copy import deepcopy
from numpy.random import default_rng
from pathlib import Path
from psutil import cpu_count
from ray import get, init, is_initialized, remote, shutdown
from time import perf_counter
from flwr.client import Client, ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents
from flwr.simulation import run_simulation

from metacs_fl.client_launcher.flower_client_launcher import FlowerClientLauncher
from metacs_fl.devices.edge_devices import generate_edge_devices
from metacs_fl.networks.networks import generate_networks
from metacs_fl.server_launcher.flower_server_launcher import FlowerServerLauncher
from metacs_fl.utils.config_parser_util import parse_config_section, get_all_section_names
from metacs_fl.utils.cpu_cores_allocator_util import CPUCoresAllocator
from metacs_fl.utils.ray_util import assign_nodes_to_actors, create_client_actors, create_server_actor, \
    get_or_create_actor, get_personalized_settings_for_client, get_personalized_settings_for_server, \
    get_ray_head_node_address, kill_all_named_detached_actors, load_ray_shared_actors_per_node, get_alive_ray_nodes, \
    get_ray_node_id
from metacs_fl.utils.system_modeler_util import get_node_ip_address


class FlowerSimulator:

    def __init__(self,
                 config_file: Path) -> None:
        # Initialize the attributes.
        self._config_file = config_file
        self._current_simulation = {}
        self._current_simulation_devices = []
        self._rng = default_rng()
        self._ray_shared_actors_per_node = {}
        self._node_assignment_per_actor = []
        self._cpu_cores_allocators = {}
        self._ray_head_node_ip_address = get_node_ip_address()
        self._ray_head_node_address = get_ray_head_node_address()
        self._namespace = "simulator_namespace"

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    @remote
    def get_cpu_core_ids() -> list:
        core_id_lists = list(range(cpu_count(logical=False)))
        return core_id_lists

    def _load_cpu_cores_allocators(self) -> dict:
        cpu_cores_allocators = {}
        # Get the alive ray nodes.
        namespace = self.get_attribute("_namespace")
        alive_ray_nodes = get_alive_ray_nodes(namespace)
        for node in alive_ray_nodes:
            ray_node_id = node["node_id"]
            ray_node_ip = node["node_manager_address"]
            ray_node_cpu_core_ids = get(self.get_cpu_core_ids
                                        .options(resources={"node:{0}".format(ray_node_ip): 0.01})
                                        .remote())
            cpu_cores_allocator = CPUCoresAllocator()
            cpu_cores_allocator.initialize(ray_node_cpu_core_ids)
            cpu_cores_allocators.update({ray_node_id: cpu_cores_allocator})
        return cpu_cores_allocators

    @staticmethod
    def _append_client_resources_to_file(client_id: int,
                                         root_output_folder: Path,
                                         device_emulation_settings: dict,
                                         output_file: Path = Path("clients_resources.csv")) -> None:
        output_file = root_output_folder.joinpath(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        filtered_device_emulation_settings = {k: v for k, v in device_emulation_settings.items() if "," not in str(v)}
        # Write the header line (if not exist yet).
        if not output_file.is_file():
            header_line = "client_id" + "," + ",".join(list(filtered_device_emulation_settings.keys())) + "\n"
            with open(file=output_file, mode="a", encoding="utf-8") as o_f:
                o_f.write(header_line)
        # Write the data line.
        data_line = str(client_id) + "," + ",".join(map(str, filtered_device_emulation_settings.values())) + "\n"
        with open(file=output_file, mode="a", encoding="utf-8") as o_f:
           o_f.write(data_line)

    def _client_fn(self,
                   context: Context) -> Client:
        # Get the necessary attributes.
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        current_simulation = self.get_attribute("_current_simulation")
        current_simulation_devices = self.get_attribute("_current_simulation_devices")
        # Get the simulation name and settings.
        simulation_name = next(iter(current_simulation))
        simulation_settings = current_simulation[simulation_name]
        # Get the client id.
        client_id = partition_id
        # Get the client resources.
        device_emulation_settings = current_simulation_devices[client_id][1]
        client_num_cpu_cores_to_acquire = device_emulation_settings["cpu_num_cores"]
        client_cpu_frequency_min = device_emulation_settings["cpu_frequency_in_hertz"]
        client_cpu_frequency_max = device_emulation_settings["cpu_frequency_in_hertz"]
        client_cpu_frequency_unit = "Hz"
        # Get the namespace.
        namespace = self.get_attribute("_namespace")
        # Initialize the list of CPU cores acquired by the client.
        client_acquired_cpu_cores = []
        # Get the Ray actors settings.
        use_shared_actors_per_node = simulation_settings["use_shared_actors_per_node"]
        use_client_actors = simulation_settings["use_client_actors"]
        # Initialize the Ray node shared actors.
        ray_node_shared_actors = None
        try:
            if use_shared_actors_per_node:
                # Get the node assigned to this client.
                if use_client_actors:
                    node_assignment_per_actor = self.get_attribute("_node_assignment_per_actor")
                    node_id = node_assignment_per_actor[client_id]["node_id"]
                else:
                    node_id = get_ray_node_id(namespace)
                # Get the shared named actors of the node assigned to this client.
                ray_shared_actors_per_node = self.get_attribute("_ray_shared_actors_per_node")
                ray_node_shared_actors = ray_shared_actors_per_node[node_id]
                # Get the CPUCoresAllocatorActor (shared actor).
                cca_actor = ray_node_shared_actors["cca_actor"]
                # Acquire exclusive CPU cores, setting specific frequencies.
                client_acquired_cpu_cores = get(cca_actor.acquire_cpu_cores.remote(client_id,
                                                                                   client_num_cpu_cores_to_acquire,
                                                                                   client_cpu_frequency_min,
                                                                                   client_cpu_frequency_max,
                                                                                   client_cpu_frequency_unit))
            else:
                pass
                # # Get the CPUCoresAllocator (shared object).
                # ray_node_id = get_ray_node_id(namespace)
                # cpu_cores_allocators = self.get_attribute("_cpu_cores_allocators")
                # cpu_cores_allocator = cpu_cores_allocators[ray_node_id]
                # # Acquire exclusive CPU cores, setting specific frequencies.
                # client_acquired_cpu_cores = cpu_cores_allocator.acquire_cpu_cores(client_id,
                #                                                                   client_num_cpu_cores_to_acquire,
                #                                                                   client_cpu_frequency_min,
                #                                                                   client_cpu_frequency_max,
                #                                                                   client_cpu_frequency_unit)
            if use_client_actors:
                # Get the ClientActor corresponding to this client.
                client_actor_name = "client_actor_{0}".format(client_id)
                client_actor_name_namespace = self.get_attribute("_namespace")
                client_actor = get_or_create_actor(actor_name=client_actor_name,
                                                   actor_namespace=client_actor_name_namespace,
                                                   use_lock_actor=False)
                # Update the CPU cores affinity.
                get(client_actor.set_affinity.remote(client_acquired_cpu_cores))
                # Get the client.
                client = get(client_actor.get_client.remote())
            else:
                # Get the base client config file.
                base_client_config_file = Path(simulation_settings["base_client_config_file"])
                # Load the personalized_settings dictionary for this client.
                client_personalized_settings = get_personalized_settings_for_client(client_id,
                                                                                    base_client_config_file,
                                                                                    current_simulation,
                                                                                    current_simulation_devices)
                # Append the client resources to the 'clients_resources.csv' file.
                root_output_folder = Path(client_personalized_settings["_root_output_folder"])
                device_emulation_settings = client_personalized_settings["_device_emulation_settings"]
                self._append_client_resources_to_file(client_id, root_output_folder, device_emulation_settings)
                # Instantiate the flower client launcher.
                fcl = FlowerClientLauncher(client_id,
                                           base_client_config_file,
                                           client_personalized_settings,
                                           all_cpu_cores_available=None,
                                           client_acquired_cpu_cores=client_acquired_cpu_cores,
                                           ray_node_shared_actors=ray_node_shared_actors,
                                           instantiate_client=True)
                # Get the client.
                client = fcl.get_attribute("_client")
            # print("CLIENT_{0}: CPUS = {1}".format(client_id, client_acquired_cpu_cores))
            # Return the client.
            return client
        finally:
            if use_shared_actors_per_node:
                # Get the CPUCoresAllocatorActor (shared actor).
                cca_actor = ray_node_shared_actors["cca_actor"]
                # Release the exclusive CPU cores acquired, setting back the original frequencies.
                get(cca_actor.release_cpu_cores.remote(client_id))
            else:
                pass
                # # Get the CPUCoresAllocator (shared object).
                # ray_node_id = get_ray_node_id(namespace)
                # cpu_cores_allocators = self.get_attribute("_cpu_cores_allocators")
                # cpu_cores_allocator = cpu_cores_allocators[ray_node_id]
                # # Release the exclusive CPU cores acquired, setting back the original frequencies.
                # cpu_cores_allocator.release_cpu_cores(client_id)
            # print("CLIENT_{0}: RELEASED CPUS = {1}".format(client_id, client_acquired_cpu_cores))

    @staticmethod
    def _get_server_resources(simulation_settings: dict) -> dict:
        # Fallback values for the server resources (4 CPU Cores, 2000 MHz).
        server_resources = {"num_cpus": 4,
                            "cpu_frequency": 2000,
                            "cpu_frequency_unit": "MHz"}
        if "server_actor_resources" in simulation_settings:
            server_actor_resources = simulation_settings["server_actor_resources"]
            if "num_cpus" in server_actor_resources:
                num_cpus = server_actor_resources["num_cpus"]
                server_resources.update({"num_cpus": num_cpus})
            if "cpu_frequency" in server_actor_resources:
                cpu_frequency = server_actor_resources["cpu_frequency"]
                server_resources.update({"cpu_frequency": cpu_frequency})
            if "cpu_frequency_unit" in server_actor_resources:
                cpu_frequency_unit = server_actor_resources["cpu_frequency_unit"]
                server_resources.update({"cpu_frequency_unit": cpu_frequency_unit})
        return server_resources

    def _server_fn(self,
                   context: Context) -> ServerAppComponents:
        # Get the necessary attributes.
        current_simulation = self.get_attribute("_current_simulation")
        # Get the simulation name and settings.
        simulation_name = next(iter(current_simulation))
        simulation_settings = current_simulation[simulation_name]
        # Get the server id.
        server_id = simulation_settings["server_id"]
        # Get the server resources.
        server_resources = self._get_server_resources(simulation_settings)
        server_num_cpu_cores_to_acquire = server_resources["num_cpus"]
        server_cpu_frequency_min = server_resources["cpu_frequency"]
        server_cpu_frequency_max = server_resources["cpu_frequency"]
        server_cpu_frequency_unit = server_resources["cpu_frequency_unit"]
        # Get the namespace.
        namespace = self.get_attribute("_namespace")
        # Set the fictional server ID (to avoid ID overlap with clients when acquiring CPU cores).
        fictional_server_id = 999_999_999
        # Initialize the list of CPU cores acquired by the server.
        server_acquired_cpu_cores = []
        # Get the Ray actors settings.
        use_shared_actors_per_node = simulation_settings["use_shared_actors_per_node"]
        use_server_actor = simulation_settings["use_server_actor"]
        # Initialize the Ray node shared actors.
        ray_node_shared_actors = None
        if use_shared_actors_per_node:
            # Get the node assigned to this server.
            if use_server_actor:
                node_assignment_per_actor = self.get_attribute("_node_assignment_per_actor")
                # The last index is the convention for the ServerActor.
                node_id = node_assignment_per_actor[-1]["node_id"]
            else:
                node_id = get_ray_node_id(namespace)
            # Get the shared named actors of the node assigned to this server.
            ray_shared_actors_per_node = self.get_attribute("_ray_shared_actors_per_node")
            ray_node_shared_actors = ray_shared_actors_per_node[node_id]
            # Get the CPUCoresAllocatorActor (shared actor).
            cca_actor = ray_node_shared_actors["cca_actor"]
            # Acquire exclusive CPU cores, setting specific frequencies.
            server_acquired_cpu_cores = get(cca_actor.acquire_cpu_cores.remote(fictional_server_id,
                                                                               server_num_cpu_cores_to_acquire,
                                                                               server_cpu_frequency_min,
                                                                               server_cpu_frequency_max,
                                                                               server_cpu_frequency_unit))
        else:
            pass
            # # Get the CPUCoresAllocator (shared object).
            # ray_node_id = get_ray_node_id(namespace)
            # cpu_cores_allocators = self._load_cpu_cores_allocators()
            # cpu_cores_allocator = cpu_cores_allocators[ray_node_id]
            # # Acquire exclusive CPU cores, setting specific frequencies.
            # server_acquired_cpu_cores = cpu_cores_allocator.acquire_cpu_cores(fictional_server_id,
            #                                                                   server_num_cpu_cores_to_acquire,
            #                                                                   server_cpu_frequency_min,
            #                                                                   server_cpu_frequency_max,
            #                                                                   server_cpu_frequency_unit)
        if use_server_actor:
            # Get the ServerActor corresponding to this server.
            server_actor_name = "server_actor_{0}".format(fictional_server_id)
            server_actor = get_or_create_actor(actor_name=server_actor_name,
                                               actor_namespace=namespace,
                                               use_lock_actor=False)
            # Update the CPU affinity, without releasing cores acquired (persistent server throughout the execution).
            get(server_actor.set_affinity.remote(server_acquired_cpu_cores))
            # Get the server (ServerAppComponents).
            server_app_components = get(server_actor.get_server_app_components.remote())
        else:
            # Get the base server config file.
            base_server_config_file = Path(simulation_settings["base_server_config_file"])
            # Load the personalized_settings dictionary for the server.
            server_personalized_settings = get_personalized_settings_for_server(base_server_config_file, current_simulation)
            # Instantiate the flower server launcher.
            fsl = FlowerServerLauncher(server_id,
                                       base_server_config_file,
                                       server_personalized_settings,
                                       server_acquired_cpu_cores=server_acquired_cpu_cores,
                                       ray_node_shared_actors=ray_node_shared_actors,
                                       instantiate_server=True)
            # Get the server strategy.
            server_strategy = fsl.get_attribute("_server_strategy")
            # Get the server config.
            server_config = fsl.get_attribute("_server_config")
            # Get the server (ServerAppComponents).
            server_app_components = ServerAppComponents(strategy=server_strategy, config=server_config)
        # print("SERVER_{0}: CPUS = {1}".format(fictional_server_id, server_acquired_cpu_cores))
        # Return the server (ServerAppComponents).
        return server_app_components

    def execute_fl_with_flower_simulation_engine(self) -> None:
        # Set the list of simulations to execute.
        simulations_to_execute = []
        config_file = self.get_attribute("_config_file")
        section_names = get_all_section_names(config_file)
        for section_name in section_names:
            simulation_settings = parse_config_section(config_file, section_name)
            simulation_name = section_name.split(" Settings")[0]
            simulations_to_execute.append({simulation_name: simulation_settings})
        # Iterate through the list of simulations to execute.
        for current_simulation in simulations_to_execute:
            # Update the current simulation.
            self._set_attribute("_current_simulation", current_simulation)
            # Get the simulation name and settings.
            simulation_name = next(iter(current_simulation))
            simulation_settings = current_simulation[simulation_name]
            # Check if the devices performance emulation is enabled.
            emulate_devices_performance = simulation_settings["emulate_devices_performance"]
            if emulate_devices_performance:
                # Set the seed (base value used by the pseudo-random functions) to allow replicable analysis.
                if "seed" in simulation_settings:
                    rng_with_seed = default_rng(seed=simulation_settings["seed"])
                    self._set_attribute("_rng", rng_with_seed)
                # Generate edge devices.
                power_source_connection_scenarios = simulation_settings["power_source_connection_scenarios"]
                edge_devices = generate_edge_devices(power_source_connection_scenarios)
                # Initialize the dictionary of emulated devices.
                emulated_devices = {}
                # Get the device types to emulate.
                device_types_to_emulate = simulation_settings["device_types_to_emulate"]
                for device_type_to_emulate in device_types_to_emulate:
                    match device_type_to_emulate:
                        case "edge":
                            # Append all edge devices' performances.
                            emulated_devices = emulated_devices | edge_devices
                        case "edge_2_cores":
                            # Append only 2-CPU cores edge devices' performances.
                            edge_2_cores = {k: v for k, v in edge_devices.items() if v["cpu_num_cores"] == 2}
                            emulated_devices = emulated_devices | edge_2_cores
                        case "edge_4_cores":
                            # Append only 4-CPU cores edge devices' performances.
                            edge_4_cores = {k: v for k, v in edge_devices.items() if v["cpu_num_cores"] == 4}
                            emulated_devices = emulated_devices | edge_4_cores
                # Check if the random sampling of emulated devices is enabled.
                random_sampling_emulated_devices = simulation_settings["random_sampling_emulated_devices"]
                if random_sampling_emulated_devices:
                    # Check if it is needed to filter the emulated devices according to the clients' number of CPUs.
                    filter_devices_according_to_client_resources = simulation_settings["filter_devices_according_to_client_resources"]
                    if filter_devices_according_to_client_resources:
                        # Filter the candidate devices according to the number of CPUs to be used by the clients.
                        client_actors_resources = simulation_settings["client_actors_resources"]
                        simulation_num_cpus = client_actors_resources["num_cpus"]
                        emulated_devices = {k: v for k, v in emulated_devices.items()
                                            if v["cpu_num_cores"] == simulation_num_cpus}
                    # Get the number of clients.
                    num_supernodes = simulation_settings["num_supernodes"]
                    # Select randomly the performance for each client (allow repetition in the sampling).
                    rng = self.get_attribute("_rng")
                    emulated_devices_items = list(emulated_devices.items())
                    current_simulation_devices = []
                    for _ in range(num_supernodes):
                        device_name, device_info = rng.choice(emulated_devices_items)
                        device_info_copy = deepcopy(device_info)
                        current_simulation_devices.append([device_name, device_info_copy])
                    # Update the current simulation devices.
                    self._set_attribute("_current_simulation_devices", current_simulation_devices)
                else:
                    manual_emulated_devices_list = simulation_settings["manual_emulated_devices_list"] # TODO
                # Generate networks.
                network_quality_scenarios = simulation_settings["network_quality_scenarios"]
                networks = generate_networks(network_quality_scenarios)
                # Initialize the dictionary of emulated networks.
                emulated_networks = {}
                # Get the network types to emulate.
                network_types_to_emulate = simulation_settings["network_types_to_emulate"]
                for network_type_to_emulate in network_types_to_emulate:
                    match network_type_to_emulate:
                        case "3g":
                            # Append 3G network.
                            network_3g = {k: v for k, v in networks.items() if v["network_name"] == "3G"}
                            emulated_networks = emulated_networks | network_3g
                        case "4g_lte":
                            # Append 4G LTE network.
                            network_4g_lte = {k: v for k, v in networks.items() if v["network_name"] == "4G LTE"}
                            emulated_networks = emulated_networks | network_4g_lte
                        case "5g":
                            # Append 5G network.
                            network_5g = {k: v for k, v in networks.items() if v["network_name"] == "5G"}
                            emulated_networks = emulated_networks | network_5g
                        case "wifi":
                            # Append Wi-Fi network.
                            network_wifi = {k: v for k, v in networks.items() if v["network_name"] == "Wi-Fi"}
                            emulated_networks = emulated_networks | network_wifi
                        case "fixed_broadband":
                            # Append Fixed Broadband network.
                            network_fixed_broadband = {k: v for k, v in networks.items() if v["network_name"] == "Fixed Broadband"}
                            emulated_networks = emulated_networks | network_fixed_broadband
                # Check if there is a list of locations to sample to the clients.
                clients_locations_list = simulation_settings["clients_locations_list"]
                if clients_locations_list:
                    # Get the server location.
                    server_location = simulation_settings["server_location"]
                    # Get the current simulation devices.
                    rng = self.get_attribute("_rng")
                    current_simulation_devices = self.get_attribute("_current_simulation_devices")
                    for idx, _ in enumerate(current_simulation_devices):
                        # Sample the network performance.
                        network_key_sampled = rng.choice(list(emulated_networks.keys()), size=1, replace=False)[0]
                        network_details = emulated_networks[network_key_sampled]
                        current_simulation_devices[idx][1].update(network_details)
                        # Sample the geographic location.
                        client_location_sampled = rng.choice(clients_locations_list, size=1, replace=False)[0]
                        current_simulation_devices[idx][1]["server_location"] = server_location
                        current_simulation_devices[idx][1]["client_location"] = client_location_sampled
                    # Update the current simulation devices.
                    self._set_attribute("_current_simulation_devices", current_simulation_devices)
                # Check if there is a list of initial remaining battery level percentages to sample to the devices.
                initial_remaining_battery_level_percentage_list = simulation_settings["initial_remaining_battery_level_percentage_list"]
                if initial_remaining_battery_level_percentage_list:
                    # Get the current simulation devices.
                    rng = self.get_attribute("_rng")
                    current_simulation_devices = self.get_attribute("_current_simulation_devices")
                    for idx, _ in enumerate(current_simulation_devices):
                        current_simulation_device = current_simulation_devices[idx][1]
                        battery_maximum_stored_energy_in_joules = current_simulation_device["battery_maximum_stored_energy_in_joules"]
                        initial_battery_level_percentage_sampled = rng.choice(initial_remaining_battery_level_percentage_list, size=1, replace=False)[0]
                        initial_remaining_battery_energy_in_joules = battery_maximum_stored_energy_in_joules * initial_battery_level_percentage_sampled
                        current_simulation_devices[idx][1]["battery_stored_energy_in_joules"] = initial_remaining_battery_energy_in_joules
                    # Update the current simulation devices.
                    self._set_attribute("_current_simulation_devices", current_simulation_devices)
            # Get the simulation settings.
            num_supernodes = simulation_settings["num_supernodes"] if "num_supernodes" in simulation_settings else 1
            server_actor_resources = simulation_settings["server_actor_resources"]
            client_actors_resources = simulation_settings["client_actors_resources"]
            backend_name = simulation_settings["backend_name"] if "backend_name" in simulation_settings else "ray"
            backend_config = simulation_settings["backend_config"] if "backend_config" in simulation_settings else {}
            use_shared_actors_per_node = simulation_settings["use_shared_actors_per_node"]
            use_server_actor = simulation_settings["use_server_actor"]
            use_client_actors = simulation_settings["use_client_actors"]
            # Connect to an existing Ray cluster or start one and connect to it, if needed.
            ray_head_node_ip_address = self.get_attribute("_ray_head_node_ip_address")
            namespace = self.get_attribute("_namespace")
            if backend_name == "ray" and not is_initialized():
                try:
                    init(address="auto", namespace=namespace)
                    print("Connected to an existing Ray cluster!")
                except Exception as e:
                    print(e)
                    print("Starting a local Ray cluster instead...".format(e))
                    init(namespace=namespace)
            # Load Ray actors, if requested.
            if backend_name == "ray" and (use_server_actor or use_client_actors):
                # Print the 'assigning nodes to all Ray actors' message.
                print("\nAssigning nodes to all Ray actors (Server and Clients)...")
                # Assign node to all actors (resource-aware assignment).
                client_actors_cpu_demands = [client_actors_resources["num_cpus"]] * num_supernodes
                server_actor_cpu_demand = [server_actor_resources["num_cpus"]]
                all_actors_cpu_demands = client_actors_cpu_demands + server_actor_cpu_demand
                node_assignment_per_actor = assign_nodes_to_actors(ray_head_node_ip_address,
                                                                   namespace,
                                                                   all_actors_cpu_demands,
                                                                   allow_oversubscription=False)
                self._set_attribute("_node_assignment_per_actor", node_assignment_per_actor)
            # Load Ray shared actors per node, if requested.
            if backend_name == "ray" and use_shared_actors_per_node:
                # Print the 'loading the Ray shared actors per node' message.
                print("\nLoading the Ray shared actors per node...")
                ray_shared_actors_per_node = load_ray_shared_actors_per_node(ray_head_node_ip_address,
                                                                             namespace)
                self._set_attribute("_ray_shared_actors_per_node", ray_shared_actors_per_node)
            # Otherwise, only load the CPU cores allocators (non-Ray actors).
            else:
               # Print the 'loading the CPU cores allocators' message.
               print("\nLoading the CPU cores allocators...")
               cpu_cores_allocators = self._load_cpu_cores_allocators()
               self._set_attribute("_cpu_cores_allocators", cpu_cores_allocators)
            # Load Ray server actor, if requested.
            if backend_name == "ray" and use_server_actor:
                # Print the 'loading the Ray server actor' message.
                print("\nLoading the Ray server actor...")
                node_assignment_per_actor = self.get_attribute("_node_assignment_per_actor")
                ray_shared_actors_per_node = self.get_attribute("_ray_shared_actors_per_node")
                base_server_config_file = Path(simulation_settings["base_server_config_file"])
                # The last index is the convention for the ServerActor.
                node_assignment_for_server_actor = node_assignment_per_actor[-1]
                server_actors_cpu_demands = [server_actor_resources["num_cpus"]]
                create_server_actor(current_simulation,
                                    ray_head_node_ip_address,
                                    server_actors_cpu_demands,
                                    namespace,
                                    node_assignment_for_server_actor,
                                    ray_shared_actors_per_node,
                                    base_server_config_file)
            # Load Ray clients actors, if requested.
            if backend_name == "ray" and use_client_actors:
                # Print the 'loading the Ray client actors' message.
                print("\nLoading the Ray client actors...")
                node_assignment_per_actor = self.get_attribute("_node_assignment_per_actor")
                ray_shared_actors_per_node = self.get_attribute("_ray_shared_actors_per_node")
                base_client_config_file = Path(simulation_settings["base_client_config_file"])
                node_assignment_for_client_actors = deepcopy(node_assignment_per_actor)
                del node_assignment_for_client_actors[-1]  # Remove the server index.
                current_simulation_devices = self.get_attribute("_current_simulation_devices")
                client_actors_cpu_demands = [client_actors_resources["num_cpus"]] * num_supernodes
                create_client_actors(current_simulation,
                                     current_simulation_devices,
                                     ray_head_node_ip_address,
                                     client_actors_cpu_demands,
                                     namespace,
                                     node_assignment_for_client_actors,
                                     ray_shared_actors_per_node,
                                     base_client_config_file)
            # Create the ServerApp passing the server generation function.
            server_app = ServerApp(server_fn=self._server_fn)
            # Create the ClientApp passing the client generation function.
            client_app = ClientApp(client_fn=self._client_fn)
            # Print the start of the simulation.
            print("\nStarting the simulation '{0}'...".format(simulation_name))
            # Start the simulation timer.
            start = perf_counter()
            # Run the simulation (Flower's Simulation Engine).
            run_simulation(server_app=server_app,
                           client_app=client_app,
                           num_supernodes=num_supernodes,
                           backend_name=backend_name,
                           backend_config=backend_config)
            # End the simulation timer.
            end = perf_counter()
            # Print the elapsed time for the simulation.
            elapsed_time_seconds = round((end - start), 2)
            print("\nElapsed time of '{0}': {1} seconds".format(simulation_name, elapsed_time_seconds))
            # Kill all named detached actors, if any was loaded.
            if backend_name == "ray" and (use_shared_actors_per_node or use_server_actor or use_client_actors):
                kill_all_named_detached_actors(namespace)
            # Gracefully disconnect Ray and clean up resources, if needed.
            if backend_name == "ray":
                shutdown()
