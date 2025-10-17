from copy import deepcopy
from multiprocessing import Barrier, Process
from numpy.random import default_rng
from pathlib import Path
from time import perf_counter, sleep
from threading import BrokenBarrierError
from traceback import format_exc

from metacs_fl.client_launcher.flower_client_launcher import FlowerClientLauncher
from metacs_fl.devices.edge_devices import generate_edge_devices
from metacs_fl.networks.networks import generate_networks
from metacs_fl.server_launcher.flower_server_launcher import FlowerServerLauncher
from metacs_fl.utils.config_parser_util import parse_config_section, get_all_section_names
from metacs_fl.utils.system_modeler_util import get_cpu_cores_available


class FlowerExecutor:

    def __init__(self,
                 config_file: Path) -> None:
        # Initialize the attributes.
        self._config_file = config_file
        self._current_execution = {}
        self._current_execution_devices = []
        self._rng = default_rng()
        self._dataset_loaded_barrier = None

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    def _get_personalized_settings_for_server(base_server_config_file: Path,
                                              execution_dict: dict | None = None) -> dict:
        # Initialize the personalized_settings dictionary.
        personalized_settings = {}
        if execution_dict:
            # Get the execution name and settings.
            execution_name = next(iter(execution_dict))
            execution_settings = execution_dict[execution_name]
            # Get the execution output folder.
            execution_output_folder = execution_settings["execution_output_folder"]
            # Set the root output folder.
            personalized_settings.update({"_root_output_folder": execution_output_folder})
            # Set the execution output folder as the parent folder for the logging file.
            logging_settings = parse_config_section(base_server_config_file, "Logging Settings")
            logging_settings["file_name"] = execution_output_folder + "/" + logging_settings["file_name"]
            personalized_settings.update({"_logging_settings": logging_settings})
            # Set the execution output folder as the parent folder for all output files.
            output_settings = parse_config_section(base_server_config_file, "Output Settings")
            output_file_keys_to_update = []
            for k, v in output_settings.items():
                if isinstance(v, str) and ".csv" in v:
                    output_file_keys_to_update.append(k)
            for k in output_file_keys_to_update:
                output_settings[k] = execution_output_folder + "/" + output_settings[k]
            personalized_settings.update({"_output_settings": output_settings})
            # Set the number of clients to wait based on the number of clients.
            num_clients_to_wait = execution_settings["num_clients"]
            fl_settings = parse_config_section(base_server_config_file, "FL Settings")
            fl_settings["wait_for_initial_clients"]["num_clients_to_wait"] = num_clients_to_wait
            personalized_settings.update({"_fl_settings": fl_settings})
        # Return the personalized_settings dictionary.
        return personalized_settings

    @staticmethod
    def _get_personalized_settings_for_client(client_id: int,
                                              base_client_config_file: Path,
                                              execution_dict: dict | None = None,
                                              current_execution_devices: list | None = None) -> dict:
        # Initialize the personalized_settings dictionary.
        personalized_settings = {}
        if execution_dict:
            # Get the execution name and settings.
            execution_name = next(iter(execution_dict))
            execution_settings = execution_dict[execution_name]
            # Get the execution output folder.
            execution_output_folder = execution_settings["execution_output_folder"]
            # Set the root output folder.
            personalized_settings.update({"_root_output_folder": execution_output_folder})
            # Set the execution output folder as the parent folder for the logging file.
            logging_settings = parse_config_section(base_client_config_file, "Logging Settings")
            logging_settings["file_name"] = execution_output_folder + "/" + logging_settings["file_name"]
            personalized_settings.update({"_logging_settings": logging_settings})
            # Set the number of partitions based on the number of clients (same value for all clients).
            num_partitions = execution_settings["num_clients"]
            personalized_settings.update({"_federated_dataset_settings": {"num_partitions": num_partitions}})
        # Set the device emulation settings.
        if current_execution_devices:
            device_emulation_settings = current_execution_devices[client_id][1]
            personalized_settings.update({"_device_emulation_settings": device_emulation_settings})
        # Return the personalized_settings dictionary.
        return personalized_settings

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

    def _launch_flower_server(self,
                              server_id: int) -> FlowerServerLauncher:
        # Get the necessary attributes.
        current_execution = self.get_attribute("_current_execution")
        # Get the execution name and settings.
        execution_name = next(iter(current_execution))
        execution_settings = current_execution[execution_name]
        # Get the base server config file.
        base_server_config_file = Path(execution_settings["base_server_config_file"])
        # Load the personalized_settings dictionary for the server.
        server_personalized_settings = self._get_personalized_settings_for_server(base_server_config_file,
                                                                                  current_execution)
        # Instantiate the flower server launcher.
        cpu_cores_available = get_cpu_cores_available()
        fsl = FlowerServerLauncher(server_id,
                                   base_server_config_file,
                                   server_personalized_settings,
                                   server_acquired_cpu_cores=cpu_cores_available,
                                   instantiate_server=True)
        # Return the flower server launcher.
        return fsl

    @staticmethod
    def _verify_dataset_loaded(fcl: FlowerClientLauncher,
                               max_retries: int = 10,
                               retry_delay: float = 30) -> bool:
        client_id = fcl._client_id
        for attempt in range(max_retries):
            try:
                # Check if the main dataset attributes exist and are not None.
                if (hasattr(fcl, '_x_train') and fcl._x_train is not None and
                    hasattr(fcl, '_y_train') and fcl._y_train is not None and
                    hasattr(fcl, '_x_test') and fcl._x_test is not None and
                    hasattr(fcl, '_y_test') and fcl._y_test is not None):
                    # Additional checks for non-empty datasets.
                    if (len(fcl._x_train) > 0 and len(fcl._y_train) > 0 and
                        len(fcl._x_test) > 0 and len(fcl._y_test) > 0):
                        print("Client {0} has successfully loaded the dataset (attempt {1}/{2})"
                              .format(client_id, attempt + 1, max_retries))
                        return True
                    else:
                        print("Client {0} has dataset variables but they are empty (attempt {1}/{2})"
                              .format(client_id, attempt + 1, max_retries))
                else:
                    print("Client {0} has None dataset variables (attempt {1}/{2})"
                          .format(client_id, attempt + 1, max_retries))
            except Exception as e:
                print("Failed to verify dataset for client {0} (attempt {1}/{2}): {3}"
                      .format(client_id, attempt + 1, max_retries, e))
            # Wait before retrying.
            if attempt < max_retries - 1:
                print("Retrying dataset verification for client {0} in {1} seconds..."
                      .format(client_id, retry_delay))
                sleep(retry_delay)
            else:
                print("Client {0} failed to load dataset after {1} attempts!"
                      .format(client_id, max_retries))
        return False

    def _launch_flower_client(self,
                              client_id: int,
                              dataset_loaded_barrier: Barrier = None) -> FlowerClientLauncher:
        # Get the necessary attributes.
        current_execution = self.get_attribute("_current_execution")
        current_execution_devices = self.get_attribute("_current_execution_devices")
        # Get the execution name and settings.
        execution_name = next(iter(current_execution))
        execution_settings = current_execution[execution_name]
        # Get the base client config file.
        base_client_config_file = Path(execution_settings["base_client_config_file"])
        # Load the personalized_settings dictionary for this client.
        client_personalized_settings = self._get_personalized_settings_for_client(client_id,
                                                                                  base_client_config_file,
                                                                                  current_execution,
                                                                                  current_execution_devices)
        # Append the client resources to the 'clients_resources.csv' file.
        if "_device_emulation_settings" in client_personalized_settings:
            root_output_folder = Path(client_personalized_settings["_root_output_folder"])
            device_emulation_settings = client_personalized_settings["_device_emulation_settings"]
            self._append_client_resources_to_file(client_id, root_output_folder, device_emulation_settings)
        # Instantiate the flower client launcher.
        cpu_cores_available = get_cpu_cores_available()
        fcl = FlowerClientLauncher(client_id,
                                   base_client_config_file,
                                   client_personalized_settings,
                                   client_acquired_cpu_cores=cpu_cores_available,
                                   instantiate_client=True)
        # Ensure dataset is loaded and verified before proceeding.
        if dataset_loaded_barrier is not None:
            # Verify that dataset was fully loaded with retry mechanism.
            dataset_loaded = self._verify_dataset_loaded(fcl)
            if dataset_loaded:
                print("Client {0} has successfully loaded the dataset and is ready!".format(client_id))
                # Signal that this client has loaded its dataset.
                dataset_loaded_barrier.wait()
            else:
                print("Client {0} failed to load dataset properly after all retries!".format(client_id))
                # Still signal the barrier to avoid deadlock, but with critical warning.
                dataset_loaded_barrier.wait()
        # Return the flower client launcher.
        return fcl

    def _launch_flower_client_safely(self,
                                     client_id: int,
                                     dataset_loaded_barrier: Barrier) -> None:
        try:
            client_launcher = self._launch_flower_client(client_id, dataset_loaded_barrier)
            # Only launch the client if dataset was verified (though barrier will be passed regardless).
            client_launcher.launch_client()
        except Exception as e:
            print("Error in client {0}: {1}".format(client_id, e))
            print(format_exc())
            try:
                dataset_loaded_barrier.wait()
            except BrokenBarrierError:
                print("[Client {0}] Barrier broken.".format(client_id))
            except Exception as e2:
                print("[Client {0}] Barrier wait failed: {1}".format(client_id, e2))
                print(format_exc())

    def execute_fl_with_flower(self) -> None:
        executions_to_execute = []
        config_file = self.get_attribute("_config_file")
        section_names = get_all_section_names(config_file)
        for section_name in section_names:
            execution_settings = parse_config_section(config_file, section_name)
            execution_name = section_name.split(" Settings")[0]
            executions_to_execute.append({execution_name: execution_settings})
        # Iterate through the list of executions to execute.
        for current_execution in executions_to_execute:
            # Update the current execution.
            self._set_attribute("_current_execution", current_execution)
            # Get the execution name and settings.
            execution_name = next(iter(current_execution))
            execution_settings = current_execution[execution_name]
            # Check if the devices performance emulation is enabled.
            emulate_devices_performance = execution_settings["emulate_devices_performance"]
            if emulate_devices_performance:
                # Set the seed (base value used by the pseudo-random functions) to allow replicable analysis.
                if "seed" in execution_settings:
                    rng_with_seed = default_rng(seed=execution_settings["seed"])
                    self._set_attribute("_rng", rng_with_seed)
                # Generate edge devices.
                power_source_connection_scenarios = execution_settings["power_source_connection_scenarios"]
                edge_devices = generate_edge_devices(power_source_connection_scenarios)
                # Initialize the dictionary of emulated devices.
                emulated_devices = {}
                # Get the device types to emulate.
                device_types_to_emulate = execution_settings["device_types_to_emulate"]
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
                random_sampling_emulated_devices = execution_settings["random_sampling_emulated_devices"]
                if random_sampling_emulated_devices:
                    # Check if it is needed to filter the emulated devices according to the clients' number of CPUs.
                    filter_devices_according_to_client_resources = execution_settings["filter_devices_according_to_client_resources"]
                    if filter_devices_according_to_client_resources:
                        # Filter the candidate devices according to the number of CPUs to be used by the clients.
                        client_resources = execution_settings["client_resources"]
                        execution_num_cpus = client_resources["num_cpus"]
                        emulated_devices = {k: v for k, v in emulated_devices.items()
                                            if v["cpu_num_cores"] == execution_num_cpus}
                    # Get the number of clients.
                    num_clients = execution_settings["num_clients"]
                    # Select randomly the performance for each client (allow repetition in the sampling).
                    rng = self.get_attribute("_rng")
                    emulated_devices_items = list(emulated_devices.items())
                    current_execution_devices = []
                    for _ in range(num_clients):
                        device_name, device_info = rng.choice(emulated_devices_items)
                        device_info_copy = deepcopy(device_info)
                        current_execution_devices.append([device_name, device_info_copy])
                    # Update the current execution devices.
                    self._set_attribute("_current_execution_devices", current_execution_devices)
                else:
                    manual_emulated_devices_list = execution_settings["manual_emulated_devices_list"] # TODO
                # Generate networks.
                network_quality_scenarios = execution_settings["network_quality_scenarios"]
                networks = generate_networks(network_quality_scenarios)
                # Initialize the dictionary of emulated networks.
                emulated_networks = {}
                # Get the network types to emulate.
                network_types_to_emulate = execution_settings["network_types_to_emulate"]
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
                clients_locations_list = execution_settings["clients_locations_list"]
                if clients_locations_list:
                    # Get the server location.
                    server_location = execution_settings["server_location"]
                    # Get the current execution devices.
                    rng = self.get_attribute("_rng")
                    current_execution_devices = self.get_attribute("_current_execution_devices")
                    for idx, _ in enumerate(current_execution_devices):
                        # Sample the network performance.
                        network_key_sampled = rng.choice(list(emulated_networks.keys()), size=1, replace=False)[0]
                        network_details = emulated_networks[network_key_sampled]
                        current_execution_devices[idx][1].update(network_details)
                        # Sample the geographic location.
                        client_location_sampled = rng.choice(clients_locations_list, size=1, replace=False)[0]
                        current_execution_devices[idx][1]["server_location"] = server_location
                        current_execution_devices[idx][1]["client_location"] = client_location_sampled
                    # Update the current execution devices.
                    self._set_attribute("_current_execution_devices", current_execution_devices)
                # Check if there is a list of initial remaining battery level percentages to sample to the devices.
                initial_remaining_battery_level_percentage_list = execution_settings["initial_remaining_battery_level_percentage_list"]
                if initial_remaining_battery_level_percentage_list:
                    # Get the current execution devices.
                    rng = self.get_attribute("_rng")
                    current_execution_devices = self.get_attribute("_current_execution_devices")
                    for idx, _ in enumerate(current_execution_devices):
                        current_execution_device = current_execution_devices[idx][1]
                        battery_maximum_stored_energy_in_joules = current_execution_device["battery_maximum_stored_energy_in_joules"]
                        initial_battery_level_percentage_sampled = rng.choice(initial_remaining_battery_level_percentage_list, size=1, replace=False)[0]
                        initial_remaining_battery_energy_in_joules = battery_maximum_stored_energy_in_joules * initial_battery_level_percentage_sampled
                        current_execution_devices[idx][1]["battery_stored_energy_in_joules"] = initial_remaining_battery_energy_in_joules
                    # Update the current execution devices.
                    self._set_attribute("_current_execution_devices", current_execution_devices)
            # Print the start of the execution.
            print("\nStarting the execution '{0}'...".format(execution_name))
            # Get the process wait time (after launching).
            process_wait_time = execution_settings.get("process_wait_time", 5)
            # Start the execution timer.
            start = perf_counter()
            # Get the number of clients.
            num_clients = execution_settings["num_clients"]
            # Create a barrier to synchronize dataset loading.
            dataset_loaded_barrier = Barrier(num_clients + 1)
            # Start the Flower server in a separate process.
            server_id = execution_settings["server_id"]
            server_process = Process(target=lambda: self._launch_flower_server(server_id).launch_server())
            server_process.start()
            # Wait for the server to start.
            sleep(process_wait_time)
            print("Launched the Server '{0}'...".format(server_id))
            # Start multiple Flower clients in separate processes.
            client_processes = []
            for idx in range(num_clients):
                p = Process(target=self._launch_flower_client_safely, args=(idx, dataset_loaded_barrier))
                p.start()
                client_processes.append(p)
                sleep(process_wait_time)
                print("Launched the Client '{0}'...".format(idx))
            # Wait for all clients to safely load their datasets.
            print("Waiting for all clients to load and verify their datasets...")
            dataset_loaded_barrier.wait()
            print("All clients have completed dataset loading verification!")
            # Wait for all client processes to finish.
            for p in client_processes:
                p.join()
            # Wait for the server process to finish (finite rounds).
            server_process.join()
            # End the execution timer.
            end = perf_counter()
            # Print the elapsed time for the execution.
            elapsed_time_seconds = round((end - start), 2)
            print("\nElapsed time of '{0}': {1} seconds".format(execution_name, elapsed_time_seconds))
