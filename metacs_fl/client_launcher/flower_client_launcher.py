from grpc._channel import _MultiThreadedRendezvous
from logging import Logger
from pathlib import Path
from random import uniform
from ray import get
from time import perf_counter, sleep
from traceback import format_exc
from typing import Optional

from flwr.client import Client, start_client

from metacs_fl.client.flower_client import FlowerClient
from metacs_fl.energy_monitor.powerjoular_energy_monitor import PowerJoularEnergyMonitor
from metacs_fl.energy_monitor.pyjoules_energy_monitor import PyJoulesEnergyMonitor
from metacs_fl.flower_simulator.logger_actor import RemoteLoggerAdapter
from metacs_fl.utils.config_parser_util import parse_config_section
from metacs_fl.utils.dataset_loader_util import instantiate_fds, load_dataset
from metacs_fl.utils.logger_util import load_logger, log_message
from metacs_fl.utils.model_loader_util import load_model


class FlowerClientLauncher:

    def __init__(self,
                 id_: int,
                 config_file: Path,
                 personalized_settings: dict = None,
                 all_cpu_cores_available: list = None,
                 client_acquired_cpu_cores: list = None,
                 ray_node_shared_actors: dict = None,
                 instantiate_client: bool = True) -> None:
        # Start the client initialization duration timer.
        initialization_start = perf_counter()
        # Initialize the attributes.
        self._client_id = id_
        self._config_file = config_file
        self._logging_settings = None
        self._daemon_settings = None
        self._affinity_settings = None
        self._learning_rate_schedule_settings = None
        self._callbacks_settings = None
        self._ssl_settings = None
        self._grpc_settings = None
        self._dataset_settings = None
        self._local_dataset_settings = None
        self._federated_dataset_settings = None
        self._task_assignment_capacities_settings = None
        self._energy_monitoring_settings = None
        self._device_emulation_settings = None
        self._model_settings = None
        self._simulation_resources_settings = None
        self._root_output_folder = None
        # Parse the settings.
        self._parse_settings()
        # Update the settings if the personalized_settings dictionary was provided.
        if isinstance(personalized_settings, dict):
            for setting_key, config_pairs_dict in personalized_settings.items():
                if hasattr(self, setting_key):
                    setting = self.get_attribute(setting_key)
                    if setting:
                        for k, v in config_pairs_dict.items():
                            if k in setting:
                                setting[k] = v
                            else:
                                setting.update({k: v})
                        self._set_attribute(setting_key, setting)
                    else:
                        self._set_attribute(setting_key, config_pairs_dict)
                else:
                    self._set_attribute(setting_key, config_pairs_dict)
        # Set the CPU cores affinity.
        self._all_cpu_cores_available = all_cpu_cores_available or None
        self._client_acquired_cpu_cores = client_acquired_cpu_cores or None
        # Set the Ray node shared actors.
        self._ray_node_shared_actors = ray_node_shared_actors or {}
        # Load the logger.
        self._logger = self._load_logger()
        # Load the dataset.
        dataset_loading_dict = {}
        loading_approach = self.get_attribute("_dataset_settings")["loading_approach"]
        local_dataset_settings = self.get_attribute("_local_dataset_settings")
        federated_dataset_settings = self.get_attribute("_federated_dataset_settings")
        model_settings = self.get_attribute("_model_settings")
        # Log a 'loading the dataset' message.
        message = ("[Client {0}] Loading the dataset (loading approach: {1})..."
                   .format(self._client_id, loading_approach))
        log_message(self._logger, message, "DEBUG")
        # Get the Ray node's shared fds_actor, if exists.
        if "fds_actor" in self._ray_node_shared_actors:
            fds_actor = self._ray_node_shared_actors["fds_actor"]
            self._x_train, self._y_train, self._x_test, self._y_test, dataset_loading_duration \
                = get(fds_actor.load_dataset_for_client.remote(self._client_id,
                                                               loading_approach,
                                                               local_dataset_settings,
                                                               federated_dataset_settings,
                                                               model_settings,
                                                               self._root_output_folder))
        else:
            self._fds = instantiate_fds(federated_dataset_settings)
            dataset_loading_dict = load_dataset(self._client_id,
                                                loading_approach,
                                                local_dataset_settings,
                                                federated_dataset_settings,
                                                model_settings,
                                                self._fds,
                                                Path(self._root_output_folder))
            self._x_train = dataset_loading_dict["x_train"]
            self._y_train = dataset_loading_dict["y_train"]
            self._x_test = dataset_loading_dict["x_test"]
            self._y_test = dataset_loading_dict["y_test"]
            dataset_loading_duration = dataset_loading_dict["dataset_loading_duration"]
                # Log the dataset loading duration.
        message = ("[Client {0}] The dataset loading took {1} seconds."
                   .format(self._client_id, dataset_loading_duration))
        log_message(self._logger, message, "DEBUG")
        # Load the energy monitor.
        self._energy_monitor = None
        #self._energy_monitor = self._load_energy_monitor()
        # Instantiate and compile the model.
        model_settings = self.get_attribute("_model_settings")
        learning_rate_schedule_settings = self.get_attribute("_learning_rate_schedule_settings")
        # Get the Ray node's shared model_actor, if exists.
        if "model_actor" in self._ray_node_shared_actors:
            model_actor = self._ray_node_shared_actors["model_actor"]
            self._model, self._metrics_names \
                = get(model_actor.load_model_for_client.remote(model_settings,
                                                               learning_rate_schedule_settings,
                                                               dataset_loading_dict))
        else:
            self._model, self._metrics_names = load_model(model_settings,
                                                          learning_rate_schedule_settings,
                                                          dataset_loading_dict)
        # Get the client initialization duration.
        initialization_duration_in_seconds = perf_counter() - initialization_start
        # Log a 'client initialization duration' message.
        message = "[Client {0}] The initialization took {1} seconds!" \
                  .format(self._client_id, round(initialization_duration_in_seconds, 2))
        log_message(self._logger, message, "DEBUG")
        # Instantiate the client, if requested.
        if instantiate_client:
            self._client = self._instantiate_client(initialization_duration_in_seconds)

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def _parse_settings(self) -> None:
        # Get the necessary attributes.
        config_file = self.get_attribute("_config_file")
        # Parse and set the logging settings.
        logging_section = "Logging Settings"
        logging_settings = parse_config_section(config_file, logging_section)
        self._set_attribute("_logging_settings", logging_settings)
        # Parse and set the daemon settings.
        daemon_section = "Daemon Settings"
        daemon_settings = parse_config_section(config_file, daemon_section)
        self._set_attribute("_daemon_settings", daemon_settings)
        # Parse and set the affinity settings.
        affinity_section = "Affinity Settings"
        affinity_settings = parse_config_section(config_file, affinity_section)
        self._set_attribute("_affinity_settings", affinity_settings)
        # Parse and set the learning rate schedule settings.
        learning_rate_schedule_section = "Learning Rate Schedule Settings"
        learning_rate_schedule_settings = parse_config_section(config_file, learning_rate_schedule_section)
        learning_rate_schedule_name = learning_rate_schedule_settings["learning_rate_schedule_name"]
        learning_rate_schedule_particular_section = "{0} Learning Rate Schedule Settings".format(learning_rate_schedule_name)
        learning_rate_schedule_particular_settings = parse_config_section(config_file, learning_rate_schedule_particular_section)
        learning_rate_schedule_settings.update(learning_rate_schedule_particular_settings)
        self._set_attribute("_learning_rate_schedule_settings", learning_rate_schedule_settings)
        # Parse and set the callbacks settings.
        callbacks_section = "Callbacks Settings"
        callbacks_settings = parse_config_section(config_file, callbacks_section)
        if callbacks_settings["enable_reduce_lr_on_plateau_callback"]:
            callback_section = "ReduceLROnPlateau Callback Settings"
            callback_settings = parse_config_section(config_file, callback_section)
            callbacks_settings.update({"ReduceLROnPlateau": callback_settings})
        if callbacks_settings["enable_early_stopping_callback"]:
            callback_section = "EarlyStopping Callback Settings"
            callback_settings = parse_config_section(config_file, callback_section)
            callbacks_settings.update({"EarlyStopping": callback_settings})
        self._set_attribute("_callbacks_settings", callbacks_settings)
        # Parse and set the ssl settings.
        ssl_section = "SSL Settings"
        ssl_settings = parse_config_section(config_file, ssl_section)
        self._set_attribute("_ssl_settings", ssl_settings)
        # Parse and set the grpc settings.
        grpc_section = "gRPC Settings"
        grpc_settings = parse_config_section(config_file, grpc_section)
        self._set_attribute("_grpc_settings", grpc_settings)
        # Parse and set the dataset settings.
        dataset_section = "Dataset Settings"
        dataset_settings = parse_config_section(config_file, dataset_section)
        self._set_attribute("_dataset_settings", dataset_settings)
        # Parse and set the local dataset settings.
        local_dataset_section = "Local Dataset Settings"
        local_dataset_settings = parse_config_section(config_file, local_dataset_section)
        self._set_attribute("_local_dataset_settings", local_dataset_settings)
        # Parse and set the federated dataset settings.
        federated_dataset_section = "FederatedDataset Settings"
        federated_dataset_settings = parse_config_section(config_file, federated_dataset_section)
        dataset_partitioner = federated_dataset_settings["dataset_partitioner"]
        match dataset_partitioner:
            case "DirichletPartitioner":
                dirichlet_partitioner_section = "DirichletPartitioner Settings"
                dirichlet_partitioner_settings = parse_config_section(config_file, dirichlet_partitioner_section)
                federated_dataset_settings.update(dirichlet_partitioner_settings)
            case "PathologicalPartitioner":
                pathological_partitioner_section = "PathologicalPartitioner Settings"
                pathological_partitioner_settings = parse_config_section(config_file, pathological_partitioner_section)
                federated_dataset_settings.update(pathological_partitioner_settings)
        self._set_attribute("_federated_dataset_settings", federated_dataset_settings)
        # Parse and set the task assignment capacities settings.
        task_assignment_capacities_section = "Task Assignment Capacities Settings"
        task_assignment_capacities_settings = parse_config_section(config_file, task_assignment_capacities_section)
        self._set_attribute("_task_assignment_capacities_settings", task_assignment_capacities_settings)
        # Parse and set the energy monitoring settings.
        energy_monitoring_section = "Energy Monitoring Settings"
        energy_monitoring_settings = parse_config_section(config_file,
                                                          energy_monitoring_section)
        energy_monitor_name = energy_monitoring_settings["energy_monitor"]
        energy_monitor_section = "{0} Monitor Settings".format(energy_monitor_name)
        energy_monitor_settings = parse_config_section(config_file, energy_monitor_section)
        energy_monitoring_settings.update({energy_monitor_name: energy_monitor_settings})
        self._set_attribute("_energy_monitoring_settings", energy_monitoring_settings)
        # Parse and set the device emulation settings.
        device_emulation_section = "Device Emulation Settings"
        device_emulation_settings = parse_config_section(config_file, device_emulation_section)
        self._set_attribute("_device_emulation_settings", device_emulation_settings)
        # Parse and set the model settings.
        model_section = "Model Settings"
        model_settings = parse_config_section(config_file, model_section)
        model_provider = model_settings["provider"]
        model_provider_section = "{0} Model Settings".format(model_provider)
        model_provider_settings = parse_config_section(config_file, model_provider_section)
        model_name = model_provider_settings["model_name"]
        model_provider_specific_section = "{0} {1} Settings".format(model_provider, model_name)
        model_provider_specific_settings = parse_config_section(config_file, model_provider_specific_section)
        optimizer = model_provider_settings["optimizer_name"]
        optimizer_section = "{0} {1} Settings".format(model_provider, optimizer)
        optimizer_settings = parse_config_section(config_file, optimizer_section)
        loss = model_provider_settings["loss_name"]
        loss_section = "{0} {1} Settings".format(model_provider, loss)
        loss_settings = parse_config_section(config_file, loss_section)
        model_settings.update({model_provider: model_provider_settings,
                               model_name: model_provider_specific_settings,
                               optimizer: optimizer_settings,
                               loss: loss_settings})
        self._set_attribute("_model_settings", model_settings)

    def _load_logger(self) -> Logger | RemoteLoggerAdapter:
        # Get the necessary attributes.
        logging_settings = self.get_attribute("_logging_settings")
        client_id = self.get_attribute("_client_id")
        # Append the client's id to the output file name.
        file_name = Path(logging_settings["file_name"]).absolute()
        file_name = str(file_name.parent.joinpath(file_name.stem + "_{0}".format(client_id) + file_name.suffix))
        logging_settings["file_name"] = file_name
        # Set the logger name.
        logger_name = type(self).__name__ + "_Logger"
        # Initialize the logger.
        if "client_logger_actor" in self._ray_node_shared_actors:
            client_logger_actor = self._ray_node_shared_actors["client_logger_actor"]
            logger = RemoteLoggerAdapter(client_logger_actor)
        else:
            # Fallback: load the local logger.
            logger = load_logger(logging_settings, logger_name)
        # Return the logger.
        return logger

    def _load_energy_monitor(self) -> any:
        # Get the necessary attributes.
        energy_monitoring_settings = self.get_attribute("_energy_monitoring_settings")
        enable_energy_monitoring = energy_monitoring_settings["enable_energy_monitoring"]
        energy_monitor_name = energy_monitoring_settings["energy_monitor"]
        energy_monitor_settings = energy_monitoring_settings[energy_monitor_name]
        # Initialize the energy monitor.
        energy_monitor = None
        # If energy monitoring is enabled...
        if enable_energy_monitoring:
            match energy_monitor_name:
                case "pyJoules":
                    monitoring_domains = energy_monitor_settings["monitoring_domains"]
                    unit = energy_monitor_settings["unit"]
                    energy_monitor = PyJoulesEnergyMonitor(monitoring_domains, unit)
                case "PowerJoular":
                    monitoring_domains = energy_monitor_settings["monitoring_domains"]
                    unit = energy_monitor_settings["unit"]
                    process_monitoring = energy_monitor_settings["process_monitoring"]
                    unique_monitor = energy_monitor_settings["unique_monitor"]
                    report_consumptions_per_timestamp = energy_monitor_settings["report_consumptions_per_timestamp"]
                    remove_energy_consumptions_files = energy_monitor_settings["remove_energy_consumptions_files"]
                    energy_consumptions_file = energy_monitor_settings["energy_consumptions_file"]
                    energy_monitor = PowerJoularEnergyMonitor(monitoring_domains,
                                                              unit,
                                                              process_monitoring,
                                                              unique_monitor,
                                                              report_consumptions_per_timestamp,
                                                              remove_energy_consumptions_files,
                                                              energy_consumptions_file)
        self._set_attribute("_energy_monitor", energy_monitor)
        # Return the energy monitor.
        return energy_monitor

    def _instantiate_client(self,
                            initialization_duration_in_seconds: float) -> Client:
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        logger = self.get_attribute("_logger")
        daemon_settings = self.get_attribute("_daemon_settings")
        affinity_settings = self.get_attribute("_affinity_settings")
        task_assignment_capacities_settings = self.get_attribute("_task_assignment_capacities_settings")
        model_settings = self.get_attribute("_model_settings")
        callbacks_settings = self.get_attribute("_callbacks_settings")
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        host_profile = self.get_attribute("_host_profile")
        late_joining_settings = self.get_attribute("_late_joining_settings")
        simulation_resources_settings = self.get_attribute("_simulation_resources_settings")
        root_output_folder = self.get_attribute("_root_output_folder")
        all_cpu_cores_available = self.get_attribute("_all_cpu_cores_available")
        client_acquired_cpu_cores = self.get_attribute("_client_acquired_cpu_cores")
        model = self.get_attribute("_model")
        metrics_names = self.get_attribute("_metrics_names")
        x_train = self.get_attribute("_x_train")
        y_train = self.get_attribute("_y_train")
        x_test = self.get_attribute("_x_test")
        y_test = self.get_attribute("_y_test")
        energy_monitor = self.get_attribute("_energy_monitor")
        # Verify if the energy consumptions monitor to be used is PowerJoular
        # and if only one monitoring process is allowed to run in the system.
        if isinstance(energy_monitor, PowerJoularEnergyMonitor) and energy_monitor.get_attribute("_unique_monitor"):
            # Get the unique PowerJoular attributes.
            powerjoular_unique_attributes = vars(energy_monitor)
            powerjoular_unique_attributes.update({"_energy_monitor": "PowerJoular_Unique"})
            powerjoular_unique_attributes = list(powerjoular_unique_attributes.items())
            energy_monitor = powerjoular_unique_attributes
        # Instantiate the flower client.
        client = FlowerClient(id_=client_id,
                              model=model,
                              metrics_names=metrics_names,
                              x_train=x_train,
                              y_train=y_train,
                              x_test=x_test,
                              y_test=y_test,
                              energy_monitor=energy_monitor,
                              daemon_settings=daemon_settings,
                              affinity_settings=affinity_settings,
                              task_assignment_capacities_settings=task_assignment_capacities_settings,
                              model_settings=model_settings,
                              callbacks_settings=callbacks_settings,
                              device_emulation_settings=device_emulation_settings,
                              host_profile=host_profile,
                              late_joining_settings=late_joining_settings,
                              logger=logger,
                              initialization_duration_in_seconds=initialization_duration_in_seconds,
                              simulation_resources_settings = simulation_resources_settings,
                              root_output_folder=root_output_folder,
                              all_cpu_cores_available=all_cpu_cores_available,
                              client_acquired_cpu_cores=client_acquired_cpu_cores)
        client = client.to_client()
        # Return the flower client.
        return client

    def _load_ssl_certificates(self) -> Optional[tuple[bytes]]:
        # Get the necessary attributes.
        ssl_settings = self.get_attribute("_ssl_settings")
        enable_ssl = ssl_settings["enable_ssl"]
        ca_certificate_file = ssl_settings["ca_certificate_file"]
        # Initialize the SSL certificates tuple.
        ssl_certificates = None
        # If SSL secure connection is enabled...
        if enable_ssl:
            # Read the SSL certificates bytes.
            ca_certificate_bytes = ca_certificate_file.read_bytes()
            # Mount the SSL certificates tuple.
            ssl_certificates = ca_certificate_bytes
        # Return the SSL certificates tuple.
        return ssl_certificates

    def _get_server_address(self) -> str:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        server_ip_address = grpc_settings["server_ip_address"]
        server_port = str(grpc_settings["server_port"])
        # Return the server address.
        return server_ip_address + ":" + server_port

    def _get_max_message_length_in_bytes(self) -> int:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        max_message_length_in_bytes = grpc_settings["max_message_length_in_bytes"]
        # Return the maximum message length in bytes.
        return max_message_length_in_bytes

    def _get_connection_retries_settings(self) -> tuple:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        max_connection_retries = grpc_settings["max_connection_retries"]
        max_backoff_in_seconds = grpc_settings["max_backoff_in_seconds"]
        # Return the maximum connection retries and maximum backoff in seconds.
        return max_connection_retries, max_backoff_in_seconds

    @staticmethod
    def _start_flower_client(client_id: int,
                             server_address: str,
                             client: Client,
                             grpc_max_message_length: int,
                             root_certificates: Optional[tuple[bytes, bytes, bytes]],
                             max_connection_retries: int,
                             max_backoff_in_seconds: float,
                             logger: Logger) -> None:
        # Start the flower client.
        current_try = 1
        while True:
            try:
                start_client(server_address=server_address,
                             client=client,
                             grpc_max_message_length=grpc_max_message_length,
                             root_certificates=root_certificates)
                break
            except _MultiThreadedRendezvous:
                traceback_exception_str = format_exc()
                if current_try == max_connection_retries:
                    raise traceback_exception_str
                random_second_fraction = round(uniform(0, 1), 2)
                wait_time = min(((2 ** current_try) + random_second_fraction), max_backoff_in_seconds)
                if "grpc_status:14" in traceback_exception_str:
                    message = ("[Client {0}] Could not connect to the Server ({1})! "
                               "Retrying in {2} seconds (retries left: {3})...") \
                              .format(client_id, server_address, wait_time, max_connection_retries - current_try)
                    log_message(logger, message, "INFO")
                sleep(wait_time)
                current_try += 1

    def launch_client(self) -> None:
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        logger = self.get_attribute("_logger")
        client = self.get_attribute("_client")
        energy_monitor = self.get_attribute("_energy_monitor")
        # Get the flower server address (IP address and port).
        server_address = self._get_server_address()
        # Get the maximum message length in bytes.
        max_message_length_in_bytes = self._get_max_message_length_in_bytes()
        # Get the Secure Socket Layer (SSL) certificates (SSL-enabled secure connection).
        ssl_certificates = self._load_ssl_certificates()
        # Get the settings for connection retries to the server.
        max_connection_retries, max_backoff_in_seconds = self._get_connection_retries_settings()
        # Start the flower client.
        if isinstance(energy_monitor, PowerJoularEnergyMonitor) and energy_monitor.get_attribute("_unique_monitor"):
            # Start the unique PowerJoular monitoring process.
            energy_monitor.start()
            self._start_flower_client(client_id,
                                      server_address,
                                      client,
                                      max_message_length_in_bytes,
                                      ssl_certificates,
                                      max_connection_retries,
                                      max_backoff_in_seconds,
                                      logger)
            # Stop the unique PowerJoular monitoring process.
            energy_monitor.stop()
        else:
            # Start the flower client.
            self._start_flower_client(client_id,
                                      server_address,
                                      client,
                                      max_message_length_in_bytes,
                                      ssl_certificates,
                                      max_connection_retries,
                                      max_backoff_in_seconds,
                                      logger)
        # End.
        exit(0)
