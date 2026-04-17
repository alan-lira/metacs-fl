from logging import Logger
from pathlib import Path
from typing import Optional

from flwr.server import ClientManager, Server, ServerConfig, SimpleClientManager, start_server

from metacs_fl.server.flower_server import FlowerServer
from metacs_fl.flower_simulator.logger_actor import RemoteLoggerAdapter
from metacs_fl.utils.config_parser_util import parse_config_section
from metacs_fl.utils.logger_util import load_logger, log_message
from metacs_fl.utils.system_modeler_util import set_cpu_cores_affinity


class FlowerServerLauncher:

    def __init__(self,
                 id_: int,
                 config_file: Path,
                 personalized_settings: dict = None,
                 server_acquired_cpu_cores: list = None,
                 ray_node_shared_actors: dict = None,
                 instantiate_server: bool = True) -> None:
        # Initialize the attributes.
        self._server_id = id_
        self._config_file = config_file
        self._logging_settings = None
        self._fl_settings = None
        self._server_strategy_settings = None
        self._ssl_settings = None
        self._grpc_settings = None
        self._fit_config_settings = None
        self._evaluate_config_settings = None
        self._output_settings = None
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
        self._server_acquired_cpu_cores = server_acquired_cpu_cores
        if server_acquired_cpu_cores:
            set_cpu_cores_affinity(server_acquired_cpu_cores)
        # Set the Ray node shared actors.
        self._ray_node_shared_actors = ray_node_shared_actors or {}
        # Load the logger.
        self._logger = self._load_logger()
        # Load the initial fit config.
        self._fit_config = self._load_fit_config()
        # Load the initial evaluate config.
        self._evaluate_config = self._load_evaluate_config()
        # Load the initial parameters.
        self._initial_parameters = self._load_initial_parameters()
        # Load the server strategy.
        self._server_strategy = self._load_server_strategy()
        # Load the server config.
        self._server_config = self._load_server_config()
        # Instantiate the server, if requested.
        if instantiate_server:
            self._server = self._instantiate_server()

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
        # Parse and set the fl settings.
        fl_section = "FL Settings"
        fl_settings = parse_config_section(config_file, fl_section)
        stopping_criteria = fl_settings["stopping_criteria"]
        fl_stopping_criteria_section = "{0} Settings".format(stopping_criteria)
        fl_stopping_criteria_settings = parse_config_section(config_file, fl_stopping_criteria_section)
        fl_settings["stopping_crit"] = fl_stopping_criteria_settings
        self._set_attribute("_fl_settings", fl_settings)
        # Parse and set the server strategy settings.
        server_strategy_section = "Server Strategy Settings"
        server_strategy_settings = parse_config_section(config_file, server_strategy_section)
        server_strategy = server_strategy_settings["strategy"]
        match server_strategy:
            case "Random":
                server_strategy_implementation_section = "{0} Settings".format(server_strategy)
                server_strategy_implementation_settings = parse_config_section(config_file,
                                                                               server_strategy_implementation_section)
                model_aggregator = server_strategy_implementation_settings["model_aggregator"]
                model_aggregator_section = "{0} Settings".format(model_aggregator)
                model_aggregator_settings = parse_config_section(config_file, model_aggregator_section)
                model_aggregator_settings["name"] = model_aggregator
                server_strategy_implementation_settings["model_aggregator"] = model_aggregator_settings
                server_strategy_settings.update(server_strategy_implementation_settings)
                self._set_attribute("_server_strategy_settings", server_strategy_settings)
            case "SBAC-PAD_2024":
                server_strategy_implementation_section = "{0} Settings".format(server_strategy)
                server_strategy_implementation_settings = parse_config_section(config_file,
                                                                               server_strategy_implementation_section)
                client_selector_training = server_strategy_implementation_settings["client_selector_training"]
                client_selector_training_section = "{0} Settings".format(client_selector_training)
                client_selector_training_settings = parse_config_section(config_file, client_selector_training_section)
                client_selector_training_settings.update({"name": client_selector_training})
                server_strategy_implementation_settings["client_selector_training"] = client_selector_training_settings
                client_selector_testing = server_strategy_implementation_settings["client_selector_testing"]
                client_selector_testing_section = "{0} Settings".format(client_selector_testing)
                client_selector_testing_settings = parse_config_section(config_file, client_selector_testing_section)
                client_selector_testing_settings.update({"name": client_selector_testing})
                server_strategy_implementation_settings["client_selector_testing"] = client_selector_testing_settings
                model_aggregator = server_strategy_implementation_settings["model_aggregator"]
                model_aggregator_section = "{0} Settings".format(model_aggregator)
                model_aggregator_settings = parse_config_section(config_file, model_aggregator_section)
                model_aggregator_settings["name"] = model_aggregator
                server_strategy_implementation_settings["model_aggregator"] = model_aggregator_settings
                data_privacy_approach = server_strategy_implementation_settings["data_privacy_approach"]
                data_privacy_approach_settings = {}
                match data_privacy_approach:
                    case "Differentially_Private":
                        data_privacy_approach_section = "{0} Settings".format(data_privacy_approach)
                        data_privacy_approach_settings = parse_config_section(config_file, data_privacy_approach_section)
                data_privacy_approach_settings["name"] = data_privacy_approach
                server_strategy_implementation_settings["data_privacy_approach"] = data_privacy_approach_settings
                server_strategy_settings.update(server_strategy_implementation_settings)
                self._set_attribute("_server_strategy_settings", server_strategy_settings)
            case "Oort":
                server_strategy_implementation_section = "{0} Settings".format(server_strategy)
                server_strategy_implementation_settings = parse_config_section(config_file,
                                                                               server_strategy_implementation_section)
                client_selector_training = server_strategy_implementation_settings["client_selector_training"]
                client_selector_training_section = "{0} Settings".format(client_selector_training)
                client_selector_training_settings = parse_config_section(config_file, client_selector_training_section)
                client_selector_training_settings.update({"name": client_selector_training})
                server_strategy_implementation_settings["client_selector_training"] = client_selector_training_settings
                client_selector_testing = server_strategy_implementation_settings["client_selector_testing"]
                client_selector_testing_section = "{0} Settings".format(client_selector_testing)
                client_selector_testing_settings = parse_config_section(config_file, client_selector_testing_section)
                client_selector_testing_settings.update({"name": client_selector_testing})
                server_strategy_implementation_settings["client_selector_testing"] = client_selector_testing_settings
                model_aggregator = server_strategy_implementation_settings["model_aggregator"]
                model_aggregator_section = "{0} Settings".format(model_aggregator)
                model_aggregator_settings = parse_config_section(config_file, model_aggregator_section)
                model_aggregator_settings["name"] = model_aggregator
                server_strategy_implementation_settings["model_aggregator"] = model_aggregator_settings
                data_privacy_approach = server_strategy_implementation_settings["data_privacy_approach"]
                data_privacy_approach_settings = {}
                match data_privacy_approach:
                    case "Differentially_Private":
                        data_privacy_approach_section = "{0} Settings".format(data_privacy_approach)
                        data_privacy_approach_settings = parse_config_section(config_file, data_privacy_approach_section)
                data_privacy_approach_settings["name"] = data_privacy_approach
                server_strategy_implementation_settings["data_privacy_approach"] = data_privacy_approach_settings
                server_strategy_settings.update(server_strategy_implementation_settings)
                self._set_attribute("_server_strategy_settings", server_strategy_settings)
            case "MetaCS-FL":
                server_strategy_implementation_section = "{0} Settings".format(server_strategy)
                server_strategy_implementation_settings = parse_config_section(config_file,
                                                                               server_strategy_implementation_section)
                new_client_selection_criteria_training = server_strategy_implementation_settings["new_client_selection_criteria_training"]
                new_client_selection_criteria_training_section = "{0} Settings".format(new_client_selection_criteria_training)
                new_client_selection_criteria_training_settings = parse_config_section(config_file,
                                                                                       new_client_selection_criteria_training_section)
                server_strategy_implementation_settings["new_client_selection_criteria_training"] = new_client_selection_criteria_training_settings
                new_initial_solution_criteria_training = server_strategy_implementation_settings["new_initial_solution_criteria_training"]
                new_initial_solution_criteria_training_section = "{0} Settings".format(new_initial_solution_criteria_training)
                new_initial_solution_criteria_training_settings = parse_config_section(config_file,
                                                                                       new_initial_solution_criteria_training_section)
                server_strategy_implementation_settings["new_initial_solution_criteria_training"] = new_initial_solution_criteria_training_settings
                initial_solution_generator_training = server_strategy_implementation_settings["initial_solution_generator_training"]
                initial_solution_generator_training_section = "{0} Settings".format(initial_solution_generator_training)
                initial_solution_generator_training_settings = parse_config_section(config_file,
                                                                                    initial_solution_generator_training_section)
                initial_solution_generator_training_settings.update({"name": initial_solution_generator_training})
                server_strategy_implementation_settings["initial_solution_generator_training"] = initial_solution_generator_training_settings
                new_client_selection_criteria_testing = server_strategy_implementation_settings["new_client_selection_criteria_testing"]
                new_client_selection_criteria_testing_section = "{0} Settings".format(new_client_selection_criteria_testing)
                new_client_selection_criteria_testing_settings = parse_config_section(config_file,
                                                                                      new_client_selection_criteria_testing_section)
                server_strategy_implementation_settings["new_client_selection_criteria_testing"] = new_client_selection_criteria_testing_settings
                new_initial_solution_criteria_testing = server_strategy_implementation_settings["new_initial_solution_criteria_testing"]
                new_initial_solution_criteria_testing_section = "{0} Settings".format(new_initial_solution_criteria_testing)
                new_initial_solution_criteria_testing_settings = parse_config_section(config_file,
                                                                                      new_initial_solution_criteria_testing_section)
                server_strategy_implementation_settings["new_initial_solution_criteria_testing"] = new_initial_solution_criteria_testing_settings
                initial_solution_generator_testing = server_strategy_implementation_settings["initial_solution_generator_testing"]
                initial_solution_generator_testing_section = "{0} Settings".format(initial_solution_generator_testing)
                initial_solution_generator_testing_settings = parse_config_section(config_file,
                                                                                    initial_solution_generator_testing_section)
                initial_solution_generator_testing_settings.update({"name": initial_solution_generator_testing})
                server_strategy_implementation_settings["initial_solution_generator_testing"] = initial_solution_generator_testing_settings
                objective_function_section = "Objective Function Settings"
                objective_function_settings = parse_config_section(config_file, objective_function_section)
                server_strategy_implementation_settings["objective_function"] = objective_function_settings
                metaheuristic = server_strategy_implementation_settings["metaheuristic"]
                metaheuristic_section = "{0} Settings".format(metaheuristic)
                metaheuristic_settings = parse_config_section(config_file, metaheuristic_section)
                match metaheuristic:
                    case "LNS":
                        stopping_criteria = metaheuristic_settings["stopping_criteria"]
                        stopping_criteria_section = "{0} Settings".format(stopping_criteria)
                        stopping_criteria_settings = parse_config_section(config_file, stopping_criteria_section)
                        stopping_criteria_settings.update({"name": stopping_criteria})
                        metaheuristic_settings["stopping_criteria"] = stopping_criteria_settings
                        destroy_approach = metaheuristic_settings["destroy_approach"]
                        destroy_approach_section = "{0} Settings".format(destroy_approach)
                        destroy_approach_settings = parse_config_section(config_file, destroy_approach_section)
                        metaheuristic_settings["destroy_approach"] = destroy_approach_settings
                        repair_approach = metaheuristic_settings["repair_approach"]
                        repair_approach_section = "{0} Settings".format(repair_approach)
                        repair_approach_settings = parse_config_section(config_file, repair_approach_section)
                        metaheuristic_settings["repair_approach"] = repair_approach_settings
                        accept_criteria = metaheuristic_settings["accept_criteria"]
                        accept_criteria_section = "{0} Settings".format(accept_criteria)
                        accept_criteria_settings = parse_config_section(config_file, accept_criteria_section)
                        metaheuristic_settings["accept_criteria"] = accept_criteria_settings
                metaheuristic_settings["name"] = metaheuristic
                server_strategy_implementation_settings["metaheuristic"] = metaheuristic_settings
                model_aggregator = server_strategy_implementation_settings["model_aggregator"]
                model_aggregator_section = "{0} Settings".format(model_aggregator)
                model_aggregator_settings = parse_config_section(config_file, model_aggregator_section)
                model_aggregator_settings["name"] = model_aggregator
                server_strategy_implementation_settings["model_aggregator"] = model_aggregator_settings
                data_privacy_approach = server_strategy_implementation_settings["data_privacy_approach"]
                data_privacy_approach_settings = {}
                match data_privacy_approach:
                    case "Differentially_Private":
                        data_privacy_approach_section = "{0} Settings".format(data_privacy_approach)
                        data_privacy_approach_settings = parse_config_section(config_file, data_privacy_approach_section)
                data_privacy_approach_settings["name"] = data_privacy_approach
                server_strategy_implementation_settings["data_privacy_approach"] = data_privacy_approach_settings
                server_strategy_settings.update(server_strategy_implementation_settings)
                monitor_clients_reliability_score = server_strategy_settings.get("monitor_clients_reliability_score", False)
                if monitor_clients_reliability_score:
                    clients_reliability_score_section = "Clients Reliability Score Settings"
                    clients_reliability_score_settings = parse_config_section(config_file, clients_reliability_score_section)
                    server_strategy_implementation_settings["clients_reliability_score"] = clients_reliability_score_settings
                    server_strategy_settings.update(server_strategy_implementation_settings)
                self._set_attribute("_server_strategy_settings", server_strategy_settings)
        # Parse and set the ssl settings.
        ssl_section = "SSL Settings"
        ssl_settings = parse_config_section(config_file, ssl_section)
        self._set_attribute("_ssl_settings", ssl_settings)
        # Parse and set the grpc settings.
        grpc_section = "gRPC Settings"
        grpc_settings = parse_config_section(config_file, grpc_section)
        self._set_attribute("_grpc_settings", grpc_settings)
        # Parse and set the fit_config settings.
        fit_config_section = "Fit_Config Settings"
        fit_config_settings = parse_config_section(config_file, fit_config_section)
        self._set_attribute("_fit_config_settings", fit_config_settings)
        # Parse and set the evaluate_config settings.
        evaluate_config_section = "Evaluate_Config Settings"
        evaluate_config_settings = parse_config_section(config_file, evaluate_config_section)
        self._set_attribute("_evaluate_config_settings", evaluate_config_settings)
        # Parse and set the output settings.
        output_section = "Output Settings"
        output_settings = parse_config_section(config_file, output_section)
        self._set_attribute("_output_settings", output_settings)

    def _load_logger(self) -> Logger:
        # Get the necessary attributes.
        logging_settings = self.get_attribute("_logging_settings")
        server_id = self.get_attribute("_server_id")
        # Append the server's id to the output file name.
        file_name = Path(logging_settings["file_name"]).absolute()
        file_name = str(file_name.parent.joinpath(file_name.stem + "_{0}".format(server_id) + file_name.suffix))
        logging_settings["file_name"] = file_name
        # Set the logger name.
        logger_name = type(self).__name__ + "_Logger"
        # Initialize the logger.
        if "server_logger_actor" in self._ray_node_shared_actors:
            server_logger_actor = self._ray_node_shared_actors["server_logger_actor"]
            logger = RemoteLoggerAdapter(server_logger_actor)
        else:
            # Fallback: load the local logger.
            logger = load_logger(logging_settings, logger_name)
        # Return the logger.
        return logger

    def _load_fit_config(self) -> dict:
        # Get the necessary attributes.
        fit_config_settings = self.get_attribute("_fit_config_settings")
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Load the initial training configuration (fit_config).
        fit_config = {"comm_round": 0}
        fit_config.update(fit_config_settings)
        # Log the initial training configuration (fit_config).
        message = "[Server {0}] Base fit_config: {1}".format(server_id, fit_config)
        log_message(logger, message, "DEBUG")
        # Return the initial training configuration (fit_config).
        return fit_config

    def _load_evaluate_config(self) -> dict:
        # Get the necessary attributes.
        evaluate_config_settings = self.get_attribute("_evaluate_config_settings")
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Load the initial testing configuration (evaluate_config).
        evaluate_config = {"comm_round": 0}
        evaluate_config.update(evaluate_config_settings)
        # Log the initial testing configuration (evaluate_config).
        message = "[Server {0}] Base evaluate_config: {1}".format(server_id, evaluate_config)
        log_message(logger, message, "DEBUG")
        # Return the initial testing configuration (evaluate_config).
        return evaluate_config

    def _load_initial_parameters(self) -> any:
        """Server-side parameter initialization. A powerful mechanism which can be used, for instance:
        \n - To resume the training from a previously saved checkpoint;
        \n - To implement hybrid approaches, such as to fine-tune a pre-trained model using federated learning.
        \n If no parameters are set, the server will randomly select one client and ask its parameters."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Load the initial model parameters.
        initial_parameters = None
        # Log the initial model parameters.
        message = "[Server {0}] Initial model parameters: {1}".format(server_id, initial_parameters)
        log_message(logger, message, "DEBUG")
        # Return the initial model parameters.
        return initial_parameters

    def _load_server_strategy(self) -> any:
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        fl_settings = self.get_attribute("_fl_settings")
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        output_settings = self.get_attribute("_output_settings")
        root_output_folder = self.get_attribute("_root_output_folder")
        fit_config = self.get_attribute("_fit_config")
        evaluate_config = self.get_attribute("_evaluate_config")
        initial_parameters = self.get_attribute("_initial_parameters")
        strategy = server_strategy_settings["strategy"]
        # Initialize the server strategy.
        server_strategy = None
        match strategy:
            case x if x in ["Random", "SBAC-PAD_2024", "Oort", "MetaCS-FL"]:
                server_strategy = FlowerServer(id_=server_id,
                                               fl_settings=fl_settings,
                                               server_strategy_settings=server_strategy_settings,
                                               fit_config=fit_config,
                                               evaluate_config=evaluate_config,
                                               output_settings=output_settings,
                                               root_output_folder=root_output_folder,
                                               initial_parameters=initial_parameters,
                                               logger=logger)
        # Return the server strategy.
        return server_strategy

    def _load_server_config(self) -> ServerConfig:
        # Get the necessary attributes.
        fl_settings = self.get_attribute("_fl_settings")
        num_rounds = fl_settings["num_rounds"]
        round_timeout_in_seconds = fl_settings["round_timeout_in_seconds"]
        if round_timeout_in_seconds == "infinity":
            round_timeout_in_seconds = None
        # Instantiate the server config.
        server_config = ServerConfig(num_rounds=num_rounds,
                                     round_timeout=round_timeout_in_seconds)
        # Return the server config.
        return server_config

    def _load_ssl_certificates(self) -> Optional[tuple[bytes]]:
        # Get the necessary attributes.
        ssl_settings = self.get_attribute("_ssl_settings")
        enable_ssl = ssl_settings["enable_ssl"]
        ca_certificate_file = ssl_settings["ca_certificate_file"]
        server_certificate_file = ssl_settings["server_certificate_file"]
        server_rsa_private_key_file = ssl_settings["server_rsa_private_key_file"]
        # Initialize the SSL certificates tuple.
        ssl_certificates = None
        # If SSL secure connection is enabled...
        if enable_ssl:
            # Read the SSL certificates bytes.
            ca_certificate_bytes = ca_certificate_file.read_bytes()
            server_certificate_bytes = server_certificate_file.read_bytes()
            server_rsa_private_key_bytes = server_rsa_private_key_file.read_bytes()
            # Mount the SSL certificates tuple.
            ssl_certificates = (ca_certificate_bytes, server_certificate_bytes, server_rsa_private_key_bytes)
        # Return the SSL certificates tuple.
        return ssl_certificates

    def _get_server_address(self) -> str:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        listen_ip_address = grpc_settings["listen_ip_address"]
        listen_port = str(grpc_settings["listen_port"])
        # Return the server address.
        return listen_ip_address + ":" + listen_port

    def _get_max_message_length_in_bytes(self) -> int:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        max_message_length_in_bytes = grpc_settings["max_message_length_in_bytes"]
        # Return the maximum message length in bytes.
        return max_message_length_in_bytes

    @staticmethod
    def _instantiate_client_manager() -> ClientManager:
        # Instantiate a simple client manager.
        client_manager = SimpleClientManager()
        # Return the client manager.
        return client_manager

    def _instantiate_server(self) -> Server:
        # Instantiate the client manager.
        client_manager = self._instantiate_client_manager()
        # Get the server strategy.
        server_strategy = self.get_attribute("_server_strategy")
        # Instantiate the flower server.
        server = Server(client_manager=client_manager,
                        strategy=server_strategy)
        # Return the flower server.
        return server

    @staticmethod
    def _start_flower_server(server_address: str,
                             server: Optional[Server],
                             config: Optional[ServerConfig],
                             grpc_max_message_length: int,
                             certificates: Optional[tuple[bytes, bytes, bytes]]) -> None:
        # Start the flower server.
        start_server(server_address=server_address,
                     server=server,
                     config=config,
                     grpc_max_message_length=grpc_max_message_length,
                     certificates=certificates)

    def launch_server(self) -> None:
        # Get the server address (to-listen IP address and port).
        server_address = self._get_server_address()
        # Get the flower server.
        server = self.get_attribute("_server")
        # Get the server config.
        server_config = self.get_attribute("_server_config")
        # Get the maximum message length in bytes.
        max_message_length_in_bytes = self._get_max_message_length_in_bytes()
        # Load the secure socket Layer (SSL) certificates (SSL-enabled secure connection).
        ssl_certificates = self._load_ssl_certificates()
        # Start the flower server.
        self._start_flower_server(server_address,
                                  server,
                                  server_config,
                                  max_message_length_in_bytes,
                                  ssl_certificates)
        # End.
        exit(0)
