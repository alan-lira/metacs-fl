from collections import defaultdict
from configparser import ConfigParser
from copy import deepcopy
from multiprocessing import Barrier, Process
from shutil import copytree
from subprocess import CalledProcessError, run
from numpy import array, exp, median, percentile, zeros
from numpy.random import default_rng, Generator
from pathlib import Path
from socket import create_connection, getaddrinfo, gethostname
from sys import stdout as sys_stdout, stderr as sys_stderr
from time import perf_counter, sleep
from threading import BrokenBarrierError
from traceback import format_exc

from metacs_fl.client_launcher.flower_client_launcher import FlowerClientLauncher
from metacs_fl.devices.edge_devices import generate_edge_devices
from metacs_fl.networks.networks import generate_networks
from metacs_fl.server_launcher.flower_server_launcher import FlowerServerLauncher
from metacs_fl.utils.config_parser_util import parse_config_section
from metacs_fl.utils.host_profiler_util import HostProfiler
from metacs_fl.utils.system_modeler_util import get_cpu_cores_available


class FlowerExecutor:

    def __init__(self,
                 config_file: Path,
                 repetitions: int,
                 hostfile: Path | None = None,
                 output_gathering_settings: dict | None = None) -> None:
        # Initialize the attributes.
        self._config_file = config_file
        self._repetitions = repetitions
        self._hostfile = hostfile
        self._output_gathering_settings = output_gathering_settings or {}
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
    def _is_localhost_ip(ip_address: str) -> bool:
        return ip_address in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    @staticmethod
    def _normalize_hostname(value: str) -> str:
        return str(value).strip().lower().rstrip(".")

    @staticmethod
    def _short_hostname(value: str) -> str:
        value = FlowerExecutor._normalize_hostname(value)
        return value.split(".")[0]

    @staticmethod
    def _resolve_host_addresses(host: str) -> set[str]:
        addresses = set()
        if not host:
            return addresses
        try:
            for info in getaddrinfo(host, None):
                addresses.add(info[4][0])
        except Exception as _:
            pass
        return addresses

    @staticmethod
    def _get_local_ip_addresses() -> set[str]:
        local_values = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        try:
            hostname = gethostname()
            hostname_normalized = FlowerExecutor._normalize_hostname(hostname)
            hostname_short = FlowerExecutor._short_hostname(hostname)
            local_values.add(hostname_normalized)
            local_values.add(hostname_short)
            # Resolve both the hostname returned by the OS and its short form.
            for address in FlowerExecutor._resolve_host_addresses(hostname_normalized):
                local_values.add(address)
            for address in FlowerExecutor._resolve_host_addresses(hostname_short):
                local_values.add(address)
        except Exception as _:
            pass
        return local_values

    @classmethod
    def _is_this_node(cls, ip_address: str) -> bool:
        if cls._is_localhost_ip(ip_address):
            return True
        target = cls._normalize_hostname(ip_address)
        target_short = cls._short_hostname(ip_address)
        local_values = cls._get_local_ip_addresses()
        local_normalized = {cls._normalize_hostname(v) for v in local_values}
        local_short = {cls._short_hostname(v) for v in local_values}
        # Match exact hostname/IP, or short hostnames (such as paradoxe-1 == paradoxe-1.rennes.g5k).
        if target in local_normalized:
            return True
        if target_short in local_short:
            return True
        # Match by resolved IP addresses. This covers cases where the hostfile
        # uses FQDNs but the local node identifies itself by IP, or vice versa.
        target_addresses = cls._resolve_host_addresses(target)
        if target_addresses.intersection(local_values):
            return True
        return False

    @staticmethod
    def _load_nodes_file(nodes_file: Path) -> dict:
        """
        Load a role-based nodes file.

        Expected format:
            server 192.168.1.10
            client 192.168.1.11
            client 192.168.1.12

        Lines starting with '#' are ignored.
        """
        if not nodes_file.is_file():
            raise FileNotFoundError("Nodes file not found: {0}".format(nodes_file))
        server_ip = None
        client_ips = []
        with open(file=nodes_file, mode="r", encoding="utf-8") as n_f:
            for line in n_f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError("Invalid nodes file line: '{0}'. Expected: '<server|client> <ip>'."
                                     .format(line))
                role, ip_address = parts[0].lower(), parts[1]
                match role:
                    case "server":
                        if server_ip is not None:
                            raise ValueError("Only one server entry is allowed in the nodes file.")
                        server_ip = ip_address
                    case "client":
                        client_ips.append(ip_address)
                    case _:
                        raise ValueError("Invalid node role '{0}'. Use 'server' or 'client'.".format(role))
        if server_ip is None:
            raise ValueError("Nodes file must contain one 'server <ip>' entry.")
        if not client_ips:
            raise ValueError("Nodes file must contain at least one 'client <ip>' entry.")
        return {"server_ip": server_ip,
                "client_ips": client_ips}

    @staticmethod
    def _assign_clients_to_nodes(num_clients: int,
                                 client_ips: list[str]) -> dict:
        clients_by_ip = defaultdict(list)
        for client_id in range(num_clients):
            client_ip = client_ips[client_id % len(client_ips)]
            clients_by_ip[client_ip].append(client_id)
        return dict(clients_by_ip)

    @classmethod
    def _get_clients_for_this_node(cls,
                                   clients_by_ip: dict) -> list[int]:
        local_client_ids = []
        for ip_address, client_ids in clients_by_ip.items():
            if cls._is_this_node(ip_address):
                local_client_ids.extend(client_ids)
        return sorted(local_client_ids)

    @staticmethod
    def _is_distributed_nodes_plan(nodes_plan: dict) -> bool:
        all_ips = [nodes_plan["server_ip"]] + nodes_plan["client_ips"]
        return any(ip not in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} for ip in all_ips)

    @staticmethod
    def _as_bool(value: any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _sanitize_for_path(value: str) -> str:
        sanitized = []
        for char in str(value):
            if char.isalnum() or char in {"-", "_"}:
                sanitized.append(char)
            elif char in {".", ":", "/", "\\", " ", "@"}:
                sanitized.append("_")
            else:
                sanitized.append("_")
        return "".join(sanitized).strip("_") or "unknown"

    @classmethod
    def _get_this_node_identifier(cls) -> str:
        hostname = gethostname()
        local_ips = sorted(cls._get_local_ip_addresses())
        non_local_ips = [ip for ip in local_ips if not cls._is_localhost_ip(ip)]
        if non_local_ips:
            return cls._sanitize_for_path("{0}_{1}".format(hostname, non_local_ips[0]))
        return cls._sanitize_for_path(hostname)

    @staticmethod
    def _write_node_manifest(output_folder: Path,
                             execution_name: str,
                             is_server_node: bool,
                             local_client_ids: list[int],
                             is_distributed_execution: bool) -> None:
        output_folder.mkdir(exist_ok=True, parents=True)
        manifest_file = output_folder.joinpath("distributed_node_manifest.txt")
        try:
            hostname = gethostname()
            local_ips = sorted(FlowerExecutor._get_local_ip_addresses())
            with open(file=manifest_file, mode="w", encoding="utf-8") as m_f:
                m_f.write("execution_name={0}\n".format(execution_name))
                m_f.write("hostname={0}\n".format(hostname))
                m_f.write("local_ips={0}\n".format(",".join(local_ips)))
                m_f.write("is_distributed_execution={0}\n".format(is_distributed_execution))
                m_f.write("is_server_node={0}\n".format(is_server_node))
                m_f.write("local_client_ids={0}\n".format(",".join(map(str, local_client_ids))))
        except Exception as e:
            print("[Output Collection] Failed to write node manifest: {0}".format(e),
                  file=sys_stdout,
                  flush=True)
            print(format_exc(), file=sys_stdout, flush=True)

    @staticmethod
    def _run_command_with_trace(command: list[str],
                                description: str) -> None:
        print("[Output Collection] {0}: {1}".format(description, " ".join(command)),
              file=sys_stdout,
              flush=True)
        try:
            completed = run(command,
                            check=True,
                            capture_output=True,
                            text=True)
            if completed.stdout:
                print(completed.stdout, file=sys_stdout, flush=True)
            if completed.stderr:
                print(completed.stderr, file=sys_stdout, flush=True)
        except CalledProcessError as e:
            print("[Output Collection] Command failed while {0}.".format(description),
                  file=sys_stdout,
                  flush=True)
            print("[Output Collection] Return code: {0}".format(e.returncode),
                  file=sys_stdout,
                  flush=True)
            print("[Output Collection] stdout:\n{0}".format(e.stdout),
                  file=sys_stdout,
                  flush=True)
            print("[Output Collection] stderr:\n{0}".format(e.stderr),
                  file=sys_stdout,
                  flush=True)
            raise

    def _collect_outputs_to_collector(self,
                                      execution_name: str,
                                      execution_settings: dict,
                                      is_distributed_execution: bool,
                                      is_server_node: bool,
                                      local_client_ids: list[int]) -> None:
        gather_outputs = self._as_bool(execution_settings.get("gather_outputs", False))
        if not gather_outputs:
            return
        gather_single_node_execution = self._as_bool(execution_settings.get("gather_single_node_execution", False))
        if not is_distributed_execution and not gather_single_node_execution:
            print("[Output Collection] Skipping output collection because this is not a distributed execution.",
                  file=sys_stdout,
                  flush=True)
            return
        fail_on_collection_error = self._as_bool(execution_settings.get("fail_on_output_collection_error", True))
        try:
            local_output_folder = Path(execution_settings["execution_output_folder"])
            if not local_output_folder.is_dir():
                print("[Output Collection] Local output folder does not exist on this node: {0}"
                      .format(local_output_folder),
                      file=sys_stdout,
                      flush=True)
                return

            self._write_node_manifest(local_output_folder,
                                      execution_name,
                                      is_server_node,
                                      local_client_ids,
                                      is_distributed_execution)
            collector_folder = Path(execution_settings["output_collector_folder"])
            collector_ip = execution_settings.get("output_collector_ip", "127.0.0.1")
            collector_user = execution_settings.get("output_collector_user", None)
            collection_method = execution_settings.get("output_collection_method", "rsync")
            node_identifier = execution_settings.get("output_node_identifier", None)
            if node_identifier is None:
                node_identifier = self._get_this_node_identifier()
            else:
                node_identifier = self._sanitize_for_path(node_identifier)
            destination_subfolder = Path(self._sanitize_for_path(execution_name)).joinpath("node_{0}".format(node_identifier))
            if self._is_this_node(collector_ip) or self._is_localhost_ip(collector_ip):
                destination_folder = collector_folder.joinpath(destination_subfolder)
                print("[Output Collection] Copying local output folder to collector path: {0}"
                      .format(destination_folder),
                      file=sys_stdout,
                      flush=True)
                destination_folder.mkdir(exist_ok=True, parents=True)
                copytree(src=local_output_folder,
                         dst=destination_folder,
                         dirs_exist_ok=True)
                return
            if collector_user is None:
                raise ValueError("output_collector_user must be set when output_collector_ip is remote.")
            remote_destination_folder = collector_folder.joinpath(destination_subfolder)
            remote_login = "{0}@{1}".format(collector_user, collector_ip)
            if collection_method == "rsync":
                mkdir_command = ["ssh", remote_login, "mkdir -p {0}".format(str(remote_destination_folder))]
                self._run_command_with_trace(mkdir_command,
                                             "creating remote collector folder")
                rsync_command = ["rsync",
                                 "-az",
                                 str(local_output_folder) + "/",
                                 "{0}:{1}/".format(remote_login, remote_destination_folder)]
                self._run_command_with_trace(rsync_command,
                                             "sending output folder to collector")
            elif collection_method == "scp":
                mkdir_command = ["ssh", remote_login, "mkdir -p {0}".format(str(remote_destination_folder))]
                self._run_command_with_trace(mkdir_command,
                                             "creating remote collector folder")
                scp_command = ["scp",
                               "-r",
                               str(local_output_folder) + "/.",
                               "{0}:{1}/".format(remote_login, remote_destination_folder)]
                self._run_command_with_trace(scp_command,
                                             "copying output folder to collector")
            else:
                raise ValueError("Unsupported output_collection_method: {0}".format(collection_method))
        except Exception as e:
            print("[Output Collection] Failed to gather outputs: {0}".format(e),
                  file=sys_stdout,
                  flush=True)
            print(format_exc(), file=sys_stdout, flush=True)
            if fail_on_collection_error:
                raise

    def _apply_output_gathering_overrides(self,
                                          execution_settings: dict) -> None:
        output_gathering_settings = self.get_attribute("_output_gathering_settings")
        if not output_gathering_settings:
            return
        for key, value in output_gathering_settings.items():
            execution_settings[key] = value

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
            # Override server gRPC settings according to the executor/nodes plan.
            grpc_settings = parse_config_section(base_server_config_file, "gRPC Settings")
            grpc_settings["listen_ip_address"] = execution_settings.get("server_listen_ip_address", "0.0.0.0")
            grpc_settings["listen_port"] = execution_settings.get("server_port", grpc_settings["listen_port"])
            personalized_settings.update({"_grpc_settings": grpc_settings})
        # Return the personalized_settings dictionary.
        return personalized_settings

    @staticmethod
    def _get_personalized_settings_for_client(client_id: int,
                                              base_client_config_file: Path,
                                              late_join_clients: set,
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
            # Profile the client's host.
            num_host_profiles = execution_settings["num_host_profiles"]
            host_profiler = HostProfiler(Path(execution_output_folder))
            host_profile = host_profiler.profile_host_n_times(client_id, num_host_profiles)
            personalized_settings["_host_profile"] = host_profile
            # Set the late-join settings.
            is_late_join_client = client_id in late_join_clients
            late_join_settings = {"is_late_join_client": is_late_join_client}
            if is_late_join_client:
                late_join_first_appearance_round = execution_settings["round_of_first_appearance_of_late_join_clients"]
                late_join_settings.update({"late_join_first_appearance_round": late_join_first_appearance_round})
            personalized_settings["_late_join_settings"] = late_join_settings
            # Override client gRPC settings according to the executor/nodes plan.
            grpc_settings = parse_config_section(base_client_config_file, "gRPC Settings")
            grpc_settings["server_ip_address"] = execution_settings.get("server_ip", grpc_settings["server_ip_address"])
            grpc_settings["server_port"] = execution_settings.get("server_port", grpc_settings["server_port"])
            personalized_settings.update({"_grpc_settings": grpc_settings})
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

    @staticmethod
    def _generate_client_failure_probabilities(rng: Generator,
                                               num_clients: int,
                                               poisson_failure_lambda_range: list) -> dict:
        if not poisson_failure_lambda_range:
            lambdas = zeros(num_clients)
        else:
            lambdas = rng.uniform(poisson_failure_lambda_range[0], poisson_failure_lambda_range[1], num_clients)
        failure_probabilities = 1 - exp(-lambdas)  # Probability of ≥1 failure for Poisson(λ).
        client_failure_probabilities = {}
        for client_id, p in enumerate(failure_probabilities):
            client_failure_probabilities.update({client_id: p})
        return client_failure_probabilities

    @staticmethod
    def _compute_baseline_device_performances(current_execution_devices: dict) -> dict:
        devices = [d for _, d in current_execution_devices]
        # Compute metrics.
        fpocc_means = [(d["fpocc_training_min"] + d["fpocc_training_max"]) / 2 for d in devices]
        cpu_perfs = [d["cpu_num_cores"] * (d["cpu_frequency_in_hertz"] / 1e9) for d in devices]
        memories = [d["memory_size_in_gigabytes"] for d in devices]
        # Network metrics.
        uploads = [d["upload_bandwidth_mean"] for d in devices]
        downloads = [d["download_bandwidth_mean"] for d in devices]
        latencies = [d["base_latency"] for d in devices]
        jitters = [d["latency_jitter"] for d in devices]
        packet_losses = [d["packet_loss_rate"] for d in devices]
        # Energy metrics.
        reception_powers = [d["mean_power_consumption_data_reception_in_watts"] for d in devices]
        heavy_powers = [d["mean_power_consumption_heavy_computational_load_in_watts"] for d in devices]
        transmission_powers = [d["mean_power_consumption_data_transmission_in_watts"] for d in devices]
        idle_powers = [d["mean_power_consumption_idle_in_watts"] for d in devices]
        # Percentile-based baselines.
        baseline_device_performances = {"fpocc_training_mean": percentile(fpocc_means, 90),
                                        "cpu_performance": percentile(cpu_perfs, 90),
                                        "memory_size_in_gigabytes": percentile(memories, 90),
                                        "upload_mbps": percentile(uploads, 90),
                                        "download_mbps": percentile(downloads, 90),
                                        "latency_ms": percentile(latencies, 10),
                                        "jitter_ms": percentile(jitters, 10),
                                        "packet_loss": percentile(packet_losses, 10),
                                        "mean_power_reception_watts": percentile(reception_powers, 10),
                                        "mean_power_heavy_watts": percentile(heavy_powers, 10),
                                        "mean_power_transmission_watts": percentile(transmission_powers, 10),
                                        "mean_power_idle_watts": percentile(idle_powers, 10)}
        return baseline_device_performances

    @staticmethod
    def _get_device_compute_score(baselines: dict,
                                  device: dict) -> float:
        fpocc_mean = (device["fpocc_training_min"] + device["fpocc_training_max"]) / 2
        fpocc_norm = min(fpocc_mean / baselines["fpocc_training_mean"], 1.0)
        cpu_perf = device["cpu_num_cores"] * (device["cpu_frequency_in_hertz"] / 1e9)
        cpu_norm = min(cpu_perf / baselines["cpu_performance"], 1.0)
        mem_norm = min(device["memory_size_in_gigabytes"] / baselines["memory_size_in_gigabytes"], 1.0)
        device_compute_score = round((fpocc_norm * 0.45 + cpu_norm * 0.35 + mem_norm * 0.20) * 100.0, 2)
        return device_compute_score

    @staticmethod
    def _get_device_network_score(baselines: dict,
                                  device: dict) -> float:
        upload_norm = min(device["upload_bandwidth_mean"] / baselines["upload_mbps"], 1.0)
        download_norm = min(device["download_bandwidth_mean"] / baselines["download_mbps"], 1.0)
        latency_norm = min(baselines["latency_ms"] / device["base_latency"], 1.0)
        jitter_norm = min(baselines["jitter_ms"] / device["latency_jitter"], 1.0)
        packet_loss_norm = min(baselines["packet_loss"] / device["packet_loss_rate"], 1.0)
        device_network_score = round((upload_norm * 0.30 +
                                      download_norm * 0.25 +
                                      latency_norm * 0.25 +
                                      jitter_norm * 0.10 +
                                      packet_loss_norm * 0.10) * 100.0, 2)
        return device_network_score

    @staticmethod
    def _get_device_energy_score(baselines: dict,
                                 device: dict) -> float:
        reception_norm = min(baselines["mean_power_reception_watts"] / max(device["mean_power_consumption_data_reception_in_watts"], 0.1), 1.0)
        heavy_norm = min(baselines["mean_power_heavy_watts"] / max(device["mean_power_consumption_heavy_computational_load_in_watts"], 0.1), 1.0)
        transmission_norm = min(baselines["mean_power_transmission_watts"] / max(device["mean_power_consumption_data_transmission_in_watts"], 0.1), 1.0)
        idle_norm = min(baselines["mean_power_idle_watts"] / max(device["mean_power_consumption_idle_in_watts"], 0.1), 1.0)
        device_energy_score = round((reception_norm + heavy_norm + transmission_norm + idle_norm) / 4 * 100.0, 2)
        return device_energy_score

    def _get_device_overall_score(self, device: dict) -> float:
        baselines = self.get_attribute("_baseline_device_performances")
        compute_score = self._get_device_compute_score(baselines, device)
        network_score = self._get_device_network_score(baselines, device)
        energy_score = self._get_device_energy_score(baselines, device)
        # Weighted overall device score.
        device_overall_score = round(compute_score * 0.5 + network_score * 0.25 + energy_score * 0.25, 2)
        return device_overall_score

    def _load_devices_scores(self) -> None:
        current_execution_devices = self.get_attribute("_current_execution_devices")
        devices_scores = []
        for idx, device in enumerate(current_execution_devices):
            device_dict = device[1]
            device_overall_score = self._get_device_overall_score(device_dict)
            device_dict["device_overall_score"] = device_overall_score
            devices_scores.append((idx, device_overall_score))
        # Store the list of devices scores.
        self._set_attribute("_devices_scores", devices_scores)

    def _write_devices_scores_to_file(self,
                                      root_output_folder: Path,
                                      output_file: Path = Path("clients_scores.csv")) -> None:
        # Get the list of devices scores.
        devices_scores = self.get_attribute("_devices_scores")
        # Sort by ascending score.
        devices_scores.sort(key=lambda x: x[1])
        output_file = root_output_folder.joinpath(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(file=output_file, mode="w", encoding="utf-8") as o_f:
            header_line = "client_id,client_score\n"
            o_f.write(header_line)
            for client_id, client_score in devices_scores:
                data_line = "{0},{1}\n".format(client_id, client_score)
                o_f.write(data_line)

    def _determine_late_join_clients(self,
                                     num_clients: int,
                                     late_join_clients_percentage: float,
                                     late_join_clients_performance_profile: str) -> set:
        if late_join_clients_percentage <= 0:
            return set()
        num_late_join_clients = min(max(1, int(num_clients * late_join_clients_percentage)), num_clients)
        # Initialize the list of late-join clients.
        late_join_clients = []
        # Get the list of devices scores.
        devices_scores = self.get_attribute("_devices_scores")
        # Sort by ascending score.
        devices_scores.sort(key=lambda x: x[1])
        # Match the user-defined performance profile for late-join clients.
        match late_join_clients_performance_profile:
            case "random":
                rng = self.get_attribute("_rng")
                late_join_clients = rng.choice(range(num_clients), size=num_late_join_clients, replace=False)
            case "worst":
                late_join_clients = [i for i, _ in devices_scores[:num_late_join_clients]]
            case "best":
                late_join_clients = [i for i, _ in reversed(devices_scores[-num_late_join_clients:])]
            case "medium":
                scores = array([s for _, s in devices_scores])
                p33 = percentile(a=scores, q=33)
                p66 = percentile(a=scores, q=66)
                # Clients whose scores fall inside the middle percentile band.
                medium_band = [(i, s) for (i, s) in devices_scores if p33 <= s <= p66]
                # If we have enough, randomly choose from inside this band.
                if len(medium_band) >= num_late_join_clients:
                    rng = self.get_attribute("_rng")
                    chosen = rng.choice(list(medium_band), size=num_late_join_clients, replace=False)
                    late_join_clients = [int(i) for (i, _) in chosen]
                else:
                    # Otherwise: include entire medium band, then expand outward closest to median score.
                    late_join_clients = [i for (i, _) in medium_band]
                    remaining = num_late_join_clients - len(late_join_clients)
                    # Sort by closeness to the median.
                    median_score = median(scores)
                    remaining_pool = [(i, abs(s - median_score)) for (i, s) in devices_scores if i not in late_join_clients]
                    remaining_pool.sort(key=lambda x: x[1])
                    late_join_clients.extend([i for (i, _) in remaining_pool[:remaining]])
        return {int(i) for i in late_join_clients}

    def _write_late_join_clients_to_file(self,
                                         late_join_clients: set[int],
                                         late_join_clients_performance_profile: str,
                                         late_join_clients_round_of_first_appearance: int,
                                         root_output_folder: Path,
                                         output_file: Path = Path("late_join_clients_ids.csv")) -> None:
        output_file = root_output_folder.joinpath(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        # Get the list of devices scores.
        devices_scores = self.get_attribute("_devices_scores")
        # Sort devices scores by ascending ID.
        devices_scores.sort(key=lambda x: x[0])
        # Sort late-join clients by ascending score.
        late_join_clients_sorted = sorted(late_join_clients, key=lambda client_id: devices_scores[client_id][1])
        with open(file=output_file, mode="w", encoding="utf-8") as o_f:
            header_line = "client_id,client_score,performance_profile,round_of_first_appearance\n"
            o_f.write(header_line)
            for client_id in late_join_clients_sorted:
                client_score = devices_scores[client_id][1]
                data_line = "{0},{1},{2},{3}\n" \
                            .format(client_id,
                                    client_score,
                                    late_join_clients_performance_profile,
                                    late_join_clients_round_of_first_appearance)
                o_f.write(data_line)

    def _launch_flower_server(self,
                              server_id: int) -> FlowerServerLauncher:
        try:
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
        except Exception as _:
            print(format_exc(), file=sys_stdout, flush=True)
            sys_stdout.flush()
            sys_stderr.flush()
            raise

    def _launch_flower_server_safely(self,
                                     server_id: int) -> None:
        try:
            server_launcher = self._launch_flower_server(server_id)
            server_launcher.launch_server()
        except Exception as e:
            print("Error in server {0}: {1}".format(server_id, e), file=sys_stdout, flush=True)
            print(format_exc(), file=sys_stdout, flush=True)
            sys_stdout.flush()
            sys_stderr.flush()
            raise SystemExit(1)

    @staticmethod
    def _wait_for_server(server_id: int,
                         server_ip: str,
                         server_port: int,
                         server_timeout: int = 60,
                         server_process: Process | None = None) -> None:
        start = perf_counter()
        while perf_counter() - start < server_timeout:
            if server_process is not None and server_process.exitcode is not None:
                raise RuntimeError("Server '{0}' process exited before becoming reachable. Exit code: {1}."
                                   .format(server_id, server_process.exitcode))
            try:
                with create_connection((server_ip, server_port), timeout=1):
                    print("Server '{0}' is reachable on {1}:{2}!".format(server_id, server_ip, server_port),
                          flush=True)
                    return
            except OSError as _:
                sleep(1)
        raise RuntimeError("Server '{0}' did not become ready in time at {1}:{2}."
                           .format(server_id, server_ip, server_port))

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
                                                                                  self.get_attribute("_late_join_clients"),
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
                                     dataset_loaded_barrier: Barrier = None) -> None:
        try:
            client_launcher = self._launch_flower_client(client_id, dataset_loaded_barrier)
            # Only launch the client if dataset was verified (though barrier will be passed regardless).
            client_launcher.launch_client()
        except Exception as e:
            print("Error in client {0}: {1}".format(client_id, e))
            print(format_exc(), file=sys_stdout, flush=True)
            if dataset_loaded_barrier is not None:
                try:
                    dataset_loaded_barrier.wait()
                except BrokenBarrierError:
                    print("[Client {0}] Barrier broken.".format(client_id))
                except Exception as e2:
                    print("[Client {0}] Barrier wait failed: {1}".format(client_id, e2), file=sys_stdout, flush=True)
                    print(format_exc(), file=sys_stdout, flush=True)
            sys_stdout.flush()
            sys_stderr.flush()
            raise SystemExit(1)

    def execute_fl_with_flower(self) -> None:
        try:
            self._execute_fl_with_flower_impl()
        except Exception as e:
            print("Fatal error in execute_fl_with_flower: {0}".format(e), file=sys_stdout, flush=True)
            print(format_exc(), file=sys_stdout, flush=True)
            sys_stdout.flush()
            sys_stderr.flush()
            raise

    def _execute_fl_with_flower_impl(self) -> None:
        executions_to_execute = []
        # Print the number of the repetitions (independent executions).
        print("Number of repetitions (independent executions): {0}".format(self._repetitions))
        config_file = self.get_attribute("_config_file")
        parser = ConfigParser()
        parser.read(config_file)
        execution_sections = [s for s in parser.sections() if s.startswith("Execution_") and s.endswith("_N Settings")]
        for execution_section in execution_sections:
            execution_settings = parse_config_section(config_file, execution_section)
            for repetition_idx in range(1, self._repetitions + 1):
                execution_name = execution_section.replace("_N Settings", "_{0}".format(repetition_idx))
                execution_settings_copy = deepcopy(execution_settings)
                execution_settings_copy["execution_output_folder"] = \
                    execution_settings_copy["execution_output_folder"].replace("N", str(repetition_idx))
                execution_settings_copy["output_collector_folder"] = \
                    execution_settings_copy["output_collector_folder"].replace("N", str(repetition_idx))
                executions_to_execute.append({execution_name: execution_settings_copy})
        # Iterate through the list of executions to execute.
        for current_execution in executions_to_execute:
            # Update the current execution.
            self._set_attribute("_current_execution", current_execution)
            # Get the execution name and settings.
            execution_name = next(iter(current_execution))
            execution_settings = current_execution[execution_name]
            # Apply command-line overrides for output gathering, if provided.
            self._apply_output_gathering_overrides(execution_settings)
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
                    # Compute the baseline device performances.
                    baseline_device_performances = self._compute_baseline_device_performances(current_execution_devices)
                    # Update the baseline device performances.
                    self._set_attribute("_baseline_device_performances", baseline_device_performances)
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
                # Generate the clients' failure probabilities.
                rng = self.get_attribute("_rng")
                num_clients = execution_settings["num_clients"]
                poisson_failure_lambda_range = execution_settings.get("poisson_failure_lambda_range", [])
                client_failure_probabilities = self._generate_client_failure_probabilities(rng,
                                                                                           num_clients,
                                                                                           poisson_failure_lambda_range)
                current_execution_devices = self.get_attribute("_current_execution_devices")
                for idx, _ in enumerate(current_execution_devices):
                    current_execution_devices[idx][1]["client_failure_probability"] = client_failure_probabilities[idx]
                # Load the devices scores.
                self._load_devices_scores()
                # Write the devices scores to file.
                execution_output_folder = Path(execution_settings["execution_output_folder"])
                self._write_devices_scores_to_file(execution_output_folder)
            # Print the start of the execution.
            print("\nStarting the execution '{0}'...".format(execution_name))
            # Start the execution timer.
            start = perf_counter()
            # Get the number of clients.
            num_clients = execution_settings["num_clients"]
            # Determine the set of late-join clients.
            late_join_clients_percentage = execution_settings.get("percentage_late_join_clients", 0.0)
            late_join_clients_performance_profile = execution_settings.get("performance_profile_late_join_clients", "random")
            late_join_clients_round_of_first_appearance = execution_settings.get("round_of_first_appearance_of_late_join_clients", 1)
            late_join_clients = self._determine_late_join_clients(num_clients,
                                                                  late_join_clients_percentage,
                                                                  late_join_clients_performance_profile)
            self._set_attribute("_late_join_clients", late_join_clients)
            # Write the set of late-join clients, if any, to output file.
            if len(late_join_clients) > 0:
                execution_output_folder = Path(execution_settings["execution_output_folder"])
                self._write_late_join_clients_to_file(late_join_clients,
                                                      late_join_clients_performance_profile,
                                                      late_join_clients_round_of_first_appearance,
                                                      execution_output_folder)
            # Resolve the execution placement.
            hostfile = self.get_attribute("_hostfile")
            if hostfile is not None:
                nodes_file = hostfile
            else:
                nodes_file = execution_settings.get("nodes_file", None)
            if nodes_file:
                nodes_plan = self._load_nodes_file(Path(nodes_file))
            else:
                nodes_plan = {"server_ip": execution_settings.get("server_ip", "127.0.0.1"),
                              "client_ips": ["127.0.0.1"]}
            server_ip = nodes_plan["server_ip"]
            client_ips = nodes_plan["client_ips"]
            server_port = execution_settings.get("server_port", 8080)
            server_timeout = execution_settings.get("server_timeout", 60)
            # Store the resolved server endpoint in the current execution settings. These values are
            # later injected into the client/server personalized gRPC settings.
            execution_settings["server_ip"] = server_ip
            execution_settings["server_port"] = server_port
            clients_by_ip = self._assign_clients_to_nodes(num_clients, client_ips)
            local_client_ids = self._get_clients_for_this_node(clients_by_ip)
            is_server_node = self._is_this_node(server_ip)
            is_distributed_execution = self._is_distributed_nodes_plan(nodes_plan)
            print("Execution placement:")
            print("  Server IP: {0}".format(server_ip))
            print("  Server port: {0}".format(server_port))
            print("  Distributed execution: {0}".format(is_distributed_execution))
            print("  Clients assigned to this node: {0}".format(local_client_ids))
            # Start the Flower server only on the server node.
            server_process = None
            server_id = execution_settings["server_id"]
            if is_server_node:
                print("Launching Server '{0}' on this node...".format(server_id))
                server_process = Process(target=self._launch_flower_server_safely, args=(server_id,))
                server_process.start()
            else:
                print("This node is not the server node. Server will not be launched here.")
            # Wait until the server gRPC endpoint is reachable.
            try:
                self._wait_for_server(server_id, server_ip, server_port, server_timeout, server_process)
            except RuntimeError as e:
                print("Server '{0}' failed to become reachable at {1}:{2}: {3}"
                      .format(server_id, server_ip, server_port, e), file=sys_stdout, flush=True)
                if server_process is not None:
                    server_process.terminate()
                    server_process.join()
                raise
            print("Server '{0}' is reachable.".format(server_id))
            # Start only the Flower clients assigned to this node.
            client_processes = []
            # Get the process wait time (after launching).
            process_wait_time = execution_settings.get("process_wait_time", 5)
            if is_distributed_execution:
                # multiprocessing.Barrier only works for local processes.
                dataset_loaded_barrier = None
            else:
                dataset_loaded_barrier = Barrier(num_clients + 1)
            for client_id in local_client_ids:
                p = Process(target=self._launch_flower_client_safely, args=(client_id, dataset_loaded_barrier))
                p.start()
                client_processes.append(p)
                sleep(process_wait_time)
                print("Launched the Client '{0}' on this node...".format(client_id))
            # Wait for all local clients to safely load their datasets.
            if not is_distributed_execution and dataset_loaded_barrier is not None:
                print("Waiting for all local clients to load and verify their datasets...")
                dataset_loaded_barrier.wait()
                print("All local clients have completed dataset loading verification!")
            # Wait for all local client processes to finish and propagate child-process failures.
            for client_id, p in zip(local_client_ids, client_processes):
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError("Client process {0} exited with non-zero exit code {1}."
                                       .format(client_id, p.exitcode))
            # Wait for the server process to finish only on the server node and propagate failures.
            if server_process is not None:
                server_process.join()
                if server_process.exitcode != 0:
                    raise RuntimeError("Server process {0} exited with non-zero exit code {1}."
                                       .format(server_id, server_process.exitcode))
            # Gather this node's local output folder into the collector machine, if enabled.
            self._collect_outputs_to_collector(execution_name,
                                               execution_settings,
                                               is_distributed_execution,
                                               is_server_node,
                                               local_client_ids)
            # End the execution timer.
            end = perf_counter()
            # Print the elapsed time for the execution.
            elapsed_time_seconds = round((end - start), 2)
            print("\nElapsed time of '{0}': {1} seconds".format(execution_name, elapsed_time_seconds))
