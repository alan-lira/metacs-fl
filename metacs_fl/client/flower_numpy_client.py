import sys
from os import devnull, environ

# Suppress TensorFlow C++ log messages (redirecting stderr to null).
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.stderr = open(devnull, "w")

# Ignore GPU.
environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from keras import losses
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from logging import Logger
from multiprocessing import Process, Queue, set_start_method
from numpy import argmax, array, asarray, clip, float32, int8, linspace, mean, minimum, ndarray, ones, sum, unique, \
    where, zeros
from numpy.random import default_rng, laplace, rand
from os import getpid
from pandas import read_csv
from pathlib import Path
from re import compile
from socket import gethostname
from tensorflow import function, GradientTape, Module, reduce_mean, TensorSpec
from tensorflow.python.framework.ops import EagerTensor, SymbolicTensor
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.types.experimental import PolymorphicFunction
from time import perf_counter, process_time

from flwr.client import NumPyClient
from flwr.common import NDArray, NDArrays

from metacs_fl.dataset_slicer.round_robin_dataset_slicer import RoundRobinDatasetSlicer
from metacs_fl.energy_monitor.powerjoular_energy_monitor import PowerJoularEnergyMonitor
from metacs_fl.energy_monitor.pyjoules_energy_monitor import PyJoulesEnergyMonitor
from metacs_fl.network_simulator.network_simulator import NetworkSimulator
from metacs_fl.utils.battery_util import get_remaining_battery_energy_from_file, \
    initialize_remaining_battery_energy_file, update_remaining_battery_energy_file
from metacs_fl.utils.dataset_loader_util import get_task_assignment_capacities, get_classes_distribution
from metacs_fl.utils.logger_util import log_message
from metacs_fl.utils.model_loader_util import load_model_from_file, save_model_to_file
from metacs_fl.utils.system_modeler_util import calculate_computation_time, \
    check_if_perf_is_available, generate_dummy_sample_batch, get_cpu_cores_available, get_cpu_and_memory_info, \
    get_flop_performance_statistics, get_memory_efficiency_rate, get_memory_traffic_per_flop_in_bytes, \
    get_model_input_shape, get_model_output_classes, launch_perf_process, parse_perf_log_file, set_cpu_cores_affinity, \
    stop_process, summarize_perf_metrics, calculate_initial_parameters_upload_time, calculate_download_time, \
    calculate_upload_time, calculate_download_energy, calculate_upload_energy, calculate_computation_energy, \
    calculate_idle_energy, calculate_initialization_energy, estimate_mt_m_device, convert_network_bandwidth, \
    convert_duration


class ProfilingModule(Module):

    def __init__(self,
                 model: Model) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def create_silent_profiler_options() -> dict:
        # Create a silent profiler options object (to disable the profiler report printing).
        opts = ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        return opts

    @staticmethod
    def _load_input_signature_train(x: EagerTensor,
                                    y: EagerTensor) -> list:
        input_signature_train = [TensorSpec(shape=x.shape, dtype=x.dtype), TensorSpec(shape=y.shape, dtype=y.dtype)]
        return input_signature_train

    @staticmethod
    def _load_input_signature_test(x: EagerTensor) -> list:
        input_signature_test = [TensorSpec(shape=x.shape, dtype=x.dtype)]
        return input_signature_test

    def build_train_fn(self,
                       x: EagerTensor,
                       y: EagerTensor) -> PolymorphicFunction:
        input_signature_train = self._load_input_signature_train(x, y)
        @function(input_signature=input_signature_train)
        def train_step(x: EagerTensor,
                       y: EagerTensor) -> SymbolicTensor:
            with GradientTape() as gradient_tape:
                pred = self.model(x, training=True)
                loss = losses.categorical_crossentropy(y, pred)
            grads = gradient_tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return reduce_mean(loss)
        return train_step

    def build_test_fn(self,
                      x: EagerTensor) -> PolymorphicFunction:
        input_signature_test = self._load_input_signature_test(x)
        @function(input_signature=input_signature_test)
        def test_step(x: EagerTensor) -> SymbolicTensor:
            return self.model(x, training=False)
        return test_step


class TrainingMeasurementsCallback(Callback):

    def __init__(self,
                 energy_monitor: any) -> None:
        super().__init__()
        # Initialize the attributes.
        self._energy_monitor = energy_monitor
        self._energy_monitor_tag = "training_energy"
        self._training_start_timestamp = None
        self._training_end_timestamp = None
        self._training_elapsed_time_start = None
        self._training_elapsed_time = None
        self._training_cpu_time_start = None
        self._training_cpu_time = None
        self._training_energy_consumptions = {}

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def on_train_begin(self,
                       logs=None) -> None:
        # Get the energy consumption monitor.
        energy_monitor = self.get_attribute("_energy_monitor")
        energy_monitor_tag = self.get_attribute("_energy_monitor_tag")
        # If there is an energy consumption monitor...
        if energy_monitor:
            # Start the model training energy consumption monitoring.
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.start(energy_monitor_tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                model_training_pid = getpid()
                energy_monitor.start(model_training_pid)
        # Set the model training start.
        training_elapsed_time_start = perf_counter()
        training_cpu_time_start = process_time()
        training_start_timestamp = datetime.now()
        self._set_attribute("_training_elapsed_time_start", training_elapsed_time_start)
        self._set_attribute("_training_cpu_time_start", training_cpu_time_start)
        self._set_attribute("_training_start_timestamp", training_start_timestamp)

    def on_train_end(self,
                     logs=None) -> None:
        # Get the model training start.
        training_elapsed_time_start = self.get_attribute("_training_elapsed_time_start")
        training_cpu_time_start = self.get_attribute("_training_cpu_time_start")
        training_start_timestamp = self.get_attribute("_training_start_timestamp")
        # Set the model training duration (elapsed and CPU times).
        training_elapsed_time = perf_counter() - training_elapsed_time_start
        training_cpu_time = process_time() - training_cpu_time_start
        self._set_attribute("_training_elapsed_time", training_elapsed_time)
        self._set_attribute("_training_cpu_time", training_cpu_time)
        # Set the model training end timestamp.
        training_end_timestamp = datetime.now()
        self._set_attribute("_training_end_timestamp", training_end_timestamp)
        # Get the energy consumption monitor.
        energy_monitor = self.get_attribute("_energy_monitor")
        energy_monitor_tag = self.get_attribute("_energy_monitor_tag")
        # If there is an energy consumption monitor...
        if energy_monitor:
            # Initialize the model training energy consumptions.
            training_energy_consumptions = {}
            # Stop the model training energy consumption monitoring and get the energy consumptions measurements.
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.stop()
                training_energy_consumptions = energy_monitor.get_energy_consumptions(energy_monitor_tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                energy_monitor.stop()
                training_energy_consumptions = energy_monitor.get_energy_consumptions(energy_monitor_tag,
                                                                                      training_start_timestamp,
                                                                                      training_end_timestamp)
            elif isinstance(energy_monitor, list):
                energy_monitor = dict(energy_monitor)
                if energy_monitor["_energy_monitor"] == "PowerJoular_Unique":
                    monitoring_domains = energy_monitor["_monitoring_domains"]
                    unit = energy_monitor["_unit"]
                    process_monitoring = energy_monitor["_process_monitoring"]
                    unique_monitor = energy_monitor["_unique_monitor"]
                    report_consumptions_per_timestamp = energy_monitor["_report_consumptions_per_timestamp"]
                    remove_energy_consumptions_files = energy_monitor["_remove_energy_consumptions_files"]
                    energy_consumptions_file = energy_monitor["_energy_consumptions_file"]
                    pj_unique = PowerJoularEnergyMonitor(monitoring_domains,
                                                         unit,
                                                         process_monitoring,
                                                         unique_monitor,
                                                         report_consumptions_per_timestamp,
                                                         remove_energy_consumptions_files,
                                                         energy_consumptions_file)
                    training_energy_consumptions = pj_unique.get_energy_consumptions(energy_monitor_tag,
                                                                                     training_start_timestamp,
                                                                                     training_end_timestamp)
            # Set the model training energy consumptions.
            self._set_attribute("_training_energy_consumptions", training_energy_consumptions)


class TestingMeasurementsCallback(Callback):

    def __init__(self,
                 energy_monitor: any) -> None:
        super().__init__()
        # Initialize the attributes.
        self._energy_monitor = energy_monitor
        self._energy_monitor_tag = "testing_energy"
        self._testing_start_timestamp = None
        self._testing_end_timestamp = None
        self._testing_elapsed_time_start = None
        self._testing_elapsed_time = None
        self._testing_cpu_time_start = None
        self._testing_cpu_time = None
        self._testing_energy_consumptions = {}

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def on_test_begin(self,
                      logs=None) -> None:
        # Get the energy consumption monitor.
        energy_monitor = self.get_attribute("_energy_monitor")
        energy_monitor_tag = self.get_attribute("_energy_monitor_tag")
        # If there is an energy consumption monitor...
        if energy_monitor:
            # Start the model testing energy consumption monitoring.
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.start(energy_monitor_tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                model_testing_pid = getpid()
                energy_monitor.start(model_testing_pid)
        # Set the model testing start.
        testing_elapsed_time_start = perf_counter()
        testing_cpu_time_start = process_time()
        testing_start_timestamp = datetime.now()
        self._set_attribute("_testing_elapsed_time_start", testing_elapsed_time_start)
        self._set_attribute("_testing_cpu_time_start", testing_cpu_time_start)
        self._set_attribute("_testing_start_timestamp", testing_start_timestamp)

    def on_test_end(self,
                    logs=None) -> None:
        # Get the model testing start.
        testing_elapsed_time_start = self.get_attribute("_testing_elapsed_time_start")
        testing_cpu_time_start = self.get_attribute("_testing_cpu_time_start")
        testing_start_timestamp = self.get_attribute("_testing_start_timestamp")
        # Set the model testing duration (elapsed and CPU times).
        testing_elapsed_time = perf_counter() - testing_elapsed_time_start
        testing_cpu_time = process_time() - testing_cpu_time_start
        self._set_attribute("_testing_elapsed_time", testing_elapsed_time)
        self._set_attribute("_testing_cpu_time", testing_cpu_time)
        # Set the model testing end timestamp.
        testing_end_timestamp = datetime.now()
        self._set_attribute("_testing_end_timestamp", testing_end_timestamp)
        # Get the energy consumption monitor.
        energy_monitor = self.get_attribute("_energy_monitor")
        energy_monitor_tag = self.get_attribute("_energy_monitor_tag")
        # If there is an energy consumption monitor...
        if energy_monitor:
            # Initialize the model testing energy consumptions.
            testing_energy_consumptions = {}
            # Stop the model testing energy consumption monitoring and get the energy consumptions measurements.
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.stop()
                testing_energy_consumptions = energy_monitor.get_energy_consumptions(energy_monitor_tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                energy_monitor.stop()
                testing_energy_consumptions = energy_monitor.get_energy_consumptions(energy_monitor_tag,
                                                                                     testing_start_timestamp,
                                                                                     testing_end_timestamp)
            elif isinstance(energy_monitor, list):
                energy_monitor = dict(energy_monitor)
                if energy_monitor["_energy_monitor"] == "PowerJoular_Unique":
                    monitoring_domains = energy_monitor["_monitoring_domains"]
                    unit = energy_monitor["_unit"]
                    process_monitoring = energy_monitor["_process_monitoring"]
                    unique_monitor = energy_monitor["_unique_monitor"]
                    report_consumptions_per_timestamp = energy_monitor["_report_consumptions_per_timestamp"]
                    remove_energy_consumptions_files = energy_monitor["_remove_energy_consumptions_files"]
                    energy_consumptions_file = energy_monitor["_energy_consumptions_file"]
                    pj_unique = PowerJoularEnergyMonitor(monitoring_domains,
                                                         unit,
                                                         process_monitoring,
                                                         unique_monitor,
                                                         report_consumptions_per_timestamp,
                                                         remove_energy_consumptions_files,
                                                         energy_consumptions_file)
                    testing_energy_consumptions = pj_unique.get_energy_consumptions(energy_monitor_tag,
                                                                                    testing_start_timestamp,
                                                                                    testing_end_timestamp)
            # Set the model testing energy consumptions.
            self._set_attribute("_testing_energy_consumptions", testing_energy_consumptions)


class FlowerNumpyClient(NumPyClient):

    def __init__(self,
                 id_: int,
                 model: Model,
                 metrics_names: list,
                 x_train: NDArray,
                 y_train: NDArray,
                 x_test: NDArray,
                 y_test: NDArray,
                 energy_monitor: any,
                 daemon_settings: dict,
                 affinity_settings: dict,
                 task_assignment_capacities_settings: dict,
                 model_settings: dict,
                 callbacks_settings: dict,
                 device_emulation_settings: dict,
                 host_profile: dict,
                 logger: Logger,
                 initialization_duration_in_seconds: float,
                 simulation_resources_settings: dict = None,
                 root_output_folder: Path = None,
                 all_cpu_cores_available: list = None,
                 client_acquired_cpu_cores: list = None) -> None:
        # Initialize the attributes.
        self._client_id = id_
        self._model = model
        self._metrics_names = metrics_names
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._energy_monitor = energy_monitor
        self._daemon_settings = daemon_settings
        self._affinity_settings = affinity_settings
        self._task_assignment_capacities_settings = task_assignment_capacities_settings
        self._model_settings = model_settings
        self._device_emulation_settings = device_emulation_settings
        self._host_profile = host_profile
        self._simulation_resources_settings = simulation_resources_settings
        self._root_output_folder = root_output_folder
        self._all_cpu_cores_available = all_cpu_cores_available
        self._client_acquired_cpu_cores = client_acquired_cpu_cores
        self._logger = logger
        self._work_dir_training_phase = (Path(root_output_folder)
                                         .joinpath("training_phase")
                                         .joinpath("client_{0}".format(self._client_id)))
        self._work_dir_testing_phase = (Path(root_output_folder)
                                        .joinpath("testing_phase")
                                        .joinpath("client_{0}".format(self._client_id)))
        self._rrds = RoundRobinDatasetSlicer(x_train, y_train, x_test, y_test)
        # Set the local model file path.
        model_file = Path(root_output_folder).joinpath("models/client_{0}.keras".format(self._client_id)).absolute()
        self._model_file = model_file
        # Dump the local model to file.
        save_model_to_file(model, model_file)
        self._training_callbacks = self._set_training_callbacks(callbacks_settings)
        self._training_measurements_callback = TrainingMeasurementsCallback(energy_monitor)
        self._testing_measurements_callback = TestingMeasurementsCallback(energy_monitor)
        self._hostname = gethostname()
        # Set the client process affinity (the list of CPU cores to be used), if needed.
        if self._all_cpu_cores_available is None:
            self._all_cpu_cores_available = get_cpu_cores_available()
        if self._client_acquired_cpu_cores is None:
            if self._device_emulation_settings:
                client_num_cpu_cores_to_acquire = self._device_emulation_settings["cpu_num_cores"]
            else:
                client_num_cpu_cores_to_acquire = len(self._all_cpu_cores_available)
            self._client_acquired_cpu_cores = self._acquire_cpu_cores(client_num_cpu_cores_to_acquire)
            # Log a 'eligible CPU cores' message.
            message = "[Client {0}] {1} CPU cores will be used (list of IDs): {2}" \
                      .format(self._client_id,
                              len(self._client_acquired_cpu_cores),
                              ",".join([str(cpu_core_id) for cpu_core_id in self._client_acquired_cpu_cores]))
            log_message(logger, message, "DEBUG")
        # Set the CPU cores affinity.
        set_cpu_cores_affinity(self._client_acquired_cpu_cores)
        # Set the starting method of daemon processes.
        self._set_starting_method_of_daemon_processes()
        # Initialize the remaining battery energy in joules.
        remaining_battery_energy_file = (Path(root_output_folder)
                                         .joinpath("remaining_battery_energy/client_{0}.csv"
                                                   .format(self._client_id))
                                         .absolute())
        self._remaining_battery_energy_file = remaining_battery_energy_file
        remaining_battery_energy_in_joules = initialize_remaining_battery_energy_file(remaining_battery_energy_file,
                                                                                      device_emulation_settings)
        self._remaining_battery_energy_in_joules = remaining_battery_energy_in_joules
        # Estimate the energy consumed by this client during the initialization event.
        initialization_energy_in_joules = self._estimate_initialization_energy_in_joules(initialization_duration_in_seconds)
        # Record the energy consumed by this client after the connection establishment event.
        comm_round = 0
        phase = "initialization"
        connection_establishment_event = "after_establishing_connection_to_the_server"
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  connection_establishment_event,
                                                  initialization_energy_in_joules)
        # Initialize the network simulator.
        phi_ar2 = [0.5, 0.3]  # AR(2): two coefficients.
        self._network_simulator = NetworkSimulator(self._device_emulation_settings["upload_bandwidth_mean"],
                                                   self._device_emulation_settings["upload_bandwidth_std"],
                                                   self._device_emulation_settings["upload_bandwidth_min"],
                                                   self._device_emulation_settings["download_bandwidth_mean"],
                                                   self._device_emulation_settings["download_bandwidth_std"],
                                                   self._device_emulation_settings["download_bandwidth_min"],
                                                   self._device_emulation_settings["base_latency"],
                                                   self._device_emulation_settings["latency_jitter"],
                                                   self._device_emulation_settings["packet_loss_rate"],
                                                   phi_ar2)

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    def _deterministic_seed(client_id: int,
                            comm_round: int) -> int:
        seed_str = "{0}_{1}".format(client_id, comm_round)
        hash_bytes = sha256(seed_str.encode()).digest()
        deterministic_seed = int.from_bytes(hash_bytes[:8], "big")
        return deterministic_seed

    @staticmethod
    def _set_training_callbacks(callbacks_settings: dict) -> list:
        training_callbacks = []
        if "ReduceLROnPlateau" in callbacks_settings:
            reduce_lr_on_plateau_settings = callbacks_settings["ReduceLROnPlateau"]
            monitor = reduce_lr_on_plateau_settings["monitor"]
            factor = reduce_lr_on_plateau_settings["factor"]
            patience = reduce_lr_on_plateau_settings["patience"]
            verbose = reduce_lr_on_plateau_settings["verbose"]
            mode = reduce_lr_on_plateau_settings["mode"]
            min_delta = reduce_lr_on_plateau_settings["min_delta"]
            cooldown = reduce_lr_on_plateau_settings["cooldown"]
            min_lr = reduce_lr_on_plateau_settings["min_lr"]
            reduce_lr_on_plateau_callback = ReduceLROnPlateau(monitor=monitor,
                                                              factor=factor,
                                                              patience=patience,
                                                              verbose=verbose,
                                                              mode=mode,
                                                              min_delta=min_delta,
                                                              cooldown=cooldown,
                                                              min_lr=min_lr)
            training_callbacks.append(reduce_lr_on_plateau_callback)
        if "EarlyStopping" in callbacks_settings:
            early_stopping_settings = callbacks_settings["EarlyStopping"]
            monitor = early_stopping_settings["monitor"]
            min_delta = early_stopping_settings["min_delta"]
            patience = early_stopping_settings["patience"]
            verbose = early_stopping_settings["verbose"]
            mode = early_stopping_settings["mode"]
            baseline = early_stopping_settings["baseline"]
            restore_best_weights = early_stopping_settings["restore_best_weights"]
            start_from_epoch = early_stopping_settings["start_from_epoch"]
            early_stopping_callback = EarlyStopping(monitor=monitor,
                                                    min_delta=min_delta,
                                                    patience=patience,
                                                    verbose=verbose,
                                                    mode=mode,
                                                    baseline=baseline,
                                                    restore_best_weights=restore_best_weights,
                                                    start_from_epoch=start_from_epoch)
            training_callbacks.append(early_stopping_callback)
        return training_callbacks

    def _set_starting_method_of_daemon_processes(self) -> None:
        # Get the necessary attributes.
        daemon_settings = self.get_attribute("_daemon_settings")
        enable_daemon_mode = daemon_settings["enable_daemon_mode"]
        start_method = daemon_settings["start_method"]
        # If the daemon mode is enabled...
        if enable_daemon_mode:
            # Set the starting method of daemon processes.
            set_start_method(start_method)
            # Clear the model attribute.
            self._set_attribute("_model", None)

    def _acquire_cpu_cores(self,
                           client_num_cpu_cores_to_acquire: int) -> list:
        client_acquired_cpu_cores = []
        count = 0
        i = 0
        while count < client_num_cpu_cores_to_acquire:
            core = self._all_cpu_cores_available[i % len(self._all_cpu_cores_available)]
            client_acquired_cpu_cores.append(core)
            count += 1
            i += 1
        return client_acquired_cpu_cores

    def _estimate_initialization_energy_in_joules(self,
                                                  initialization_time_in_seconds: float) -> float:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        # Set the dictionary of client attributes to be used for the estimation.
        client_attributes = {"base_latency": device_emulation_settings["base_latency"],
                             "base_latency_unit": device_emulation_settings["base_latency_unit"],
                             "server_location": device_emulation_settings["server_location"],
                             "client_location": device_emulation_settings["client_location"],
                             "mpc_comp_i": device_emulation_settings["mean_power_consumption_heavy_computational_load_in_watts"],
                             "mpc_send_i": device_emulation_settings["mean_power_consumption_data_transmission_in_watts"]}
        # Estimate the energy consumed by this client during the initialization event.
        initialization_energy_in_joules = calculate_initialization_energy(client_attributes,
                                                                          initialization_time_in_seconds)
        # Return the estimated initialization energy (in joules).
        return initialization_energy_in_joules

    def _estimate_idle_energy_in_joules(self,
                                        idle_time_in_seconds: float) -> float:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        # Set the dictionary of client attributes to be used for the estimation.
        client_attributes = {"mpc_idle_i": device_emulation_settings["mean_power_consumption_idle_in_watts"]}
        # Estimate the energy consumed by this client while idle during an event of round r.
        idle_energy_in_joules = calculate_idle_energy(client_attributes, idle_time_in_seconds)
        # Return the estimated idle energy (in joules).
        return idle_energy_in_joules

    def _record_energy_consumed_during_event(self,
                                             comm_round: int,
                                             phase: str,
                                             event: str,
                                             consumed_energy_in_joules: float) -> None:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        remaining_battery_energy_file = self.get_attribute("_remaining_battery_energy_file")
        # Update the remaining battery energy file.
        battery_maximum_stored_energy_in_joules = device_emulation_settings["battery_maximum_stored_energy_in_joules"]
        device_connected_to_a_power_source = device_emulation_settings["device_connected_to_a_power_source"]
        update_remaining_battery_energy_file(remaining_battery_energy_file,
                                             comm_round,
                                             phase,
                                             event,
                                             consumed_energy_in_joules,
                                             battery_maximum_stored_energy_in_joules,
                                             device_connected_to_a_power_source)

    def _record_past_idle_events(self,
                                 config: dict) -> None:
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        client_id_str = "client_{0}".format(client_id)
        # Initialize the structured dict for the idle events' data.
        idle_events_data = defaultdict(lambda: {"train": {"idle_clients": [],
                                                          "makespan": 0,
                                                          "client_selection_time": 0},
                                                "test": {"idle_clients": [],
                                                         "makespan": 0,
                                                         "client_selection_time": 0}})
        # Regular expression to filter the target idle events keys.
        pattern = compile(r"^(idle_clients|makespan|client_selection_time)_(train|test)_comm_round_(\d+)$")
        # Parse the idle events' data.
        for key, value in config.items():
            match = pattern.match(key)
            if match:
                metric, phase, comm_round = match.groups()
                comm_round = int(comm_round)
                if metric == "idle_clients":
                    if isinstance(value, str):
                        if value:
                            value = value.split("|")
                        else:
                            value = []
                idle_events_data[comm_round][phase][metric] = value
        # Remove empty lists and zero values from each key pair.
        for comm_round, phases in idle_events_data.items():
            for phase, metrics in phases.items():
                filtered_metrics = {k: v for k, v in metrics.items() if v not in ([], 0)}
                idle_events_data[comm_round][phase] = filtered_metrics
        # Sort by comm_round.
        idle_events_data = dict(sorted(idle_events_data.items()))
        for past_round, past_round_phases in idle_events_data.items():
            # Record idle events for the training phase.
            phase = "train"
            train_metrics = past_round_phases[phase]
            if "client_selection_time" in train_metrics:
                # Get the time spent by the client selection event on training phase of round r.
                client_selection_time_in_seconds = train_metrics["client_selection_time"]
                # Estimate the energy consumed by this client while idle during the client selection event of round r.
                idle_client_selection_energy_in_joules = self._estimate_idle_energy_in_joules(client_selection_time_in_seconds)
                # Record the energy consumed by this client while idle during the client selection event of round r.
                idle_client_selection_event = "after_idle_during_the_client_selection"
                self._record_energy_consumed_during_event(past_round,
                                                          phase,
                                                          idle_client_selection_event,
                                                          idle_client_selection_energy_in_joules)
            if all(key in train_metrics for key in ("idle_clients", "makespan")):
                # Get the list of idle clients on training phase of round r.
                idle_clients_train = train_metrics["idle_clients"]
                # Get the makespan on training phase of round r.
                makespan_train = train_metrics["makespan"]
                if client_id_str in idle_clients_train:
                    # Estimate the energy consumed by this client while idle during the training phase event of round r.
                    idle_training_phase_energy_in_joules = self._estimate_idle_energy_in_joules(makespan_train)
                    # Record the energy consumed by this client while idle during the training phase event of round r.
                    idle_training_phase_event = "after_idle_during_the_{0}ing_phase".format(phase)
                    self._record_energy_consumed_during_event(past_round,
                                                              phase,
                                                              idle_training_phase_event,
                                                              idle_training_phase_energy_in_joules)
            # Record idle events for the testing phase.
            phase = "test"
            test_metrics = past_round_phases[phase]
            if "client_selection_time" in test_metrics:
                # Get the time spent by the client selection event on testing phase of round r.
                client_selection_time_in_seconds = test_metrics["client_selection_time"]
                # Estimate the energy consumed by this client while idle during the client selection event of round r.
                idle_client_selection_energy_in_joules = self._estimate_idle_energy_in_joules(client_selection_time_in_seconds)
                # Record the energy consumed by this client while idle during the client selection event of round r.
                idle_client_selection_event = "after_idle_during_the_client_selection"
                self._record_energy_consumed_during_event(past_round,
                                                          phase,
                                                          idle_client_selection_event,
                                                          idle_client_selection_energy_in_joules)
            if all(key in test_metrics for key in ("idle_clients", "makespan")):
                # Get the list of idle clients on testing phase of round r.
                idle_clients_test = test_metrics["idle_clients"]
                # Get the makespan on testing phase of round r.
                makespan_test = test_metrics["makespan"]
                if client_id_str in idle_clients_test:
                    # Estimate the energy consumed by this client while idle during the testing phase event of round r.
                    idle_testing_phase_energy_in_joules = self._estimate_idle_energy_in_joules(makespan_test)
                    # Record the energy consumed by this client while idle during the testing phase event of round r.
                    idle_testing_phase_event = "after_idle_during_the_{0}ing_phase".format(phase)
                    self._record_energy_consumed_during_event(past_round,
                                                              phase,
                                                              idle_testing_phase_event,
                                                              idle_testing_phase_energy_in_joules)

    @staticmethod
    def _dp_presence_randomized_response(local_classes: ndarray,
                                         true_presence_probability: float) -> ndarray:
        n = len(local_classes)
        if n == 0:
            return zeros(0, dtype=int8)
        # True presence: all classes in 'local_classes' are present -> ones.
        true_presence = ones(n, dtype=int8)
        # Randomized response:
        # - With probability p, report true bit (1);
        # - With probability (1-p), report random bit (0/1 with prob 0.5).
        rnd = rand(n)
        random_bits = (rand(n) < 0.5).astype(int8)
        reported = where(rnd < true_presence_probability, true_presence, random_bits)
        return reported.astype(int8)

    @staticmethod
    def _dp_noisy_histogram_from_counts(y_local_mapped: ndarray,
                                        num_global_classes: int,
                                        epsilon: float,
                                        clip_max: float = None) -> ndarray:
        hist = zeros(num_global_classes, dtype=float32)
        if len(y_local_mapped) > 0:
            uniq, counts = unique(y_local_mapped, return_counts=True)
            hist[uniq] = counts.astype(float32)
        if clip_max is not None:
            hist = minimum(hist, float(clip_max))
        # Each record contributes to exactly one bin of the histogram.
        # Clipping ensures no single client can push a bin above clip_max (if defined).
        sensitivity = 1.0
        scale = sensitivity / float(epsilon)
        noise = laplace(loc=0.0, scale=scale, size=hist.shape)
        noisy = hist + noise
        noisy_clipped = clip(noisy, 0.0, None).astype(float32)
        return noisy_clipped

    def get_properties(self,
                       config: dict) -> dict:
        """ Implementation of the abstract method from the NumPyClient class."""
        # Record the energy consumed by this client during the past idle events, if needed.
        self._record_past_idle_events(config)
        if "client_dp_presence" in config:
            y_train = self.get_attribute("_y_train")
            local_classes = asarray(unique(y_train))
            true_presence_probability = config["true_presence_probability"]
            # Get the privatized presence vector aligned with local_classes.
            dp_presence = self._dp_presence_randomized_response(local_classes, true_presence_probability)
            config.update({"client_dp_presence": "|".join(map(str, dp_presence.tolist())),
                           "client_local_classes_claimed": "|".join(map(str, local_classes.tolist()))})
        if "client_dp_histogram" in config:
            epsilon = config["epsilon"]
            y_train = self.get_attribute("_y_train")
            num_global_classes = int(config["num_global_classes"])
            class_index_map_str = config["class_index_map"]
            class_index_map = {int(k): int(v)
                               for k, v in (pair.split("=") for pair in class_index_map_str.split("|") if pair)}
            y_local_mapped = array([class_index_map[y] for y in y_train if y in class_index_map])
            dp_hist = self._dp_noisy_histogram_from_counts(y_local_mapped, num_global_classes, epsilon)
            config.update({"client_dp_histogram": "|".join(["{0}".format(float(x)) for x in dp_hist])})
        if "client_id" in config:
            client_id = self.get_attribute("_client_id")
            config.update({"client_id": client_id})
        if "client_hostname" in config:
            client_hostname = self.get_attribute("_hostname")
            config.update({"client_hostname": client_hostname})
        if "client_num_cpus" in config:
            client_num_cpus = len(self.get_attribute("_client_acquired_cpu_cores"))
            config.update({"client_num_cpus": client_num_cpus})
        if "client_cpu_cores_list" in config:
            client_cpu_cores_list = self.get_attribute("_client_acquired_cpu_cores")
            client_cpu_cores_list_str = "|".join([str(cpu_core_id) for cpu_core_id in client_cpu_cores_list])
            config.update({"client_cpu_cores_list": client_cpu_cores_list_str})
        if "client_num_training_examples_available" in config:
            client_num_training_examples_available = len(self.get_attribute("_x_train"))
            config.update({"client_num_training_examples_available": client_num_training_examples_available})
        if "client_num_testing_examples_available" in config:
            client_num_testing_examples_available = len(self.get_attribute("_x_test"))
            config.update({"client_num_testing_examples_available": client_num_testing_examples_available})
        task_properties = ["client_task_assignment_capacities_train", "client_task_assignment_capacities_test",
                           "client_tasks_per_class_train", "client_tasks_per_class_test"]
        if any(task_property in config for task_property in task_properties):
            samples_per_task = config["samples_per_task"]
            # Get the task assignment capacities.
            task_assignment_capacities_settings = self.get_attribute("_task_assignment_capacities_settings")
            client_task_assignment_capacities_train, client_task_assignment_capacities_test \
                = get_task_assignment_capacities(self._x_train,
                                                 self._x_test,
                                                 task_assignment_capacities_settings,
                                                 samples_per_task)
            # Get the tasks' occurrence per class.
            client_tasks_per_class_train = get_classes_distribution(self._y_train)
            client_tasks_per_class_test = get_classes_distribution(self._y_test)
            if "client_task_assignment_capacities_train" in config:
                client_task_assignment_capacities_train_str \
                    = "|".join([str(capacity) for capacity in client_task_assignment_capacities_train])
                config.update({"client_task_assignment_capacities_train": client_task_assignment_capacities_train_str})
            if "client_task_assignment_capacities_test" in config:
                client_task_assignment_capacities_test_str \
                    = "|".join([str(capacity) for capacity in client_task_assignment_capacities_test])
                config.update({"client_task_assignment_capacities_test": client_task_assignment_capacities_test_str})
            if "client_tasks_per_class_train" in config:
                client_tasks_per_class_train_str = "|".join([str(k) + "=" + str(v)
                                                             for k, v in client_tasks_per_class_train.items()])
                config.update({"client_tasks_per_class_train": client_tasks_per_class_train_str})
            if "client_tasks_per_class_test" in config:
                client_tasks_per_class_test_str = "|".join([str(k) + "=" + str(v)
                                                            for k, v in client_tasks_per_class_test.items()])
                config.update({"client_tasks_per_class_test": client_tasks_per_class_test_str})
        if "client_remaining_battery_energy" in config:
            remaining_battery_energy_file = self.get_attribute("_remaining_battery_energy_file")
            remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
            config.update({"client_remaining_battery_energy": remaining_battery_energy_in_joules})
        if "client_mean_power_consumption_idle_mode" in config:
            device_emulation_settings = self.get_attribute("_device_emulation_settings")
            mpc_idle_i = device_emulation_settings["mean_power_consumption_idle_in_watts"]
            config.update({"client_mean_power_consumption_idle_mode": mpc_idle_i})
        if "client_current_download_bandwidth_in_bytes_per_second" in config:
            device_emulation_settings = self.get_attribute("_device_emulation_settings")
            network_simulator = self.get_attribute("_network_simulator")
            # Simulate network performance change.
            _, current_download_bandwidth = network_simulator.simulate_network_performance_change()
            # Save the current state of the network simulator.
            self._set_attribute("_network_simulator", network_simulator)
            # Convert the current download bandwidth to Bytes per second.
            download_bandwidth_unit = device_emulation_settings["download_bandwidth_unit"]
            current_download_bandwidth_in_bytes_per_second = convert_network_bandwidth(current_download_bandwidth,
                                                                                       download_bandwidth_unit,
                                                                                       "Bps")
            config.update({"client_current_download_bandwidth_in_bytes_per_second":
                               current_download_bandwidth_in_bytes_per_second})
        if "client_current_upload_bandwidth_in_bytes_per_second" in config:
            device_emulation_settings = self.get_attribute("_device_emulation_settings")
            network_simulator = self.get_attribute("_network_simulator")
            # Simulate network performance change.
            current_upload_bandwidth, _ = network_simulator.simulate_network_performance_change()
            # Save the current state of the network simulator.
            self._set_attribute("_network_simulator", network_simulator)
            # Convert the current upload bandwidth to Bytes per second.
            upload_bandwidth_unit = device_emulation_settings["upload_bandwidth_unit"]
            current_upload_bandwidth_in_bytes_per_second = convert_network_bandwidth(current_upload_bandwidth,
                                                                                     upload_bandwidth_unit,
                                                                                     "Bps")
            config.update({"client_current_upload_bandwidth_in_bytes_per_second":
                               current_upload_bandwidth_in_bytes_per_second})
        if "client_current_latency_in_milliseconds" in config:
            device_emulation_settings = self.get_attribute("_device_emulation_settings")
            network_simulator = self.get_attribute("_network_simulator")
            # Simulate network latency change.
            current_latency = network_simulator.simulate_latency_change()
            # Save the current state of the network simulator.
            self._set_attribute("_network_simulator", network_simulator)
            # Convert the current latency to milliseconds.
            base_latency_unit = device_emulation_settings["base_latency_unit"]
            current_latency_in_milliseconds = convert_duration(current_latency, base_latency_unit, "ms")
            config.update({"client_current_latency_in_milliseconds": current_latency_in_milliseconds})
        # Return the properties requested by the server.
        return config

    def _estimate_initial_parameters_upload_time_in_seconds(self,
                                                            model: Model) -> float:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        network_simulator = self.get_attribute("_network_simulator")
        # Simulate network performance change, latency change, and packet loss event occurrence.
        current_upload_bandwidth, _ = network_simulator.simulate_network_performance_change()
        current_latency = network_simulator.simulate_latency_change()
        packet_loss_event_occurred = network_simulator.simulate_packet_loss_event()
        # Save the current state of the network simulator.
        self._set_attribute("_network_simulator", network_simulator)
        # Get the packet loss rate of the device's network.
        packet_loss_rate = device_emulation_settings["packet_loss_rate"]
        # Set the dictionary of client attributes to be used for the estimation.
        client_attributes = {"upload_bandwidth": current_upload_bandwidth,
                             "upload_bandwidth_unit": device_emulation_settings["upload_bandwidth_unit"],
                             "base_latency": current_latency,
                             "base_latency_unit": device_emulation_settings["base_latency_unit"],
                             "mss_ipv4_in_bytes": device_emulation_settings["mss_ipv4_in_bytes"],
                             "server_location": device_emulation_settings["server_location"],
                             "client_location": device_emulation_settings["client_location"]}
        # Estimate the time needed to upload the local model parameters to the server.
        initial_parameters_upload_time_in_seconds = calculate_initial_parameters_upload_time(client_attributes,
                                                                                             model,
                                                                                             packet_loss_event_occurred,
                                                                                             packet_loss_rate)
        # Return the estimated initial parameters upload time (in seconds).
        return initial_parameters_upload_time_in_seconds

    def get_parameters(self,
                       config: dict) -> NDArrays:
        """ Implementation of the abstract method from the NumPyClient class."""
        # Load the local model from the file.
        model_file = self.get_attribute("_model_file")
        model = load_model_from_file(model_file)
        # Estimate the time spent by this client during the initial parameters upload event.
        initial_parameters_upload_time_in_seconds = self._estimate_initial_parameters_upload_time_in_seconds(model)
        # Estimate the energy consumed by this client during the initial parameters upload event.
        initial_parameters_upload_energy_in_joules = self._estimate_upload_energy_in_joules(initial_parameters_upload_time_in_seconds)
        # Record the energy consumed by this client during the initial parameters upload event.
        comm_round = 0
        phase = "initialization"
        initial_parameters_upload_event = "after_sending_initial_parameters_to_the_server"
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  initial_parameters_upload_event,
                                                  initial_parameters_upload_energy_in_joules)
        # Get the initial model parameters.
        local_model_parameters = model.get_weights()
        # Return the current parameters (weights) of the local model requested by the server.
        return local_model_parameters

    @staticmethod
    def _get_client_selection_time_in_seconds(config: dict) -> float:
        client_selection_time_in_seconds = config["client_selection_time_in_seconds"]
        return client_selection_time_in_seconds

    @staticmethod
    def _append_data_to_file(output_file: Path,
                             data: str) -> None:
        with open(file=output_file, mode="a", encoding="utf-8") as of:
            of.write(data)

    @staticmethod
    def _read_client_metrics_from_file(input_file: Path,
                                       comm_round: int,
                                       columns_to_remove: list = None) -> dict:
        # Load a dataframe from the input file (.csv).
        df = read_csv(input_file)
        # Filter by comm_round.
        df = df[df["comm_round"] == comm_round]
        # Remove data columns, if any.
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
        # Get the client metrics.
        client_metrics = df.to_dict(orient="records")[0]
        # Return the client metrics.
        return client_metrics

    def _estimate_download_time_in_seconds(self,
                                           down_metrics_file: Path,
                                           comm_round: int,
                                           phase: str,
                                           global_parameters: NDArrays,
                                           phase_config: dict) -> float:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        network_simulator = self.get_attribute("_network_simulator")
        # Simulate network performance change, latency change, and packet loss event occurrence.
        _, current_download_bandwidth = network_simulator.simulate_network_performance_change()
        current_latency = network_simulator.simulate_latency_change()
        packet_loss_event_occurred = network_simulator.simulate_packet_loss_event()
        # Save the current state of the network simulator.
        self._set_attribute("_network_simulator", network_simulator)
        # Get the packet loss rate of the device's network.
        packet_loss_rate = device_emulation_settings["packet_loss_rate"]
        # Set the dictionary of client attributes to be used for the estimation.
        client_attributes = {"download_bandwidth": current_download_bandwidth,
                             "download_bandwidth_unit": device_emulation_settings["download_bandwidth_unit"],
                             "base_latency": current_latency,
                             "base_latency_unit": device_emulation_settings["base_latency_unit"],
                             "mss_ipv4_in_bytes": device_emulation_settings["mss_ipv4_in_bytes"],
                             "server_location": device_emulation_settings["server_location"],
                             "client_location": device_emulation_settings["client_location"]}
        # Estimate the time needed to download the global model parameters and instruction from the server.
        download_time_in_seconds, client_down_metrics = calculate_download_time(client_attributes,
                                                                                phase,
                                                                                global_parameters,
                                                                                phase_config,
                                                                                packet_loss_event_occurred,
                                                                                packet_loss_rate)
        # Save the client download metrics to file.
        if not down_metrics_file.is_file():
            # Append the header line.
            down_header_data = "comm_round," + ",".join(list(client_down_metrics.keys())) + "\n"
            self._append_data_to_file(down_metrics_file, down_header_data)
        down_metrics_data = "{0},".format(comm_round) + ",".join([str(down_metric) for down_metric in list(client_down_metrics.values())]) + "\n"
        self._append_data_to_file(down_metrics_file, down_metrics_data)
        # Return the estimated download time (in seconds).
        return download_time_in_seconds

    def _estimate_download_energy_in_joules(self,
                                            download_time_in_seconds: float) -> float:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        # Set the dictionary of client attributes to be used for the estimation.
        client_attributes = {"mpc_recv_i": device_emulation_settings["mean_power_consumption_data_reception_in_watts"]}
        # Estimate the energy consumed by this client during the download event of round r.
        download_energy_in_joules = calculate_download_energy(client_attributes, download_time_in_seconds)
        # Return the estimated download energy (in joules).
        return download_energy_in_joules

    def _estimate_upload_time_in_seconds(self,
                                         up_metrics_file: Path,
                                         comm_round: int,
                                         phase: str,
                                         phase_metrics: dict,
                                         local_parameters: NDArrays = None) -> float:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        network_simulator = self.get_attribute("_network_simulator")
        # Simulate network performance change, latency change, and packet loss event occurrence.
        current_upload_bandwidth, _ = network_simulator.simulate_network_performance_change()
        current_latency = network_simulator.simulate_latency_change()
        packet_loss_event_occurred = network_simulator.simulate_packet_loss_event()
        # Save the current state of the network simulator.
        self._set_attribute("_network_simulator", network_simulator)
        # Get the packet loss rate of the device's network.
        packet_loss_rate = device_emulation_settings["packet_loss_rate"]
        # Set the dictionary of client attributes to be used for the estimation.
        client_attributes = {"upload_bandwidth": current_upload_bandwidth,
                             "upload_bandwidth_unit": device_emulation_settings["upload_bandwidth_unit"],
                             "base_latency": current_latency,
                             "base_latency_unit": device_emulation_settings["base_latency_unit"],
                             "mss_ipv4_in_bytes": device_emulation_settings["mss_ipv4_in_bytes"],
                             "server_location": device_emulation_settings["server_location"],
                             "client_location": device_emulation_settings["client_location"]}
        # Estimate the time needed to upload the local model parameters and metrics to the server.
        upload_time_in_seconds, client_up_metrics = calculate_upload_time(client_attributes,
                                                                          phase,
                                                                          phase_metrics,
                                                                          packet_loss_event_occurred,
                                                                          packet_loss_rate,
                                                                          local_parameters)
        # Save the client upload metrics to file.
        if not up_metrics_file.is_file():
            # Append the header line.
            up_header_data = "comm_round," + ",".join(list(client_up_metrics.keys())) + "\n"
            self._append_data_to_file(up_metrics_file, up_header_data)
        up_metrics_data = "{0},".format(comm_round) + ",".join([str(up_metric) for up_metric in list(client_up_metrics.values())]) + "\n"
        self._append_data_to_file(up_metrics_file, up_metrics_data)
        # Return the estimated upload time (in seconds).
        return upload_time_in_seconds

    def _estimate_upload_energy_in_joules(self,
                                          upload_time_in_seconds: float) -> float:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        # Set the dictionary of client attributes to be used for the estimation.
        client_attributes = {"mpc_send_i": device_emulation_settings["mean_power_consumption_data_transmission_in_watts"]}
        # Estimate the energy consumed by this client during the upload event of round r.
        upload_energy_in_joules = calculate_upload_energy(client_attributes, upload_time_in_seconds)
        # Return the estimated upload energy (in joules).
        return upload_energy_in_joules

    def _training_task(self,
                       global_parameters: NDArrays,
                       x_train: NDArray,
                       y_train: NDArray,
                       fit_config: dict,
                       fit_callbacks: list,
                       fit_queue: Queue) -> None:
        # Start the timer.
        time_start = datetime.now()
        # Load the local model from the file.
        model_file = self.get_attribute("_model_file")
        model = load_model_from_file(model_file)
        # Update the parameters (weights) of the local model with those received from the server (global parameters).
        model.set_weights(global_parameters)
        # Log the local model weights' sum (before training).
        message = ("[Client {0} | Round {1}] Local model weight's sum (before training): {2}"
                   .format(self._client_id, fit_config["comm_round"], sum(model.get_weights()[0])))
        log_message(self._logger, message, "DEBUG")
        # Train the local model using the local training dataset slice.
        history = model.fit(x=x_train,
                            y=y_train,
                            shuffle=fit_config["shuffle"],
                            batch_size=fit_config["batch_size"],
                            initial_epoch=fit_config["initial_epoch"],
                            epochs=fit_config["epochs"],
                            steps_per_epoch=fit_config["steps_per_epoch"],
                            validation_split=fit_config["validation_split"],
                            validation_batch_size=fit_config["validation_batch_size"],
                            verbose=fit_config["verbose"],
                            callbacks=fit_callbacks).history
        # Dump the local model (with updated parameters from the training) to file.
        model_file = self.get_attribute("_model_file")
        save_model_to_file(model, model_file)
        # Stop the timer.
        actual_computation_time_in_seconds = (datetime.now() - time_start).total_seconds()
        # Put the model training result into the fit_queue.
        model_training_result = {"history": history,
                                 "actual_computation_time_in_seconds": actual_computation_time_in_seconds}
        fit_queue.put({"model_training_result": model_training_result})

    def _train_and_profile(self,
                           perf_log_file: Path,
                           model_metrics_file: Path,
                           comp_metrics_file: Path,
                           comm_round: int,
                           phase: str,
                           global_parameters: NDArrays,
                           x_train: NDArray,
                           y_train: NDArray,
                           fit_config: dict,
                           fit_callbacks: list) -> float:
        # Set the starting method of daemon processes.
        start_method = "forkserver"
        set_start_method(start_method, force=True)
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        # Get the device's memory bandwidth peak (in bytes per seconds).
        device_peak_memory_bandwidth_in_Bps = device_emulation_settings["peak_memory_bandwidth_in_bytes_per_second"]
        # Initializations.
        actual_computation_time_in_seconds = 0
        total_float_ops = 0
        x_size = len(x_train)
        learning_rate = fit_config["learning_rate"]
        batch_size = fit_config["batch_size"]
        epochs = fit_config["epochs"]
        client_model_metrics = {}
        if "use_replay_values" in fit_config:
            client_model_metrics = fit_config["client_model_metrics_replay"]
            fpo_train_m = fit_config["fpo_train_m"]
            mt_m_node = fit_config["mt_m_node"]
            mem_bw_node = fit_config["mem_bw_node"]
            cpu_gflops_node = fit_config["cpu_gflops_node"]
        else:
            # Set the perf settings.
            perf_events = ["cache-misses", "cache-references", "instructions", "cycles", "LLC-load-misses",
                           "LLC-store-misses"]
            perf_events_to_extract = ["cache-misses", "cache-references", "instructions", "cycles", "LLC-load-misses",
                                      "LLC-store-misses", "seconds time elapsed"]
            perf_training_process = None
            # Initialize the model training queue (fit_queue).
            fit_queue = Queue()
            # Launch the training process.
            training_process_target = self._training_task
            training_process_args = (global_parameters,
                                     x_train,
                                     y_train,
                                     fit_config,
                                     fit_callbacks,
                                     fit_queue)
            training_process = Process(target=training_process_target, args=training_process_args)
            training_process.start()
            # Check if perf is available.
            perf_is_available = check_if_perf_is_available()
            # Launch the perf process, if perf is available.
            if perf_is_available:
                perf_training_process = launch_perf_process(training_process.pid, perf_events, perf_log_file)
            # Wait for the training process to finish.
            training_process.join()
            # Stop the perf process, if launched.
            if perf_training_process:
                stop_process(perf_training_process)
            # Get the model training result.
            fit_queue_element = fit_queue.get()
            model_training_result = fit_queue_element["model_training_result"]
            history = model_training_result["history"]
            # Get the client model metrics names.
            client_model_metrics_names = history.keys()
            # Store the client model metrics of the last epoch.
            for client_model_metric_name in client_model_metrics_names:
                client_model_metrics.update({client_model_metric_name: history[client_model_metric_name][-1]})
            actual_computation_time_in_seconds = model_training_result["actual_computation_time_in_seconds"]
            # Load the local model from the file.
            model_file = self.get_attribute("_model_file")
            model = load_model_from_file(model_file)
            # Get the model settings.
            model_settings = self.get_attribute("_model_settings")
            # Get the input shape of the model.
            model_input_shape = get_model_input_shape(model)
            # Get the number of output classes of the model.
            model_num_output_classes = get_model_output_classes(model)
            # Generate a batch of dummy samples for training (size of batch_size).
            x_train_dummy, y_train_dummy = generate_dummy_sample_batch(batch_size,
                                                                       model_input_shape,
                                                                       model_num_output_classes,
                                                                       model,
                                                                       model_settings)
            # Build the concrete training function that will be used by the profiler.
            profiling_module = ProfilingModule(model)
            silent_profiler_options = profiling_module.create_silent_profiler_options()
            train_step_fn = profiling_module.build_train_fn(x_train_dummy, y_train_dummy)
            concrete_train_fn = train_step_fn.get_concrete_function(x_train_dummy, y_train_dummy)
            # Profile the model using the concrete_train_fn function and the dummy samples.
            graph_info = profile(concrete_train_fn.graph, options=silent_profiler_options)
            # Collect the total float ops per batch of the profiling (dummy samples have size of batch_size).
            float_ops_per_batch = graph_info.total_float_ops
            # Get the float operations performance statistics of the profiling.
            flop_performance_statistics_training = get_flop_performance_statistics(phase,
                                                                                   float_ops_per_batch,
                                                                                   x_size,
                                                                                   batch_size,
                                                                                   epochs)
            fpo_train_m = flop_performance_statistics_training["float_ops_per_sample"]
            total_float_ops = flop_performance_statistics_training["total_float_ops"]
            # Parse the perf log file, if exists.
            perf_events_metrics = {"memory_traffic_per_flop_in_bytes": 1}
            if perf_log_file.is_file():
                parsed_perf_log = parse_perf_log_file(perf_log_file, perf_events_to_extract)
                perf_events_metrics = summarize_perf_metrics(parsed_perf_log)
                if all(perf_event_metric_key in perf_events_metrics
                       for perf_event_metric_key in ["memory_traffic_in_bytes", "memory_bandwidth_usage_in_Bps"]):
                    # Measure how many bytes were moved per floating point operation during the training.
                    total_memory_traffic_in_bytes = perf_events_metrics["memory_traffic_in_bytes"]
                    mt_m_node = get_memory_traffic_per_flop_in_bytes(total_memory_traffic_in_bytes, total_float_ops)
                    # Get the memory efficiency rate during the training.
                    memory_bandwidth_usage_in_Bps = perf_events_metrics["memory_bandwidth_usage_in_Bps"]
                    memory_efficiency_rate = get_memory_efficiency_rate(device_peak_memory_bandwidth_in_Bps,
                                                                        memory_bandwidth_usage_in_Bps)
                    # Update the perf events metrics dictionary.
                    perf_events_metrics["memory_traffic_per_flop_in_bytes"] = mt_m_node
                    perf_events_metrics["memory_efficiency_rate"] = memory_efficiency_rate
            # Get the memory traffic per floating point operation during the training (in bytes).
            mt_m_node = perf_events_metrics["memory_traffic_per_flop_in_bytes"]
            # Get the peak memory bandwidth and theoretical CPU GFLOPs of the executing node's hardware.
            node_info = get_cpu_and_memory_info()
            mem_bw_node = node_info["peak_memory_bandwidth_in_GBps"]
            cpu_gflops_node = node_info["theoretical_cpu_gflops"]
        # Get the device's CPU number of cores and frequency (in hertz).
        device_cpu_num_cores = device_emulation_settings["cpu_num_cores"]
        device_cpu_frequency_in_hertz = device_emulation_settings["cpu_frequency_in_hertz"]
        # Select randomly the device's float operations per cycle per CPU core performance during the training.
        seed = self._deterministic_seed(client_id, comm_round)
        rng = default_rng(seed)
        fpocc_training = device_emulation_settings["fpocc_training"]
        device_cpu_float_ops_per_cycle_per_core = rng.choice(linspace(fpocc_training[0], fpocc_training[1], num=1000))
        # Estimate the device's memory traffic per floating point operation during the training (in bytes).
        mem_bw_device = device_emulation_settings["peak_memory_bandwidth_in_gigabytes_per_second"]
        cpu_gflops_device = device_emulation_settings["peak_cpu_gflops"]
        device_memory_traffic_per_flop_in_bytes = estimate_mt_m_device(mt_m_node,
                                                                       mem_bw_node,
                                                                       cpu_gflops_node,
                                                                       mem_bw_device,
                                                                       cpu_gflops_device)
        # Add the number of examples, learning rate, batch size, and epochs to the client model metrics.
        client_model_metrics.update({"num_examples": x_size,
                                     "learning_rate": learning_rate,
                                     "batch_size": batch_size,
                                     "epochs": epochs})
        # Save the client model metrics to file.
        if not model_metrics_file.is_file():
            # Append the header line.
            model_header_data = "comm_round," + ",".join(list(client_model_metrics.keys())) + "\n"
            self._append_data_to_file(model_metrics_file, model_header_data)
        model_metrics_data = "{0},".format(comm_round) + ",".join([str(model_metric) for model_metric in list(client_model_metrics.values())]) + "\n"
        self._append_data_to_file(model_metrics_file, model_metrics_data)
        # Initialize the set of client computation metrics.
        client_comp_metrics = {"fpo_train_m": fpo_train_m,
                               "total_float_ops": total_float_ops,
                               "bs_train_i": batch_size,
                               "ds_train_i": x_size,
                               "e_i": epochs,
                               "nc_cpu_i": device_cpu_num_cores,
                               "fq_cpu_i": device_cpu_frequency_in_hertz,
                               "fpocc_m_i_train": device_cpu_float_ops_per_cycle_per_core,
                               "bw_mem_i": device_peak_memory_bandwidth_in_Bps,
                               "mt_m_train": device_memory_traffic_per_flop_in_bytes}
        # Save the client computation metrics to file.
        if not comp_metrics_file.is_file():
            # Append the header line.
            comp_header_data = "comm_round," + ",".join(list(client_comp_metrics.keys())) + "\n"
            self._append_data_to_file(comp_metrics_file, comp_header_data)
        comp_metrics_data = "{0},".format(comm_round) + ",".join([str(comp_metric) for comp_metric in list(client_comp_metrics.values())]) + "\n"
        self._append_data_to_file(comp_metrics_file, comp_metrics_data)
        # Return the actual computation time (in seconds).
        return actual_computation_time_in_seconds

    def _estimate_device_computation_time_in_seconds_via_scaling(self,
                                                                 perf_log_file: Path,
                                                                 comp_metrics_file: Path,
                                                                 comm_round: int,
                                                                 host_actual_computation_time_in_seconds: float,
                                                                 scaling_approach: str = "flop") -> float:
        # Get the set of client computation metrics from the corresponding file.
        client_comp_metrics = self._read_client_metrics_from_file(comp_metrics_file, comm_round)
        total_float_ops = client_comp_metrics["total_float_ops"]
        # Get the client's host profile dictionary.
        host_profile = self.get_attribute("_host_profile")
        # Get the client's device emulation dictionary.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        # Get the 'alpha' parameter value (cpu_ratio x mem_ratio).
        alpha = 0.7  # Default fallback for the alpha parameter.
        perf_events_to_extract = ["memory_traffic_in_bytes", "sustained_gflops", "bandwidth_gbs"]
        if perf_log_file.is_file():
            parsed_perf_log = parse_perf_log_file(perf_log_file, perf_events_to_extract)
            perf_events_metrics = summarize_perf_metrics(parsed_perf_log)
            if total_float_ops > 0:
                bytes_moved = perf_events_metrics.get("memory_traffic_in_bytes", 1)
                operational_intensity = total_float_ops / bytes_moved
                peak_flops_per_s = host_profile["compute"]["sustained_gflops"] * 1e9
                peak_bw_per_s = max(1e9, host_profile["memory"]["bandwidth_gbs"] * 1e9)
                machine_balance = peak_flops_per_s / peak_bw_per_s
                alpha = operational_intensity / (operational_intensity + machine_balance)
                alpha = float(clip(alpha, 0.05, 0.95))
                cache_misses = perf_events_metrics.get("cache-misses", 0)
                cache_references = perf_events_metrics.get("cache-references", 1)
                miss_ratio = cache_misses / cache_references
                alpha *= max(0.1, (1 - miss_ratio))
        # Compute actual achieved GFLOPS on host.
        actual_host_gflops = total_float_ops / (host_actual_computation_time_in_seconds * 1e9)
        eta_host = actual_host_gflops / host_profile["compute"]["sustained_gflops"]
        eta_device = eta_host * 0.8
        # Macro scaling approach.
        cpu_ratio = host_profile["compute"]["sustained_gflops"] / device_emulation_settings["peak_cpu_gflops"]
        mem_ratio = host_profile["memory"]["bandwidth_gbs"] / device_emulation_settings["peak_memory_bandwidth_in_gigabytes_per_second"]
        scaled_device_computation_time_macro = host_actual_computation_time_in_seconds * (alpha * cpu_ratio + (1 - alpha) * mem_ratio)
        # FLOP-based scaling approach.
        scaled_device_computation_time_flop = total_float_ops / (device_emulation_settings["peak_cpu_gflops"] * 1e9 * eta_device)
        # Get the client's device scaled computation time dictionary.
        scaled_device_computation_time_dict = {"scaled_device_computation_time_macro": scaled_device_computation_time_macro,
                                               "scaled_device_computation_time_flop": scaled_device_computation_time_flop,
                                               "eta_host": eta_host,
                                               "eta_device": eta_device,
                                               "actual_host_gflops": actual_host_gflops}
        # Get the client's device scaled computation time (in seconds).
        estimate_device_computation_time_in_seconds = 0
        match scaling_approach:
            case "macro":
                estimate_device_computation_time_in_seconds \
                    = scaled_device_computation_time_dict["scaled_device_computation_time_macro"]
            case "flop":
                estimate_device_computation_time_in_seconds \
                    = scaled_device_computation_time_dict["scaled_device_computation_time_flop"]
        # Return the estimated computation time (in seconds).
        return estimate_device_computation_time_in_seconds

    def _estimate_computation_time_in_seconds(self,
                                              comp_metrics_file: Path,
                                              comm_round: int,
                                              phase: str) -> float:
        # Get the set of client computation metrics from the corresponding file.
        client_comp_metrics = self._read_client_metrics_from_file(comp_metrics_file, comm_round)
        # Estimate the time spent by this client during the computation event of round r.
        computation_time_in_seconds = calculate_computation_time(client_comp_metrics, phase)
        # Return the estimated computation time (in seconds).
        return computation_time_in_seconds

    def _estimate_computation_energy_in_joules(self,
                                               computation_time_in_seconds: float) -> float:
        # Get the necessary attributes.
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        # Set the dictionary of client attributes to be used for the estimation.
        client_attributes = {"mpc_comp_i": device_emulation_settings["mean_power_consumption_heavy_computational_load_in_watts"]}
        # Estimate the energy consumed by this client during the computation event of round r.
        computation_energy_in_joules = calculate_computation_energy(client_attributes, computation_time_in_seconds)
        # Return the estimated computation energy (in joules).
        return computation_energy_in_joules

    def fit(self,
            global_parameters: NDArrays,
            fit_config: dict) -> tuple[NDArrays, int, dict]:
        """ Implementation of the abstract method from the NumPyClient class."""
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        x_train = self.get_attribute("_x_train")
        y_train = self.get_attribute("_y_train")
        remaining_battery_energy_file = self.get_attribute("_remaining_battery_energy_file")
        logger = self.get_attribute("_logger")
        work_dir_training_phase = self.get_attribute("_work_dir_training_phase")
        training_callbacks = self.get_attribute("_training_callbacks")
        training_measurements_callback = self.get_attribute("_training_measurements_callback")
        # Set the callbacks list.
        fit_callbacks = training_callbacks + [training_measurements_callback]
        # Set the phase.
        phase = "train"
        # Initialize the training metrics dictionary.
        training_metrics = {}
        # Get the current communication round.
        comm_round = fit_config["comm_round"]
        # Set the output files.
        down_metrics_file_name = "down_metrics.csv"
        down_metrics_file = Path(work_dir_training_phase).joinpath(down_metrics_file_name).absolute()
        perf_log_file_name = "perf.log"
        perf_log_file = Path(work_dir_training_phase).joinpath(perf_log_file_name).absolute()
        model_metrics_file_name = "model_metrics.csv"
        model_metrics_file = Path(work_dir_training_phase).joinpath(model_metrics_file_name).absolute()
        comp_metrics_file_name = "comp_metrics.csv"
        comp_metrics_file = Path(work_dir_training_phase).joinpath(comp_metrics_file_name).absolute()
        up_metrics_file_name = "up_metrics.csv"
        up_metrics_file = Path(work_dir_training_phase).joinpath(up_metrics_file_name).absolute()
        # Create the parents directories of the output files (if not exist yet).
        down_metrics_file.parent.mkdir(exist_ok=True, parents=True)
        perf_log_file.parent.mkdir(exist_ok=True, parents=True)
        model_metrics_file.parent.mkdir(exist_ok=True, parents=True)
        comp_metrics_file.parent.mkdir(exist_ok=True, parents=True)
        up_metrics_file.parent.mkdir(exist_ok=True, parents=True)
        # Slice the training dataset, according to server's request.
        rrds = self.get_attribute("_rrds")
        if "num_training_examples_per_class_to_use" in fit_config.keys():
            num_training_examples_per_class_to_use_str = fit_config["num_training_examples_per_class_to_use"]
            num_training_examples_per_class_to_use = {k: int(v) for k, v in (pair.split("=") for pair in num_training_examples_per_class_to_use_str.split("|"))}
            x_train, y_train = rrds.slice_round_robin_per_class(num_training_examples_per_class_to_use, phase)
        elif "num_training_examples_to_use" in fit_config.keys():
            num_training_examples_to_use = fit_config["num_training_examples_to_use"]
            x_train, y_train = rrds.slice_round_robin_randomly(num_training_examples_to_use, phase)
        # Replace 'None' values to None (necessary workaround on Flower).
        fit_config = {k: (None if v == "None" else v) for k, v in fit_config.items()}
        # Log the training configuration (fit_config) received from the server.
        message = "[Client {0} | Round {1}] Received fit_config: {2}".format(client_id, comm_round, fit_config)
        log_message(logger, message, "DEBUG")
        # Get the time spent by the client selection event of round r.
        client_selection_time_in_seconds = self._get_client_selection_time_in_seconds(fit_config)
        # Estimate the energy consumed by this client during the client selection event of round r.
        client_selection_energy_in_joules = self._estimate_idle_energy_in_joules(client_selection_time_in_seconds)
        # Record the energy consumed by this client during the client selection event of round r.
        client_selection_event = "after_idle_during_the_client_selection"
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  client_selection_event,
                                                  client_selection_energy_in_joules)
        # Get the remaining battery energy in joules.
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules == 0:
            # Client has dropped due to lack of battery level.
            return [], 0, {}
        # Estimate the time spent by this client during the download event of round r.
        download_time_in_seconds = self._estimate_download_time_in_seconds(down_metrics_file,
                                                                           comm_round,
                                                                           phase,
                                                                           global_parameters,
                                                                           fit_config)
        # Get the set of client download metrics from the corresponding file.
        client_down_metrics = self._read_client_metrics_from_file(down_metrics_file, comm_round)
        # Estimate the energy consumed by this client during the download event of round r.
        download_energy_in_joules = self._estimate_download_energy_in_joules(download_time_in_seconds)
        # Record the energy consumed by this client during the download event of round r.
        download_event = "after_downloading_global_parameters_and_{0}ing_instructions_from_the_server".format(phase)
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  download_event,
                                                  download_energy_in_joules)
        # Get the remaining battery energy in joules.
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules == 0:
            # Client has dropped due to lack of battery level.
            return [], 0, {}
        # Log a 'training my model' message.
        message = "[Client {0} | Round {1}] Training my model...".format(client_id, comm_round)
        log_message(logger, message, "INFO")
        # Unset the logger.
        self._set_attribute("_logger", None)
        # Train and profile.
        actual_computation_time_in_seconds = self._train_and_profile(perf_log_file,
                                                                     model_metrics_file,
                                                                     comp_metrics_file,
                                                                     comm_round,
                                                                     phase,
                                                                     global_parameters,
                                                                     x_train,
                                                                     y_train,
                                                                     fit_config,
                                                                     fit_callbacks)
        # Get the set of client model metrics from the corresponding file.
        client_model_metrics = self._read_client_metrics_from_file(model_metrics_file, comm_round)
        # Get the set of client computation metrics from the corresponding file.
        client_comp_metrics = self._read_client_metrics_from_file(comp_metrics_file, comm_round)
        # Estimate the time spent by this client during the computation event of round r.
        computation_time_in_seconds = self._estimate_device_computation_time_in_seconds_via_scaling(perf_log_file,
                                                                                                    comp_metrics_file,
                                                                                                    comm_round,
                                                                                                    actual_computation_time_in_seconds)
        # Estimate the energy consumed by this client during the computation event of round r.
        computation_energy_in_joules = self._estimate_computation_energy_in_joules(computation_time_in_seconds)
        # Record the energy consumed by this client during the computation event of round r.
        computation_event = "after_local_training"
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  computation_event,
                                                  computation_energy_in_joules)
        # Get the remaining battery energy in joules.
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules == 0:
            # Client has dropped due to lack of battery level.
            return [], 0, {}
        # Append the client download metrics to the set of training metrics.
        training_metrics = training_metrics | client_down_metrics
        # Append the client model metrics to the set of training metrics.
        training_metrics = training_metrics | client_model_metrics
        # Append the client computation metrics to the set of training metrics.
        training_metrics = training_metrics | client_comp_metrics
        # Get the number of examples used.
        num_examples = client_model_metrics["num_examples"]
        # Get the parameters (weights) of the local model after the training (dumped to file).
        model_file = self.get_attribute("_model_file")
        model = load_model_from_file(model_file)
        local_model_parameters = model.get_weights()
        # Estimate the time spent by this client during the upload event of round r.
        upload_time_in_seconds = self._estimate_upload_time_in_seconds(up_metrics_file,
                                                                       comm_round,
                                                                       phase,
                                                                       training_metrics,
                                                                       local_model_parameters)
        # Get the set of client upload metrics from the corresponding file.
        client_up_metrics = self._read_client_metrics_from_file(up_metrics_file, comm_round)
        # Estimate the energy consumed by this client during the upload event of round r.
        upload_energy_in_joules = self._estimate_upload_energy_in_joules(upload_time_in_seconds)
        # Record the energy consumed by this client during the upload event of round r.
        upload_event = "after_uploading_local_parameters_and_{0}ing_metrics_to_the_server".format(phase)
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  upload_event,
                                                  upload_energy_in_joules)
        # Get the remaining battery energy in joules.
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules == 0:
            # Client has dropped due to lack of battery level.
            return [], 0, {}
        # Append the client upload metrics to the set of training metrics.
        training_metrics = training_metrics | client_up_metrics
        # Calculate the total time spent by this client during the training phase event of round r.
        training_time_in_seconds = download_time_in_seconds + computation_time_in_seconds + upload_time_in_seconds
        # Calculate the total energy consumed by this client during the training phase event of round r.
        training_energy_in_joules = download_energy_in_joules + computation_energy_in_joules + upload_energy_in_joules
        # Append the mean power consumptions to the training metrics.
        mpc_idle_i = device_emulation_settings["mean_power_consumption_idle_in_watts"]
        mpc_recv_i = device_emulation_settings["mean_power_consumption_data_reception_in_watts"]
        mpc_comp_i = device_emulation_settings["mean_power_consumption_heavy_computational_load_in_watts"]
        mpc_send_i = device_emulation_settings["mean_power_consumption_data_transmission_in_watts"]
        training_metrics.update({"mpc_idle_i": mpc_idle_i,
                                 "mpc_recv_i": mpc_recv_i,
                                 "mpc_comp_i": mpc_comp_i,
                                 "mpc_send_i": mpc_send_i})
        # Append the times and energy consumptions to the training metrics.
        training_metrics.update({"download_time_in_seconds": download_time_in_seconds,
                                 "download_energy_in_joules": download_energy_in_joules,
                                 "computation_time_in_seconds": computation_time_in_seconds,
                                 "computation_energy_in_joules": computation_energy_in_joules,
                                 "upload_time_in_seconds": upload_time_in_seconds,
                                 "upload_energy_in_joules": upload_energy_in_joules,
                                 "training_time_in_seconds": training_time_in_seconds,
                                 "training_energy_in_joules": training_energy_in_joules})
        # Set the logger.
        self._set_attribute("_logger", logger)
        # Log a 'finished training my model' message.
        message = ("[Client {0} | Round {1}] Finished training my model... "
                   "Download costs -> time: {2} seconds | energy: {3} joules; "
                   "Computation costs -> time: {4} seconds | energy: {5} joules; "
                   "Upload costs -> time: {6} seconds | energy: {7} joules; "
                   "Total costs -> time: {8} seconds | energy: {9} joules."
                  .format(client_id,
                          comm_round,
                          round(download_time_in_seconds, 2),
                          round(download_energy_in_joules, 2),
                          round(computation_time_in_seconds, 2),
                          round(computation_energy_in_joules, 2),
                          round(upload_time_in_seconds, 2),
                          round(upload_energy_in_joules, 2),
                          round(training_time_in_seconds, 2),
                          round(training_energy_in_joules, 2)))
        log_message(logger, message, "INFO")
        # Log the local model weights' sum (after training).
        message = ("[Client {0} | Round {1}] Local model weight's sum (after training): {2}"
                   .format(self._client_id, fit_config["comm_round"], sum(model.get_weights()[0])))
        log_message(self._logger, message, "DEBUG")
        # Send to the server the local model parameters (weights), number of examples used, and training metrics.
        return local_model_parameters, num_examples, training_metrics

    @staticmethod
    def _evaluate_relatively_to_local_labels(model: Model,
                                             x_test: NDArray,
                                             y_test: NDArray,
                                             evaluate_config: dict,
                                             evaluate_callbacks: list) -> dict:
        # Get the y_test labels (unique occurrences).
        y_test_local_labels = argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        y_test_local_labels_unique = unique(y_test_local_labels)
        # Convert the inputs to numpy arrays.
        x_test = array(x_test)
        y_test_local_labels = array(y_test_local_labels)
        # Predict full logits.
        full_logits_prediction = model.predict(x=x_test,
                                               batch_size=evaluate_config["batch_size"],
                                               steps=evaluate_config["steps"],
                                               verbose=evaluate_config["verbose"],
                                               callbacks=evaluate_callbacks)
        # Check if client has all labels.
        num_model_classes = model.output_shape[-1]
        has_all_labels = len(y_test_local_labels_unique) == num_model_classes
        if has_all_labels:
            # Directly evaluate using global labels and logits.
            loss_function = model.loss
            loss = loss_function(y_test_local_labels, full_logits_prediction).numpy()
            accuracy = mean(argmax(full_logits_prediction, axis=1) == y_test_local_labels)
        else:
            # Create full logits mask: suppress logits for irrelevant classes.
            y_test_labels_indices = sorted(y_test_local_labels_unique)
            irrelevant_class_indices = list(set(range(num_model_classes)) - set(y_test_labels_indices))
            local_logits_prediction = full_logits_prediction.copy()
            local_logits_prediction[:, irrelevant_class_indices] = -1e9  # Effectively mask irrelevant logits
            # Compute loss.
            loss_function = model.loss
            loss = loss_function(y_test_local_labels, local_logits_prediction).numpy()
            # Compute accuracy.
            local_labels_prediction = argmax(local_logits_prediction, axis=1)
            accuracy = mean(local_labels_prediction == y_test_local_labels)
        # Set the dictionary containing the evaluation history.
        history = {"loss": loss,
                   "sparse_categorical_accuracy": accuracy}
        # Return the dictionary containing the evaluation history.
        return history

    def _testing_task(self,
                      global_parameters: NDArrays,
                      x_test: NDArray,
                      y_test: NDArray,
                      evaluate_config: dict,
                      evaluate_callbacks: list,
                      evaluate_queue: Queue) -> None:
        # Start the timer.
        time_start = datetime.now()
        # Load the local model from the file.
        model_file = self.get_attribute("_model_file")
        model = load_model_from_file(model_file)
        # Update the parameters (weights) of the local model with those received from the server (global parameters).
        model.set_weights(global_parameters)
        # Initialize the dictionary containing the evaluation history.
        history = None
        # Test the local model using the local testing dataset slice.
        testing_approach = "all_labels"  # TODO: add to config file.
        match testing_approach:
            case "all_labels":
                history = model.evaluate(x=x_test,
                                         y=y_test,
                                         batch_size=evaluate_config["batch_size"],
                                         steps=evaluate_config["steps"],
                                         verbose=evaluate_config["verbose"],
                                         callbacks=evaluate_callbacks)
            case "local_labels":
                history = self._evaluate_relatively_to_local_labels(model,
                                                                    x_test,
                                                                    y_test,
                                                                    evaluate_config,
                                                                    evaluate_callbacks)
        # Dump the local model (with updated parameters from the aggregation) to file.
        model_file = self.get_attribute("_model_file")
        save_model_to_file(model, model_file)
        # Stop the timer.
        actual_computation_time_in_seconds = (datetime.now() - time_start).total_seconds()
        # Put the model testing result into the evaluate_queue.
        model_testing_result = {"history": history,
                                "actual_computation_time_in_seconds": actual_computation_time_in_seconds}
        evaluate_queue.put({"model_testing_result": model_testing_result})

    def _test_and_profile(self,
                          perf_log_file: Path,
                          model_metrics_file: Path,
                          comp_metrics_file: Path,
                          comm_round: int,
                          phase: str,
                          global_parameters: NDArrays,
                          x_test: NDArray,
                          y_test: NDArray,
                          evaluate_config: dict,
                          evaluate_callbacks: list) -> float:
        # Set the starting method of daemon processes.
        start_method = "forkserver"
        set_start_method(start_method, force=True)
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        # Get the device's memory bandwidth peak (in bytes per seconds).
        device_peak_memory_bandwidth_in_Bps = device_emulation_settings["peak_memory_bandwidth_in_bytes_per_second"]
        # Initializations.
        actual_computation_time_in_seconds = 0
        total_float_ops = 0
        x_size = len(x_test)
        batch_size = evaluate_config["batch_size"]
        client_model_metrics = {}
        if "use_replay_values" in evaluate_config:
            client_model_metrics = evaluate_config["client_model_metrics_replay"]
            fpo_test_m = evaluate_config["fpo_test_m"]
            mt_m_node = evaluate_config["mt_m_node"]
            mem_bw_node = evaluate_config["mem_bw_node"]
            cpu_gflops_node = evaluate_config["cpu_gflops_node"]
        else:
            # Set the perf settings.
            perf_events = ["cache-misses", "cache-references", "instructions", "cycles", "LLC-load-misses",
                           "LLC-store-misses"]
            perf_events_to_extract = ["cache-misses", "cache-references", "instructions", "cycles", "LLC-load-misses",
                                      "LLC-store-misses", "seconds time elapsed"]
            perf_testing_process = None
            # Initialize the model testing queue (evaluate_queue).
            evaluate_queue = Queue()
            # Launch the testing process.
            testing_process_target = self._testing_task
            testing_process_args = (global_parameters,
                                    x_test,
                                    y_test,
                                    evaluate_config,
                                    evaluate_callbacks,
                                    evaluate_queue)
            testing_process = Process(target=testing_process_target, args=testing_process_args)
            testing_process.start()
            # Check if perf is available.
            perf_is_available = check_if_perf_is_available()
            # Launch the perf process, if perf is available.
            if perf_is_available:
                perf_testing_process = launch_perf_process(testing_process.pid, perf_events, perf_log_file)
            # Wait for the testing process to finish.
            testing_process.join()
            # Stop the perf process, if launched.
            if perf_testing_process:
                stop_process(perf_testing_process)
            # Get the model testing result.
            evaluate_queue_element = evaluate_queue.get()
            model_testing_result = evaluate_queue_element["model_testing_result"]
            history = model_testing_result["history"]
            if isinstance(history, list):
                # Get a copy of the model's list of metrics names.
                metrics_names_copy = list(self.get_attribute("_metrics_names")).copy()
                # Add the loss metric name at index 0 (from index 1 onward the values are disposed by ordered metrics names).
                metrics_names_copy.insert(0, "loss")
                # Store the testing metrics.
                for index, metric_name in enumerate(metrics_names_copy):
                    client_model_metrics.update({metric_name: history[index]})
            elif isinstance(history, dict):
                client_model_metrics.update(history)
            actual_computation_time_in_seconds = model_testing_result["actual_computation_time_in_seconds"]
            # Load the local model from the file.
            model_file = self.get_attribute("_model_file")
            model = load_model_from_file(model_file)
            # Get the model settings.
            model_settings = self.get_attribute("_model_settings")
            # Get the input shape of the model.
            model_input_shape = get_model_input_shape(model)
            # Get the number of output classes of the model.
            model_num_output_classes = get_model_output_classes(model)
            # Generate a batch of dummy samples for testing (size of batch_size).
            x_test_dummy, _ = generate_dummy_sample_batch(batch_size,
                                                          model_input_shape,
                                                          model_num_output_classes,
                                                          model,
                                                          model_settings)
            # Build the concrete testing function that will be used by the profiler.
            profiling_module = ProfilingModule(model)
            silent_profiler_options = profiling_module.create_silent_profiler_options()
            test_step_fn = profiling_module.build_test_fn(x_test_dummy)
            concrete_test_fn = test_step_fn.get_concrete_function(x_test_dummy)
            # Profile the model using the concrete_test_fn function and the dummy samples.
            graph_info = profile(concrete_test_fn.graph, options=silent_profiler_options)
            # Collect the total float ops per batch of the profiling (dummy samples have size of batch_size).
            float_ops_per_batch = graph_info.total_float_ops
            # Get the float operations performance statistics of the profiling.
            flop_performance_statistics_testing = get_flop_performance_statistics(phase,
                                                                                  float_ops_per_batch,
                                                                                  x_size,
                                                                                  batch_size,
                                                                                  1)
            fpo_test_m = flop_performance_statistics_testing["float_ops_per_sample"]
            total_float_ops = flop_performance_statistics_testing["total_float_ops"]
            # Parse the perf log file, if exists.
            perf_events_metrics = {"memory_traffic_per_flop_in_bytes": 1}
            if perf_log_file.is_file():
                parsed_perf_log = parse_perf_log_file(perf_log_file, perf_events_to_extract)
                perf_events_metrics = summarize_perf_metrics(parsed_perf_log)
                if all(perf_event_metric_key in perf_events_metrics
                       for perf_event_metric_key in ["memory_traffic_in_bytes", "memory_bandwidth_usage_in_Bps"]):
                    # Measure how many bytes were moved per floating point operation during the testing.
                    total_memory_traffic_in_bytes = perf_events_metrics["memory_traffic_in_bytes"]
                    mt_m_node = get_memory_traffic_per_flop_in_bytes(total_memory_traffic_in_bytes, total_float_ops)
                    # Get the memory efficiency rate during the testing.
                    memory_bandwidth_usage_in_Bps = perf_events_metrics["memory_bandwidth_usage_in_Bps"]
                    memory_efficiency_rate = get_memory_efficiency_rate(device_peak_memory_bandwidth_in_Bps,
                                                                        memory_bandwidth_usage_in_Bps)
                    # Update the perf events metrics dictionary.
                    perf_events_metrics["memory_traffic_per_flop_in_bytes"] = mt_m_node
                    perf_events_metrics["memory_efficiency_rate"] = memory_efficiency_rate
            # Get the memory traffic per floating point operation during the testing (in bytes).
            mt_m_node = perf_events_metrics["memory_traffic_per_flop_in_bytes"]
            # Get the peak memory bandwidth and theoretical CPU GFLOPs of the executing node's hardware.
            node_info = get_cpu_and_memory_info()
            mem_bw_node = node_info["peak_memory_bandwidth_in_GBps"]
            cpu_gflops_node = node_info["theoretical_cpu_gflops"]
        # Get the device's CPU number of cores and frequency (in hertz).
        device_cpu_num_cores = device_emulation_settings["cpu_num_cores"]
        device_cpu_frequency_in_hertz = device_emulation_settings["cpu_frequency_in_hertz"]
        # Select randomly the device's float operations per cycle per CPU core performance during the testing.
        seed = self._deterministic_seed(client_id, comm_round)
        rng = default_rng(seed)
        fpocc_inference = device_emulation_settings["fpocc_inference"]
        device_cpu_float_ops_per_cycle_per_core = rng.choice(linspace(fpocc_inference[0], fpocc_inference[1], num=1000))
        # Estimate the device's memory traffic per floating point operation during the inference (in bytes).
        mem_bw_device = device_emulation_settings["peak_memory_bandwidth_in_gigabytes_per_second"]
        cpu_gflops_device = device_emulation_settings["peak_cpu_gflops"]
        device_memory_traffic_per_flop_in_bytes = estimate_mt_m_device(mt_m_node,
                                                                       mem_bw_node,
                                                                       cpu_gflops_node,
                                                                       mem_bw_device,
                                                                       cpu_gflops_device)
        # Add the number of examples and batch size to the client model metrics.
        client_model_metrics.update({"num_examples": x_size,
                                     "batch_size": batch_size})
        # Save the client model metrics to file.
        if not model_metrics_file.is_file():
            # Append the header line.
            model_header_data = "comm_round," + ",".join(list(client_model_metrics.keys())) + "\n"
            self._append_data_to_file(model_metrics_file, model_header_data)
        model_metrics_data = "{0},".format(comm_round) + ",".join([str(model_metric) for model_metric in list(client_model_metrics.values())]) + "\n"
        self._append_data_to_file(model_metrics_file, model_metrics_data)
        # Initialize the set of client computation metrics.
        client_comp_metrics = {"fpo_test_m": fpo_test_m,
                               "total_float_ops": total_float_ops,
                               "bs_test_i": batch_size,
                               "ds_test_i": x_size,
                               "nc_cpu_i": device_cpu_num_cores,
                               "fq_cpu_i": device_cpu_frequency_in_hertz,
                               "fpocc_m_i_test": device_cpu_float_ops_per_cycle_per_core,
                               "bw_mem_i": device_peak_memory_bandwidth_in_Bps,
                               "mt_m_test": device_memory_traffic_per_flop_in_bytes}
        # Save the client computation metrics to file.
        if not comp_metrics_file.is_file():
            # Append the header line.
            comp_header_data = "comm_round," + ",".join(list(client_comp_metrics.keys())) + "\n"
            self._append_data_to_file(comp_metrics_file, comp_header_data)
        comp_metrics_data = "{0},".format(comm_round) + ",".join([str(comp_metric) for comp_metric in list(client_comp_metrics.values())]) + "\n"
        self._append_data_to_file(comp_metrics_file, comp_metrics_data)
        # Return the actual computation time (in seconds).
        return actual_computation_time_in_seconds

    def evaluate(self,
                 global_parameters: NDArrays,
                 evaluate_config: dict) -> tuple[float, int, dict]:
        """ Implementation of the abstract method from the NumPyClient class."""
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        x_test = self.get_attribute("_x_test")
        y_test = self.get_attribute("_y_test")
        remaining_battery_energy_file = self.get_attribute("_remaining_battery_energy_file")
        logger = self.get_attribute("_logger")
        work_dir_testing_phase = self.get_attribute("_work_dir_testing_phase")
        testing_measurements_callback = self.get_attribute("_testing_measurements_callback")
        # Set the callbacks list.
        evaluate_callbacks = [testing_measurements_callback]
        # Set the phase.
        phase = "test"
        # Initialize the testing metrics dictionary.
        testing_metrics = {}
        # Get the current communication round.
        comm_round = evaluate_config["comm_round"]
        # Set the output files.
        down_metrics_file_name = "down_metrics.csv"
        down_metrics_file = Path(work_dir_testing_phase).joinpath(down_metrics_file_name).absolute()
        perf_log_file_name = "perf.log"
        perf_log_file = Path(work_dir_testing_phase).joinpath(perf_log_file_name).absolute()
        model_metrics_file_name = "model_metrics.csv"
        model_metrics_file = Path(work_dir_testing_phase).joinpath(model_metrics_file_name).absolute()
        comp_metrics_file_name = "comp_metrics.csv"
        comp_metrics_file = Path(work_dir_testing_phase).joinpath(comp_metrics_file_name).absolute()
        up_metrics_file_name = "up_metrics.csv"
        up_metrics_file = Path(work_dir_testing_phase).joinpath(up_metrics_file_name).absolute()
        # Create the parents directories of the output files (if not exist yet).
        down_metrics_file.parent.mkdir(exist_ok=True, parents=True)
        perf_log_file.parent.mkdir(exist_ok=True, parents=True)
        model_metrics_file.parent.mkdir(exist_ok=True, parents=True)
        comp_metrics_file.parent.mkdir(exist_ok=True, parents=True)
        up_metrics_file.parent.mkdir(exist_ok=True, parents=True)
        # Slice the testing dataset, according to server's request.
        rrds = self.get_attribute("_rrds")
        if "num_testing_examples_per_class_to_use" in evaluate_config.keys():
            num_testing_examples_per_class_to_use_str = evaluate_config["num_testing_examples_per_class_to_use"]
            num_testing_examples_per_class_to_use = {k: int(v) for k, v in (pair.split("=") for pair in num_testing_examples_per_class_to_use_str.split("|"))}
            x_test, y_test = rrds.slice_round_robin_per_class(num_testing_examples_per_class_to_use, phase)
        elif "num_testing_examples_to_use" in evaluate_config.keys():
            num_testing_examples_to_use = evaluate_config["num_testing_examples_to_use"]
            x_test, y_test = rrds.slice_round_robin_randomly(num_testing_examples_to_use, phase)
        # Replace 'None' values to None (necessary workaround on Flower).
        evaluate_config = {k: (None if v == "None" else v) for k, v in evaluate_config.items()}
        # Log the testing configuration (evaluate_config) received from the server.
        message = "[Client {0} | Round {1}] Received evaluate_config: {2}".format(client_id, comm_round, evaluate_config)
        log_message(logger, message, "DEBUG")
        # Get the time spent by the client selection event of round r.
        client_selection_time_in_seconds = self._get_client_selection_time_in_seconds(evaluate_config)
        # Estimate the energy consumed by this client during the client selection event of round r.
        client_selection_energy_in_joules = self._estimate_idle_energy_in_joules(client_selection_time_in_seconds)
        # Record the energy consumed by this client during the client selection event of round r.
        client_selection_event = "after_idle_during_the_client_selection"
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  client_selection_event,
                                                  client_selection_energy_in_joules)
        # Get the remaining battery energy in joules.
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules == 0:
            # Client has dropped due to lack of battery level.
            return 0, 0, {}
        # Estimate the time spent by this client during the download event of round r.
        download_time_in_seconds = self._estimate_download_time_in_seconds(down_metrics_file,
                                                                           comm_round,
                                                                           phase,
                                                                           global_parameters,
                                                                           evaluate_config)
        # Get the set of client download metrics from the corresponding file.
        client_down_metrics = self._read_client_metrics_from_file(down_metrics_file, comm_round)
        # Estimate the energy consumed by this client during the download event of round r.
        download_energy_in_joules = self._estimate_download_energy_in_joules(download_time_in_seconds)
        # Record the energy consumed by this client during the download event of round r.
        download_event = "after_downloading_global_parameters_and_{0}ing_instructions_from_the_server".format(phase)
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  download_event,
                                                  download_energy_in_joules)
        # Get the remaining battery energy in joules.
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules == 0:
            # Client has dropped due to lack of battery level.
            return 0, 0, {}
        # Log a 'testing my model' message.
        message = "[Client {0} | Round {1}] Testing my model...".format(client_id, comm_round)
        log_message(logger, message, "INFO")
        # Unset the logger.
        self._set_attribute("_logger", None)
        # Test and profile.
        actual_computation_time_in_seconds = self._test_and_profile(perf_log_file,
                                                                    model_metrics_file,
                                                                    comp_metrics_file,
                                                                    comm_round,
                                                                    phase,
                                                                    global_parameters,
                                                                    x_test,
                                                                    y_test,
                                                                    evaluate_config,
                                                                    evaluate_callbacks)
        # Get the set of client model metrics from the corresponding file.
        client_model_metrics = self._read_client_metrics_from_file(model_metrics_file, comm_round)
        # Get the set of client computation metrics from the corresponding file.
        client_comp_metrics = self._read_client_metrics_from_file(comp_metrics_file, comm_round)
        # Estimate the time spent by this client during the computation event of round r.
        computation_time_in_seconds = self._estimate_device_computation_time_in_seconds_via_scaling(perf_log_file,
                                                                                                    comp_metrics_file,
                                                                                                    comm_round,
                                                                                                    actual_computation_time_in_seconds)
        # Estimate the energy consumed by this client during the computation event of round r.
        computation_energy_in_joules = self._estimate_computation_energy_in_joules(computation_time_in_seconds)
        # Record the energy consumed by this client during the computation event of round r.
        computation_event = "after_local_testing"
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  computation_event,
                                                  computation_energy_in_joules)
        # Get the remaining battery energy in joules.
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules == 0:
            # Client has dropped due to lack of battery level.
            return 0, 0, {}
        # Append the client download metrics to the set of testing metrics.
        testing_metrics = testing_metrics | client_down_metrics
        # Append the client model metrics to the set of testing metrics.
        testing_metrics = testing_metrics | client_model_metrics
        # Append the client computation metrics to the set of testing metrics.
        testing_metrics = testing_metrics | client_comp_metrics
        # Get the loss value.
        loss = client_model_metrics["loss"]
        # Get the number of examples used.
        num_examples = client_model_metrics["num_examples"]
        # Get the parameters (weights) of the local model after the testing (dumped to file).
        model_file = self.get_attribute("_model_file")
        model = load_model_from_file(model_file)
        local_model_parameters = model.get_weights()
        # Estimate the time spent by this client during the upload event of round r.
        upload_time_in_seconds = self._estimate_upload_time_in_seconds(up_metrics_file,
                                                                       comm_round,
                                                                       phase,
                                                                       testing_metrics,
                                                                       local_model_parameters)
        # Get the set of client upload metrics from the corresponding file.
        client_up_metrics = self._read_client_metrics_from_file(up_metrics_file, comm_round)
        # Estimate the energy consumed by this client during the upload event of round r.
        upload_energy_in_joules = self._estimate_upload_energy_in_joules(upload_time_in_seconds)
        # Record the energy consumed by this client during the upload event of round r.
        upload_event = "after_uploading_local_parameters_and_{0}ing_metrics_to_the_server".format(phase)
        self._record_energy_consumed_during_event(comm_round,
                                                  phase,
                                                  upload_event,
                                                  upload_energy_in_joules)
        # Get the remaining battery energy in joules.
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules == 0:
            # Client has dropped due to lack of battery level.
            return 0, 0, {}
        # Append the client upload metrics to the set of testing metrics.
        testing_metrics = testing_metrics | client_up_metrics
        # Calculate the total time spent by this client during the testing phase event of round r.
        testing_time_in_seconds = download_time_in_seconds + computation_time_in_seconds + upload_time_in_seconds
        # Calculate the total energy consumed by this client during the testing phase event of round r.
        testing_energy_in_joules = download_energy_in_joules + computation_energy_in_joules + upload_energy_in_joules
        # Append the mean power consumptions to the testing metrics.
        mpc_idle_i = device_emulation_settings["mean_power_consumption_idle_in_watts"]
        mpc_recv_i = device_emulation_settings["mean_power_consumption_data_reception_in_watts"]
        mpc_comp_i = device_emulation_settings["mean_power_consumption_heavy_computational_load_in_watts"]
        mpc_send_i = device_emulation_settings["mean_power_consumption_data_transmission_in_watts"]
        testing_metrics.update({"mpc_idle_i": mpc_idle_i,
                                "mpc_recv_i": mpc_recv_i,
                                "mpc_comp_i": mpc_comp_i,
                                "mpc_send_i": mpc_send_i})
        # Append the times and energy consumptions to the testing metrics.
        testing_metrics.update({"download_time_in_seconds": download_time_in_seconds,
                                 "download_energy_in_joules": download_energy_in_joules,
                                 "computation_time_in_seconds": computation_time_in_seconds,
                                 "computation_energy_in_joules": computation_energy_in_joules,
                                 "upload_time_in_seconds": upload_time_in_seconds,
                                 "upload_energy_in_joules": upload_energy_in_joules,
                                 "testing_time_in_seconds": testing_time_in_seconds,
                                 "testing_energy_in_joules": testing_energy_in_joules})
        # Set the logger.
        self._set_attribute("_logger", logger)
        # Log a 'finished testing my model' message.
        message = ("[Client {0} | Round {1}] Finished testing my model... "
                   "Download costs -> time: {2} seconds | energy: {3} joules; "
                   "Computation costs -> time: {4} seconds | energy: {5} joules; "
                   "Upload costs -> time: {6} seconds | energy: {7} joules; "
                   "Total costs -> time: {8} seconds | energy: {9} joules."
                  .format(client_id,
                          comm_round,
                          round(download_time_in_seconds, 2),
                          round(download_energy_in_joules, 2),
                          round(computation_time_in_seconds, 2),
                          round(computation_energy_in_joules, 2),
                          round(upload_time_in_seconds, 2),
                          round(upload_energy_in_joules, 2),
                          round(testing_time_in_seconds, 2),
                          round(testing_energy_in_joules, 2)))
        log_message(logger, message, "INFO")
        # Send to the server the loss, number of examples used, and testing metrics.
        return loss, num_examples, testing_metrics
