import sys
from os import devnull, environ

# Suppress TensorFlow C++ log messages (redirecting stderr to null).
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.stderr = open(devnull, "w")

# Ignore GPU.
environ["CUDA_VISIBLE_DEVICES"] = "-1"

from csv import reader as csv_reader
from datetime import datetime
from flwr.common import NDArrays
from keras import losses, Model, utils
from math import atan2, ceil, cos, inf, radians, sin, sqrt
from numpy import dtype as np_dtype, ndarray, unique
from os import kill, sched_setaffinity
from pandas import concat, read_csv, to_datetime
from pathlib import Path
from platform import system
from psutil import cpu_freq, NoSuchProcess, Process as psutil_Process, STATUS_ZOMBIE
from random import random
from re import escape, findall, search
from signal import SIGINT, SIGKILL, SIGTERM
from socket import AF_INET, AF_INET6, create_connection, gaierror, SOCK_DGRAM, socket, timeout
from subprocess import CalledProcessError, CompletedProcess, DEVNULL, PIPE, Popen, run
from tensorflow import int32, random as tf_random
from time import time, sleep
from typing import List


# Locations coordinates (latitude, longitude).
LOCATIONS_COORDINATES = {"Bordeaux": (44.8378, -0.5792),
                         "Grenoble": (45.1885, 5.7245),
                         "Lille": (50.6292, 3.0573),
                         "Louvain": (50.8798, 4.7005),
                         "Luxembourg": (49.6116, 6.1319),
                         "Lyon": (45.7640, 4.8357),
                         "Nancy": (48.6921, 6.1844),
                         "Nantes": (47.2184, -1.5536),
                         "Paris": (48.8566, 2.3522),
                         "Rennes": (48.1173, -1.6778),
                         "Sophia": (43.6144, 7.0720),
                         "Strasbourg": (48.5734, 7.7521),
                         "Toulouse": (43.6047, 1.4442)}


def get_node_ip_address(ipv6: bool = False) -> str:
    family = AF_INET6 if ipv6 else AF_INET
    ip_socket = None
    try:
        ip_socket = socket(family, SOCK_DGRAM)
        # For IPv6, use an IPv6 address; for IPv4, use IPv4.
        dest = ("2001:4860:4860::8888", 80) if ipv6 else ("8.8.8.8", 80)
        ip_socket.connect(dest)
        ip_address = ip_socket.getsockname()[0]
    except Exception:
        ip_address = "::1" if ipv6 else "127.0.0.1"
    finally:
        ip_socket.close()
    return ip_address


def set_cpu_cores_affinity(target_cpu_cores: list | None) -> None:
    if target_cpu_cores:
        os_name = system()
        match os_name:
            case "Windows":
                p = psutil_Process()
                p.cpu_affinity(target_cpu_cores)
            case "Linux":
                sched_setaffinity(0, set(target_cpu_cores))
            case "Darwin":
                sched_setaffinity(0, set(target_cpu_cores))


def run_command(command_args: list,
                use_taskset: bool = False,
                target_cpu_cores: list = None,
                verbose: bool = False,
                check: bool = True,
                capture_output: bool = False,
                text = False) -> CompletedProcess:
    if use_taskset:
        target_cpu_cores_str = ",".join(map(str, target_cpu_cores))
        taskset_args = ["taskset", "-c", target_cpu_cores_str]
        command_args = taskset_args + command_args
    if capture_output:
        completed_process = run(args=command_args,
                                check=check,
                                capture_output=True,
                                text=text)
    else:
        completed_process = run(args=command_args,
                                check=check,
                                stdout=None if verbose else DEVNULL,
                                stderr=None if verbose else DEVNULL,
                                text=text)
    return completed_process


def get_cpu_cores_available() -> list:
    cpu_cores_available = psutil_Process().cpu_affinity()
    return cpu_cores_available


def get_num_cpu_cores_available() -> int:
    num_cpu_cores_available = len(get_cpu_cores_available())
    return num_cpu_cores_available


def check_if_powerjoular_is_available() -> bool:
    command_args = ["sudo", "powerjoular", "-v"]
    try:
        run(command_args, stdout=DEVNULL, stderr=DEVNULL, check=True)
        return True
    except (FileNotFoundError, CalledProcessError, PermissionError):
        # FileNotFoundError: powerjoular is not installed or not in PATH.
        # CalledProcessError: command failed unexpectedly.
        # PermissionError: insufficient privileges to run powerjoular.
        return False


def check_if_powerjoular_supports_timestamps_in_milliseconds() -> bool:
    command_args = ["sudo", "powerjoular", "-h"]
    try:
        help_output = run(command_args, capture_output=True, text=True)
        return "-c" in help_output.stdout
    except (FileNotFoundError, CalledProcessError):
        # FileNotFoundError: powerjoular is not installed or not in PATH.
        # CalledProcessError: command failed unexpectedly.
        return False


def launch_powerjoular_process(output_file: Path,
                               process_id: int | None = None,
                               timeout: float = 5.0) -> Popen:
    powerjoular_process_args = ["sudo", "powerjoular", "-f", str(output_file)]
    if process_id is not None:
        powerjoular_process_args.extend(["-p", str(process_id)])
    powerjoular_supports_timestamps_in_milliseconds = check_if_powerjoular_supports_timestamps_in_milliseconds()
    if powerjoular_supports_timestamps_in_milliseconds:
        powerjoular_process_args.append("-c")
    powerjoular_process = launch_process(powerjoular_process_args, timeout)
    while True:
        if output_file.exists():
            with output_file.open("r") as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    break # At least one power consumption record was written to the output (excluding the header).
    sleep(1)
    return powerjoular_process


def parse_powerjoular_timestamp(timestamp_str: str) -> datetime:
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            date_part, time_part = timestamp_str.split(" ")
            h, m, s_ms = time_part.split(":")
            s, ms = s_ms.split(".")
            s = s.zfill(2)
            ts_fixed = "{0} {1}:{2}:{3}.{4}".format(date_part, h, m, s, ms)
            return datetime.strptime(ts_fixed, "%Y-%m-%d %H:%M:%S.%f")
        except Exception as e:
            raise ValueError("Failed to parse timestamp '{0}': {1}".format(timestamp_str, e))


def kill_powerjoular_process(output_file: Path,
                             powerjoular_process: Popen,
                             start_timestamp: datetime,
                             end_timestamp: datetime,
                             check_interval: float = 0.5) -> None:
    while True:
        with output_file.open("r") as f:
            reader = csv_reader(f)
            header = next(reader, None)
            timestamps = []
            for row in reader:
                if not row or len(row) < 1:
                    continue
                try:
                    ts = parse_powerjoular_timestamp(row[0])
                    timestamps.append(ts)
                except Exception:
                    pass
            if timestamps:
                first_ts = timestamps[0]
                last_ts = timestamps[-1]
                if first_ts <= start_timestamp and last_ts >= end_timestamp:
                    kill_process(powerjoular_process)
                    break
        sleep(check_interval)


def measure_power_consumption_for_idle_task(powerjoular_idle_log_file: Path,
                                            duration_in_seconds: int,
                                            target_cpu_cores: list,
                                            process_id: int | None = None) -> tuple:
    powerjoular_process = launch_powerjoular_process(powerjoular_idle_log_file, process_id)
    idle_command_args = ["sleep", str(duration_in_seconds)]
    start_time = datetime.now()
    run_command(command_args=idle_command_args, use_taskset=True, target_cpu_cores=target_cpu_cores)
    end_time = datetime.now()
    kill_powerjoular_process(powerjoular_idle_log_file, powerjoular_process, start_time, end_time)
    return start_time, end_time


def measure_power_consumption_for_computation_task(powerjoular_computation_log_file: Path,
                                                   duration_in_seconds: int,
                                                   target_cpu_cores: list,
                                                   process_id: int | None = None) -> tuple:
    powerjoular_process = launch_powerjoular_process(powerjoular_computation_log_file, process_id)
    computation_command_args = ["stress-ng",
                                "--cpu", str(len(target_cpu_cores)),
                                "--cpu-method", "fft",
                                "--timeout", str(duration_in_seconds)]
    start_time = datetime.now()
    run_command(command_args=computation_command_args, use_taskset=True, target_cpu_cores=target_cpu_cores)
    end_time = datetime.now()
    kill_powerjoular_process(powerjoular_computation_log_file, powerjoular_process, start_time, end_time)
    return start_time, end_time


def measure_power_consumption_for_transmission_task(powerjoular_transmission_log_file: Path,
                                                    duration_in_seconds: int,
                                                    transmission_bandwidth: int,
                                                    transmission_bandwidth_unit: str,
                                                    target_ip: str,
                                                    target_cpu_cores: list,
                                                    process_id: int | None = None) -> tuple:
    powerjoular_process = launch_powerjoular_process(powerjoular_transmission_log_file, process_id)
    transmission_command_args = ["iperf3",
                                 "-c", target_ip,
                                 "-t", str(duration_in_seconds),
                                 "-b", str(transmission_bandwidth) + transmission_bandwidth_unit.replace("bps", "")]
    start_time = datetime.now()
    run_command(command_args=transmission_command_args, use_taskset=True, target_cpu_cores=target_cpu_cores)
    end_time = datetime.now()
    kill_powerjoular_process(powerjoular_transmission_log_file, powerjoular_process, start_time, end_time)
    return start_time, end_time


def measure_power_consumption_for_reception_task(powerjoular_reception_log_file: Path,
                                                 duration_in_seconds: int,
                                                 reception_bandwidth: int,
                                                 reception_bandwidth_unit: str,
                                                 target_ip: str,
                                                 target_cpu_cores: list,
                                                 process_id: int | None = None) -> tuple:
    powerjoular_process = launch_powerjoular_process(powerjoular_reception_log_file, process_id)
    reception_command_args = ["iperf3",
                              "-c", target_ip,
                              "--reverse",
                              "-t", str(duration_in_seconds),
                              "-b", str(reception_bandwidth) + reception_bandwidth_unit.replace("bps", "")]
    start_time = datetime.now()
    run_command(command_args=reception_command_args, use_taskset=True, target_cpu_cores=target_cpu_cores)
    end_time = datetime.now()
    kill_powerjoular_process(powerjoular_reception_log_file, powerjoular_process, start_time, end_time)
    return start_time, end_time


def get_cpu_governors(target_cpu_cores: list) -> dict:
    cpu_governors = {}
    for target_core in target_cpu_cores:
        try:
            scaling_governor_file = "/sys/devices/system/cpu/cpu{0}/cpufreq/scaling_governor".format(target_core)
            with open(file=scaling_governor_file, mode="r") as sgf:
                cpu_governors[target_core] = sgf.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            # FileNotFoundError: governor interface is not available on this core.
            # PermissionError: lack of sysfs permissions.
            # OSError: generic OS error while accessing the file.
            continue
    return cpu_governors


def set_cpu_governors(target_cpu_cores: list,
                      cpu_governors: list | str,
                      verbose: bool = False) -> None:
    for target_core in target_cpu_cores:
        try:
            cpu_governor = cpu_governors[target_core] if isinstance(cpu_governors, list) else cpu_governors
            cpu_governor_set_args = ["sudo", "cpufreq-set", "-c", str(target_core), "-g", cpu_governor]
            if verbose:
                print("Setting CPU governor of CPU{0} to '{1}'".format(target_core, cpu_governor))
            run_command(command_args=cpu_governor_set_args, verbose=verbose)
        except CalledProcessError as e:
            if verbose:
                print("Failed to set CPU governor of CPU{0}: {1}".format(target_core, e))


def restore_original_cpu_cores_governors(target_cpu_cores: list,
                                         original_cpu_governors: dict,
                                         verbose: bool = False) -> None:
    for target_core in target_cpu_cores:
        try:
            original_cpu_governor = original_cpu_governors[target_core]
            cpu_governor_set_args = ["sudo", "cpufreq-set", "-c", str(target_core), "-g", original_cpu_governor]
            if verbose:
                print("Restoring CPU governor of CPU{0} to '{1}'".format(target_core, original_cpu_governor))
            run_command(command_args=cpu_governor_set_args, verbose=verbose)
        except CalledProcessError as e:
            if verbose:
                print("Failed to restore CPU governor of CPU{0}: {1}".format(target_core, e))


def set_global_cpu_governor(cpu_governor: str = "performance",
                            verbose: bool = False) -> None:
    global_cpu_governor_set_args = ["sudo", "cpupower", "frequency-set", "-g", cpu_governor]
    try:
        if verbose:
            print("Setting global CPU governor to '{0}'".format(cpu_governor))
        run_command(command_args=global_cpu_governor_set_args, verbose=verbose)
    except CalledProcessError as e:
        if verbose:
            print("Failed to set global governor: {0}".format(e))


def convert_frequency_unit(freq: float,
                           from_unit: str,
                           to_unit: str) -> float:
    unit_to_hz = {"Hz": 1, "kHz": 1_000, "MHz": 1_000_000, "GHz": 1_000_000_000}
    if from_unit not in unit_to_hz:
        raise ValueError("Unsupported unit '{0}'. Supported units: {1}".format(from_unit, ", ".join(unit_to_hz)))
    if to_unit not in unit_to_hz:
        raise ValueError("Unsupported unit '{0}'. Supported units: {1}".format(to_unit, ", ".join(unit_to_hz)))
    if from_unit == to_unit:
        return freq
    # Convert the input to Hz.
    freq_in_hz = freq * unit_to_hz[from_unit]
    # Convert the input in Hz to the target unit.
    converted_freq = freq_in_hz / unit_to_hz[to_unit]
    return converted_freq


def get_cpu_cores_frequencies_via_psutil(target_cpu_cores: list = None,
                                         unit: str = "MHz") -> dict:
    cpu_cores_frequencies = {}
    for cpu_idx, cpu_frequency in enumerate(cpu_freq(percpu=True)):
        # Skip unwanted cores.
        if target_cpu_cores is not None and cpu_idx not in target_cpu_cores:
            continue
        # Skip offline/inaccessible cores.
        if cpu_frequency is None:
            continue
        cpu_frequency_min_in_MHz = cpu_frequency.min
        cpu_frequency_min_in_unit = convert_frequency_unit(cpu_frequency_min_in_MHz, "MHz", unit)
        cpu_frequency_max_in_MHz = cpu_frequency.max
        cpu_frequency_max_in_unit = convert_frequency_unit(cpu_frequency_max_in_MHz, "MHz", unit)
        cpu_cores_frequencies[cpu_idx] = {"freq_min_{0}".format(unit): cpu_frequency_min_in_unit,
                                          "freq_max_{0}".format(unit): cpu_frequency_max_in_unit}
    cpu_cores_frequencies = dict(sorted(cpu_cores_frequencies.items(), key=lambda item: item[0]))
    return cpu_cores_frequencies


def get_cpu_cores_frequencies_via_sysfs(target_cpu_cores: list = None,
                                        unit: str = "MHz") -> dict:
    cpu_cores_frequencies = {}
    for cpu_path in Path("/sys/devices/system/cpu/").glob("cpu[0-9]*"):
        try:
            cpu_idx = int(cpu_path.name.replace("cpu", ""))
            # Skip unwanted cores.
            if target_cpu_cores is not None and cpu_idx not in target_cpu_cores:
                continue
            cpufreq_path = cpu_path / "cpufreq"
            if not cpufreq_path.exists():
                continue
            freq_min_file = cpufreq_path / "scaling_min_freq"
            freq_max_file = cpufreq_path / "scaling_max_freq"
            if not freq_min_file.exists() or not freq_max_file.exists():
                continue
            cpu_frequency_min_in_kHz = int(freq_min_file.read_text().strip())
            cpu_frequency_min_in_unit = convert_frequency_unit(cpu_frequency_min_in_kHz, "kHz", unit)
            cpu_frequency_max_in_kHz = int(freq_max_file.read_text().strip())
            cpu_frequency_max_in_unit = convert_frequency_unit(cpu_frequency_max_in_kHz, "kHz", unit)
            cpu_cores_frequencies[cpu_idx] = {"freq_min_{0}".format(unit): cpu_frequency_min_in_unit,
                                              "freq_max_{0}".format(unit): cpu_frequency_max_in_unit}
        except (ValueError, OSError, FileNotFoundError):
            continue
    cpu_cores_frequencies = dict(sorted(cpu_cores_frequencies.items(), key=lambda item: item[0]))
    return cpu_cores_frequencies


def get_cpu_core_supported_frequencies(cpu_core: int,
                                       unit: str = "MHz",
                                       verbose: bool = False) -> tuple:
    try:
        cpu_min_freq_path = "/sys/devices/system/cpu/cpu{0}/cpufreq/cpuinfo_min_freq".format(cpu_core)
        cpu_max_freq_path = "/sys/devices/system/cpu/cpu{0}/cpufreq/cpuinfo_max_freq".format(cpu_core)
        with open(file=cpu_min_freq_path, mode="r") as min_file:
            supported_freq_min_in_kHz = int(min_file.read().strip())
            supported_freq_min_in_unit = convert_frequency_unit(supported_freq_min_in_kHz, "kHz", unit)
        with open(file=cpu_max_freq_path, mode="r") as max_file:
            supported_freq_max_in_kHz = int(max_file.read().strip())
            supported_freq_max_in_unit = convert_frequency_unit(supported_freq_max_in_kHz, "kHz", unit)
        return supported_freq_min_in_unit, supported_freq_max_in_unit
    except FileNotFoundError as e:
        # Fallback values: min: 800 MHz; max: 2000 MHz.
        fallback_freq_min_in_MHz = 800
        fallback_freq_min_in_unit = convert_frequency_unit(fallback_freq_min_in_MHz, "MHz", unit)
        fallback_freq_max_in_MHz = 2000
        fallback_freq_max_in_unit = convert_frequency_unit(fallback_freq_max_in_MHz, "MHz", unit)
        if verbose:
            print("Error reading the CPU frequency info for CPU{0}: {1}! Returning the fallback values...".format(cpu_core, e))
        return fallback_freq_min_in_unit, fallback_freq_max_in_unit


def set_cpu_cores_frequencies(target_cpu_cores: list,
                              target_freq_min: float,
                              target_freq_max: float,
                              target_freq_unit: str,
                              verbose: bool = False) -> None:
    for target_core in target_cpu_cores:
        # Get the supported frequencies for the target core (with fallback).
        supported_freq_min, supported_freq_max = get_cpu_core_supported_frequencies(target_core, target_freq_unit)
        # Calculate the new target frequencies by clamping to the supported ranges.
        target_freq_min_clamped = max(target_freq_min, supported_freq_min)
        target_freq_max_clamped = min(target_freq_max, supported_freq_max)
        cpu_frequency_set_args = ["sudo",
                                  "cpufreq-set",
                                  "-c", str(target_core),
                                  "-d", str(target_freq_min_clamped) + target_freq_unit,
                                  "-u", str(target_freq_max_clamped) + target_freq_unit]
        try:
            if verbose:
                print("Setting CPU{0} min. frequency to {1}{2} and max. frequency to {3}{2}"
                      .format(target_core, target_freq_min_clamped, target_freq_unit, target_freq_max_clamped))
            run_command(command_args=cpu_frequency_set_args, verbose=verbose)
        except CalledProcessError as e:
            if verbose:
                print("Failed to set CPU min/max frequency on CPU{0}: {1}".format(target_core, e))


def restore_original_cpu_cores_frequencies(target_cpu_cores: list,
                                           original_cpu_cores_frequencies: dict,
                                           unit: str = "MHz",
                                           verbose: bool = False) -> None:
    for target_core in target_cpu_cores:
        original_freq_min_in_unit = original_cpu_cores_frequencies[target_core]["freq_min_{0}".format(unit)]
        original_freq_max_in_unit = original_cpu_cores_frequencies[target_core]["freq_max_{0}".format(unit)]
        cpu_frequency_set_args = ["sudo",
                                  "cpufreq-set",
                                  "-c", str(target_core),
                                  "-d", str(original_freq_min_in_unit) + unit,
                                  "-u", str(original_freq_max_in_unit) + unit]
        try:
            if verbose:
                print("Restoring CPU{0} to its original minimum and maximum frequencies: {1} (min) / {2} (max)"
                      .format(target_core, original_freq_min_in_unit, original_freq_max_in_unit))
            run_command(command_args=cpu_frequency_set_args, verbose=verbose)
        except CalledProcessError as e:
            if verbose:
                print("Failed to restore the original frequencies for CPU{0}: {1}".format(target_core, e))


def get_peak_memory_bandwidth_via_mlc(target_cpu_cores: list,
                                      mlc_binary_path: Path = None,
                                      verbose: bool = False) -> float:
    if mlc_binary_path is None:
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parents[2]
        mlc_binary_path = project_root / "mlc" / "mlc"
    mlc_binary_path = mlc_binary_path.resolve()
    if not mlc_binary_path.exists():
        raise FileNotFoundError("MLC binary not found at: {0}".format(mlc_binary_path))
    peak_memory_bandwidth = 0
    peak_memory_bandwidth_args = ["sudo", str(mlc_binary_path), "--peak_injection_bandwidth"]
    try:
        result = run_command(command_args=peak_memory_bandwidth_args,
                             use_taskset=True,
                             target_cpu_cores=target_cpu_cores,
                             check=True,
                             capture_output=True,
                             text=True)
        output = result.stdout
        all_reads_match = search(r"ALL Reads\s*:\s*([\d.]+)", output)
        if all_reads_match:
            peak_memory_bandwidth = float(all_reads_match.group(1))
        else:
            raise ValueError("Could not find 'ALL Reads' in the MLC output.")
    except CalledProcessError as e:
        if verbose:
            print("Error running MLC: {0}".format(e))
    return peak_memory_bandwidth


def convert_memory_bandwidth(memory_bandwidth: float,
                             from_unit: str,
                             to_unit: str) -> float:
    unit_multipliers = {"Bps": 1, "KBps": 1e3, "MBps": 1e6, "GBps": 1e9, "TBps": 1e12}
    if from_unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'!\nSupported units: {1}"
                         .format(from_unit, {", ".join(unit_multipliers.keys())}))
    if to_unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'!\nSupported units: {1}"
                         .format(to_unit, {", ".join(unit_multipliers.keys())}))
    if from_unit == to_unit:
        return memory_bandwidth
    # Convert the memory bandwidth to the base unit (Bps), then to the target unit.
    memory_bandwidth_in_Bps = memory_bandwidth * unit_multipliers[from_unit]
    converted_memory_bandwidth = memory_bandwidth_in_Bps / unit_multipliers[to_unit]
    return converted_memory_bandwidth


def launch_process(process_args: list,
                   timeout: float = 5.0) -> Popen:
    process = Popen(args=process_args, stdout=DEVNULL, stderr=DEVNULL)
    # Wait until the process is confirmed to be running.
    try:
        ps_proc = psutil_Process(process.pid)
        start_time = time()
        while True:
            if ps_proc.is_running() and ps_proc.status() != STATUS_ZOMBIE:
                break
            if time() - start_time > timeout:
                raise RuntimeError("The process did not start running within the timeout.")
            sleep(0.05)
    except NoSuchProcess:
        raise RuntimeError("The process process exited unexpectedly.")
    return process


def stop_process(process: Popen,
                 timeout: float = 5.0) -> None:
    try:
        # Send SIGINT to terminate gracefully.
        process.send_signal(SIGINT)
        # Wait for a graceful exit.
        start_time = time()
        while process.poll() is None:
            elapsed_time = time() - start_time
            if elapsed_time > timeout:
                print("Process did not terminate after {0} seconds, escalating to SIGTERM.".format(timeout))
                process.send_signal(SIGTERM)
                break
            sleep(0.1)
        # Final wait for process to exit (without hanging).
        process.wait(timeout=timeout)
    except Exception as e:
        print("Failed to stop process gracefully: {0}".format(e))


def kill_process(process: Popen) -> None:
    kill(process.pid, SIGKILL)


def get_mean_power_consumption_in_watts(powerjoular_log_file: Path,
                                        start_timestamp: datetime,
                                        end_timestamp: datetime,
                                        power_prefix: str = "CPU") -> float:
    powerjoular_log_df = read_csv(powerjoular_log_file)
    powerjoular_log_df["Date"] = to_datetime(powerjoular_log_df["Date"])
    filtered_df = powerjoular_log_df[(powerjoular_log_df["Date"] >= start_timestamp) &
                                     (powerjoular_log_df["Date"] <= end_timestamp)]
    immediate_previous_record = powerjoular_log_df[powerjoular_log_df["Date"] < start_timestamp]
    immediate_next_record = powerjoular_log_df[powerjoular_log_df["Date"] > end_timestamp]
    if not immediate_previous_record.empty:
        filtered_df = concat([immediate_previous_record.iloc[[-1]], filtered_df])
    if not immediate_next_record.empty:
        filtered_df = concat([filtered_df, immediate_next_record.iloc[[0]]])
    if len(filtered_df) < 2:
        raise ValueError("Not enough data points to compute weighted mean.")
    filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)
    time_deltas = (filtered_df["Date"].shift(-1) - filtered_df["Date"]).dt.total_seconds()
    power_values = filtered_df["{0} Power".format(power_prefix)]
    time_deltas = time_deltas.iloc[:-1]
    power_values = power_values.iloc[:-1]
    weighted_power = (power_values * time_deltas).sum()
    total_duration = (end_timestamp - start_timestamp).total_seconds()
    if total_duration <= 0:
        raise ValueError("End timestamp must be greater than start timestamp.")
    mean_power_consumption_in_watts = weighted_power / total_duration
    return mean_power_consumption_in_watts


def get_energy_consumption_in_joules(powerjoular_log_file: Path,
                                     start_timestamp: datetime,
                                     end_timestamp: datetime,
                                     power_prefix: str = "CPU") -> float:
    powerjoular_log_df = read_csv(powerjoular_log_file)
    powerjoular_log_df["Date"] = to_datetime(powerjoular_log_df["Date"])
    filtered_df = powerjoular_log_df[(powerjoular_log_df["Date"] >= start_timestamp) &
                                     (powerjoular_log_df["Date"] <= end_timestamp)]
    if filtered_df.empty:
        immediate_previous_record = None
        if not powerjoular_log_df[powerjoular_log_df["Date"] < start_timestamp].empty:
            immediate_previous_record = powerjoular_log_df[powerjoular_log_df["Date"] < start_timestamp].iloc[-1:]
        immediate_next_record = None
        if not powerjoular_log_df[powerjoular_log_df["Date"] > end_timestamp].empty:
            immediate_next_record = powerjoular_log_df[powerjoular_log_df["Date"] > end_timestamp].iloc[:1]
        if immediate_previous_record is not None:
            filtered_df = concat([immediate_previous_record, filtered_df], ignore_index=True)
        if immediate_next_record is not None:
            filtered_df = concat([filtered_df, immediate_next_record], ignore_index=True)
    if filtered_df.empty:
        raise ValueError("No records found within the provided timestamp range.")
    actual_time_duration = (end_timestamp - start_timestamp).total_seconds()
    first_record_timestamp = filtered_df["Date"].iloc[0]
    last_record_timestamp = filtered_df["Date"].iloc[-1]
    time_span_of_records = (last_record_timestamp - first_record_timestamp).total_seconds()
    if time_span_of_records <= 0:
        return 0
    sum_of_power = filtered_df["{0} Power".format(power_prefix)].sum()
    energy_consumption_in_joules = sum_of_power * (actual_time_duration / time_span_of_records)
    return energy_consumption_in_joules


def scale_measure(measure: float,
                  num_cpu_cores_available: int,
                  num_cpu_cores_used: int) -> float:
    scaled_measure = measure * (num_cpu_cores_used / num_cpu_cores_available)
    return scaled_measure


def get_cpu_peak_frequency(unit: str = "Hz") -> float:
    unit_divisors = {"Hz": 1, "MHz": 1e6, "GHz": 1e9}
    if unit not in unit_divisors:
        raise ValueError("Unsupported unit '{0}'. Supported units: {1}".format(unit, ', '.join(unit_divisors.keys())))
    # Get the peak frequency for all CPU cores using pathlib.
    cpu_peak_frequencies_in_hertz = []
    cpu_dir = Path("/sys/devices/system/cpu/")
    for cpu in cpu_dir.iterdir():
        if cpu.name.startswith("cpu") and (cpu / "cpufreq").is_dir():
            max_freq_file = cpu / "cpufreq" / "cpuinfo_max_freq"
            try:
                with max_freq_file.open("r") as f:
                    max_freq_khz = int(f.read().strip())
                    max_freq_hz = max_freq_khz * 1000
                    cpu_peak_frequencies_in_hertz.append(max_freq_hz)
            except (FileNotFoundError, PermissionError, OSError):
                # CPU might be offline, isolated, or resource busy...
                continue
    if not cpu_peak_frequencies_in_hertz:
        raise RuntimeError("Could not retrieve peak CPU frequencies.")
    # Get the highest frequency among all cores (max of the peak frequencies).
    cpu_peak_frequency_in_hertz = max(cpu_peak_frequencies_in_hertz)
    # Convert from Hz to the requested unit.
    divisor = unit_divisors[unit]
    cpu_peak_frequency = cpu_peak_frequency_in_hertz / divisor
    # Return the CPU's peak frequency.
    return cpu_peak_frequency


def get_cpu_instructions_set() -> list:
    # Get the instructions set of CPU using the 'lscpu' command.
    lscpu_args = ["lscpu"]
    lscpu_out = run_command(command_args=lscpu_args, check=True, capture_output=True, text=True).stdout
    cpu_instructions_set = []
    for line in lscpu_out.splitlines():
        if "Flags" in line:
            cpu_instructions_set = line.strip().split(":")[1].split()
    return cpu_instructions_set


def estimate_float_ops_per_cycle() -> int:
    cpu_instructions_set = [cpu_instruction.lower() for cpu_instruction in get_cpu_instructions_set()]
    if "avx-512" in cpu_instructions_set:
        return 4  # Typically 4 FLOPs per cycle for AVX-512.
    elif "avx2" in cpu_instructions_set:
        return 2  # Typically 2 FLOPs per cycle for AVX2.
    else:
        return 1  # Default FLOPs per cycle for basic integer ops.


def estimate_theoretical_cpu_float_ops_per_second() -> float:
    num_cpu_cores = get_num_cpu_cores_available()
    clock_speed_in_hz = get_cpu_peak_frequency(unit="Hz")
    float_ops_per_cycle = estimate_float_ops_per_cycle()
    theoretical_cpu_float_ops_per_second = num_cpu_cores * clock_speed_in_hz * float_ops_per_cycle
    return theoretical_cpu_float_ops_per_second


def get_cpu_current_frequency_via_psutil(unit: str = "Hz") -> float:
        unit_multipliers = {"Hz": 1000000, "MHz": 1000, "GHz": 1}
        if unit not in unit_multipliers:
            raise ValueError("Unsupported unit '{0}'. Supported units: {1}".format(unit, ", ".join(unit_multipliers.keys())))
        cpu_frequency = cpu_freq(percpu=False)
        if cpu_frequency is None:
            raise RuntimeError("Could not retrieve CPU frequency.")
        multiplier = 1 / unit_multipliers[unit]
        cpu_frequency = cpu_frequency.current * multiplier
        return cpu_frequency


def get_cpu_current_frequency_via_sysfs(unit: str = "Hz") -> float:
    unit_multipliers = {"Hz": 1000000, "MHz": 1000, "GHz": 1}
    if unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'. Supported units: {1}".format(unit, ", ".join(unit_multipliers.keys())))
    scaling_cur_freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if not scaling_cur_freq_path.exists():
        raise FileNotFoundError("Could not find scaling_cur_freq at expected path.")
    try:
        cpu_frequency_in_kHz = int(scaling_cur_freq_path.read_text().strip())
        cpu_frequency = cpu_frequency_in_kHz / unit_multipliers[unit]
        return cpu_frequency
    except Exception as e:
        raise RuntimeError("Failed to read or convert CPU frequency: {0}".format(e))


def get_cpu_model() -> str:
    # Get CPU model using the 'lscpu' command.
    lscpu_args = ["lscpu"]
    lscpu_out = run_command(command_args=lscpu_args, check=True, capture_output=True, text=True).stdout
    match = search(r"Model name:\s+(.*)", lscpu_out)
    if match:
        return match.group(1).strip()
    # Otherwise, get it from '/proc/cpuinfo' (fallback).
    cpuinfo_args = ["cat", "/proc/cpuinfo"]
    cpuinfo_out = run_command(command_args=cpuinfo_args, check=True, capture_output=True, text=True).stdout
    match = search(r"model name\s+:\s+(.*)", cpuinfo_out)
    if match:
        return match.group(1).strip()
    return "unknown"


def get_cpu_num_sockets() -> int:
    # Get the number of CPU sockets using the 'lscpu' command.
    lscpu_args = ["lscpu"]
    lscpu_out = run_command(command_args=lscpu_args, check=True, capture_output=True, text=True).stdout
    match = search(r"Socket\(s\):\s+(\d+)", lscpu_out)
    num_cpu_sockets = int(match.group(1)) if match else 1  # Fallback: 1 socket.
    return num_cpu_sockets


def get_num_memory_channels_from_known_cpus_list(cpu_model: str) -> int:
    # Get the number of memory channels from a list of known CPUs.
    default_num_memory_channels = 4  # Fallback: 4 memory channels.
    known_num_memory_channels = {"Intel(R) Xeon(R) CPU E5-2630 v3": 4,
                                 "Intel(R) Xeon(R) Gold 6130": 6,
                                 "AMD EPYC 7302P": 8,
                                 "AMD EPYC 7742": 8,
                                 "Intel(R) Xeon(R) Platinum 8260": 6,
                                 "Intel(R) Xeon(R) Gold 5218": 6}
    if cpu_model in known_num_memory_channels.keys():
        return known_num_memory_channels[cpu_model]
    return default_num_memory_channels


def get_memory_speed_in_mtps() -> int:
    # Get the memory speed in MT/s using the 'dmidecode' command.
    dmidecode_args = ["sudo", "dmidecode", "-t", "memory"]
    dmidecode_out = run_command(command_args=dmidecode_args, check=True, capture_output=True, text=True).stdout
    matches = findall(r"Configured Memory Speed:\s+(\d+)\s+MT/s", dmidecode_out)
    speeds = list(map(int, matches))
    default_memory_speed = 2133  # Fallback: 2133 (most common for DDR4).
    memory_speed_in_MTps = max(speeds) if speeds else default_memory_speed
    return memory_speed_in_MTps


def get_cpu_and_memory_info() -> dict:
    # Get CPU model using the 'lscpu' command.
    cpu_model = get_cpu_model()
    # Get the number of CPU sockets using the 'lscpu' command.
    num_cpu_sockets = get_cpu_num_sockets()
    # Get the number of memory channels from a list of known CPUs.
    num_memory_channels_per_cpu = get_num_memory_channels_from_known_cpus_list(cpu_model)
    # Get the memory speed in MT/s using the 'dmidecode' command.
    memory_speed_in_MTps = get_memory_speed_in_mtps()
    # Estimate the peak memory bandwidth (theoretical).
    memory_bus_width_in_bits = 64  # Default DDR width.
    total_channels = num_cpu_sockets * num_memory_channels_per_cpu
    peak_memory_bandwidth_in_Bps = total_channels * memory_speed_in_MTps * 1e6 * (memory_bus_width_in_bits / 8)
    peak_memory_bandwidth_in_GBps = convert_memory_bandwidth(peak_memory_bandwidth_in_Bps, "Bps", "GBps")
    # Get the peak CPU frequency in hertz.
    peak_cpu_frequency_in_Hz = get_cpu_peak_frequency("Hz")
    peak_cpu_frequency_in_GHz = convert_frequency_unit(peak_cpu_frequency_in_Hz, "Hz", "GHz")
    # Get the theoretical CPU's floating-point operations per second (and GFLOPs).
    theoretical_cpu_float_ops_per_second = estimate_theoretical_cpu_float_ops_per_second()
    theoretical_cpu_gflops = theoretical_cpu_float_ops_per_second / 1e9
    # Set the dictionary of CPU and memory info.
    cpu_and_memory_info = {"cpu_model": cpu_model,
                           "num_cpu_sockets": num_cpu_sockets,
                           "num_memory_channels_per_cpu": num_memory_channels_per_cpu,
                           "memory_bus_width_in_bits": memory_bus_width_in_bits,
                           "memory_speed_in_MTps": memory_speed_in_MTps,
                           "peak_memory_bandwidth_in_Bps": peak_memory_bandwidth_in_Bps,
                           "peak_memory_bandwidth_in_GBps": peak_memory_bandwidth_in_GBps,
                           "peak_cpu_frequency_in_Hz": peak_cpu_frequency_in_Hz,
                           "peak_cpu_frequency_in_GHz": peak_cpu_frequency_in_GHz,
                           "theoretical_cpu_float_ops_per_second": theoretical_cpu_float_ops_per_second,
                           "theoretical_cpu_gflops": theoretical_cpu_gflops}
    # Return the dictionary of CPU and memory info.
    return cpu_and_memory_info


def get_known_networks_info(network_bandwidth_unit: str = "mbps") -> dict:
    # Set the base values of networks info (in megabits per second).
    networks_info_mbps = {"4G LTE": {"upload_bandwidth": 10, "download_bandwidth": 50},
                          "5G": {"upload_bandwidth": 100, "download_bandwidth": 500},
                          "Wi-Fi 5": {"upload_bandwidth": 100, "download_bandwidth": 200},
                          "Gigabit Ethernet": {"upload_bandwidth": 1000, "download_bandwidth": 1000}}
    # Normalize and validate the desired unit.
    network_bandwidth_unit = network_bandwidth_unit.lower()
    if network_bandwidth_unit in ["mbps", "megabits", "megabits per second"]:
        multiplier = 1
    elif network_bandwidth_unit in ["gbps", "gigabits", "gigabits per second"]:
        multiplier = 1 / 1000
    elif network_bandwidth_unit in ["bps", "bits", "bits per second"]:
        multiplier = 1000000
    else:
        raise ValueError("Unsupported unit: {0}".format(network_bandwidth_unit))
    # Convert and return the dictionary of networks info.
    networks_info = {}
    for network, speeds in networks_info_mbps.items():
        networks_info[network] = {"upload_bandwidth": speeds["upload_bandwidth"] * multiplier,
                                  "download_bandwidth": speeds["download_bandwidth"] * multiplier}
    return networks_info


def generate_dummy_sample_batch(batch_size: int,
                                input_shape: tuple,
                                num_output_classes: int) -> tuple:
    # Generate a batch of dummy samples (size of batch_size).
    x_dummy_batch = tf_random.normal((batch_size,) + input_shape)
    y_dummy_batch = utils.to_categorical(tf_random.uniform((batch_size,), minval=0, maxval=num_output_classes, dtype=int32),
                                         num_classes=num_output_classes)
    # Return the generated batch of dummy samples.
    return x_dummy_batch, y_dummy_batch


def get_flop_performance_statistics(phase: str,
                                    float_ops_per_batch: int,
                                    x_size: int,
                                    batch_size: int,
                                    epochs: int) -> dict:
    float_ops_per_sample = 0
    total_float_ops = 0
    match phase:
        case "train":
            float_ops_per_sample = float_ops_per_batch // batch_size
            steps_per_epoch = ceil(x_size / batch_size)
            total_float_ops = float_ops_per_batch * steps_per_epoch * epochs
        case "test":
            float_ops_per_sample = float_ops_per_batch // batch_size
            steps = ceil(x_size / batch_size)
            total_float_ops = float_ops_per_batch * steps
    model_flop_training_statistics = {"float_ops_per_sample": float_ops_per_sample,
                                      "float_ops_per_batch": float_ops_per_batch,
                                      "total_float_ops": total_float_ops}
    return model_flop_training_statistics


def get_actual_cpu_flop_per_cycle_per_core_performance(total_float_ops: int,
                                                       duration_in_seconds: float,
                                                       cpu_frequency_in_hertz: float,
                                                       cpu_num_cores: int) -> float:
    total_cpu_cycles = cpu_frequency_in_hertz * duration_in_seconds
    cpu_float_ops_per_cycle = total_float_ops / total_cpu_cycles
    cpu_float_ops_per_cycle_per_core = cpu_float_ops_per_cycle / cpu_num_cores
    return cpu_float_ops_per_cycle_per_core


def get_cpu_flop_per_cycle_per_core_inference_performance(total_inference_float_ops: int,
                                                          actual_test_time_in_seconds: float,
                                                          cpu_frequency_in_hertz: float,
                                                          num_cpu_cores: int) -> float:
    total_cpu_cycles_inference = cpu_frequency_in_hertz * actual_test_time_in_seconds
    cpu_float_ops_per_cycle_inference = total_inference_float_ops / total_cpu_cycles_inference
    cpu_float_ops_per_cycle_per_core_inference = cpu_float_ops_per_cycle_inference / num_cpu_cores
    return cpu_float_ops_per_cycle_per_core_inference


def get_cpu_flop_actual_performance(total_training_float_ops: int,
                                    total_inference_float_ops: int,
                                    actual_training_time_in_seconds: float,
                                    actual_test_time_in_seconds: float,
                                    cpu_frequency_in_hertz: float,
                                    num_cpu_cores: int) -> dict:
    # Get the CPU performance for the training.
    cpu_float_ops_per_cycle_per_core_training \
        = get_actual_cpu_flop_per_cycle_per_core_performance(total_training_float_ops,
                                                             actual_training_time_in_seconds,
                                                             cpu_frequency_in_hertz,
                                                             num_cpu_cores)
    # Get the CPU performance for the inference.
    cpu_float_ops_per_cycle_per_core_inference \
        = get_cpu_flop_per_cycle_per_core_inference_performance(total_inference_float_ops,
                                                                actual_test_time_in_seconds,
                                                                cpu_frequency_in_hertz,
                                                                num_cpu_cores)
    # Set the dictionary of the CPU's floating-point operations actual performance.
    cpu_flop_actual_performance = {"cpu_float_ops_per_cycle_per_core_training": cpu_float_ops_per_cycle_per_core_training,
                                   "cpu_float_ops_per_cycle_per_core_inference": cpu_float_ops_per_cycle_per_core_inference}
    # Return the dictionary of the CPU's floating-point operations actual performance.
    return cpu_flop_actual_performance


def check_if_perf_is_available() -> bool:
    command_args = ["perf", "stat", "--", "true"]
    try:
        run(command_args, stdout=DEVNULL, stderr=DEVNULL, check=True)
        return True
    except (FileNotFoundError, CalledProcessError, PermissionError):
        # FileNotFoundError: perf is not installed or not in PATH.
        # CalledProcessError: perf ran but exited with error (e.g., unsupported counters, kernel restrictions).
        # PermissionError: lack of permissions to run perf (common in WSL2 or unprivileged environments).
        return False


def launch_perf_process(process_id: int,
                        perf_events: list,
                        perf_log_file: Path) -> Popen:
    command_args = ["perf", "stat", "-e", ",".join(perf_events), "-p", str(process_id), "-o", str(perf_log_file)]
    perf_process = Popen(command_args, stdout=PIPE, stderr=PIPE)
    return perf_process


def extract_perf_value_from_text(perf_event: str,
                                 log_text: str) -> int:
    # First pattern: CPU-specific or generic event names with optional '/' suffix.
    first_perf_events_pattern = r"([\d,]+)\s+(?:cpu_atom/|cpu_core/|)?{0}/".format(escape(perf_event))
    match = search(first_perf_events_pattern, log_text)
    if match:
        # Remove commas and convert the value to an integer.
        return int(match.group(1).replace(",", ""))
    # Second pattern: General event name with word boundary.
    second_perf_events_pattern = r"([\d,]+)\s+{0}\b".format(escape(perf_event))
    match = search(second_perf_events_pattern, log_text)
    if match:
        # Remove commas and convert the value to an integer.
        return int(match.group(1).replace(",", ""))
    # If no match found...
    return 0


def parse_perf_log_file(perf_log_file: Path,
                        perf_events_to_extract: list) -> dict:
    parsed_perf_log = {}
    # Read the log file content.
    with open(file=perf_log_file, mode="r") as f:
        perf_log_file_text = f.read()
    # Extract each performance event from the log text.
    for perf_event in perf_events_to_extract:
        perf_event_value = extract_perf_value_from_text(perf_event, perf_log_file_text)
        parsed_perf_log[perf_event] = perf_event_value
    # Extract elapsed time in seconds (special case).
    time_pattern = r"([\d.]+) seconds time elapsed"
    time_match = search(time_pattern, perf_log_file_text)
    if time_match:
        parsed_perf_log["seconds time elapsed"] = float(time_match.group(1))
    return parsed_perf_log


def get_cache_line_size_in_bytes() -> int | None:
    cache_line_size_in_bytes = 64  # Fallback: on most modern x86 systems (Intel/AMD), the cache line size is 64 bytes.
    try:
        coherency_line_size_path = Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
        if coherency_line_size_path.exists():
            with open(file=coherency_line_size_path, mode="r") as f:
                cache_line_size_in_bytes = int(f.read().strip())
    except FileNotFoundError:
        pass
    return cache_line_size_in_bytes


def summarize_perf_metrics(parsed_perf_log: dict) -> dict:
    # Get the size of a cache line (in bytes).
    cache_line_size_in_bytes = get_cache_line_size_in_bytes()
    # Define possible key patterns to adapt to different systems
    possible_keys = {"cache-misses": ["cache-misses", "cpu_atom/cache-misses/", "cpu_core/cache-misses/"],
                     "cache-references": ["cache-references", "cpu_atom/cache-references/", "cpu_core/cache-references/"],
                     "instructions": ["instructions", "cpu_atom/instructions/", "cpu_core/instructions/"],
                     "cycles": ["cycles", "cpu_atom/cycles/", "cpu_core/cycles/"],
                     "seconds_time_elapsed": ["seconds time elapsed"]}
    # Helper to find the first available key in the parsed log.
    def get_value(keys):
        for key in keys:
            if key in parsed_perf_log:
                return parsed_perf_log[key]
        return None
    # Load the values of the 'perf' counters:
    # cache_misses: total number of cache misses during execution, across all cache levels (L1, L2, L3).
    cache_misses = get_value(possible_keys["cache-misses"]) or 0
    # cache_references: total number of cache accesses (hits + misses).
    cache_references = get_value(possible_keys["cache-references"]) or 0
    # instructions: total number of CPU instructions executed.
    instructions = get_value(possible_keys["instructions"]) or 0
    # cycles: total CPU cycles used during execution.
    cycles = get_value(possible_keys["cycles"]) or 0
    # seconds_time_elapsed: total wall-clock time of the 'perf' execution.
    seconds_time_elapsed = get_value(possible_keys["seconds_time_elapsed"]) or 0
    # Calculate derived metrics:
    # cache_miss_ratio: shows how well caches are being used.
    # Calculate how often a cache reference results in a miss (a higher value is worse).
    cache_miss_ratio = cache_misses / cache_references if cache_references else 0
    # instructions_per_cycle: how many instructions the CPU executes per cycle.
    # Measure how efficiently the CPU is utilized (a higher IPC is better).
    instructions_per_cycle = instructions / cycles if cycles else 0
    # Estimate how many bytes were loaded from main memory due to cache misses,
    # assuming that each cache miss causes the CPU to load one full line.
    memory_traffic_in_bytes = cache_misses * cache_line_size_in_bytes
    # memory_bandwidth_usage_in_Bps: how many B/s of memory bandwidth were actually used.
    # Calculate the effective memory bandwidth usage during execution (in B/s).
    memory_bandwidth_usage_in_Bps = (memory_traffic_in_bytes / seconds_time_elapsed if seconds_time_elapsed else 0)
    # Set the dictionary of 'perf' metrics.
    perf_metrics = {"cache_misses": cache_misses,
                    "cache_references": cache_references,
                    "cache_miss_ratio": cache_miss_ratio,
                    "instructions": instructions,
                    "cycles": cycles,
                    "seconds_time_elapsed": seconds_time_elapsed,
                    "instructions_per_cycle": instructions_per_cycle,
                    "memory_traffic_in_bytes": memory_traffic_in_bytes,
                    "memory_bandwidth_usage_in_Bps": memory_bandwidth_usage_in_Bps}
    # Return the dictionary of 'perf' metrics.
    return perf_metrics


def get_memory_traffic_per_flop_in_bytes(total_memory_traffic_in_bytes: float,
                                         total_flops: int) -> float:
    if total_memory_traffic_in_bytes == 0 or total_flops == 0:
        return 1
    return total_memory_traffic_in_bytes / total_flops


def get_memory_efficiency_rate(peak_memory_bandwidth_in_Bps: float,
                               memory_bandwidth_usage_in_Bps: float) -> float:
    memory_efficiency_rate = memory_bandwidth_usage_in_Bps / peak_memory_bandwidth_in_Bps
    return memory_efficiency_rate


def get_model_parameters(model: Model) -> list:
    model_parameters = model.get_weights()
    return model_parameters


def get_model_num_parameters(model: Model) -> int:
    model_num_parameters = model.count_params()
    return model_num_parameters


def get_model_precision_in_bits(model: Model) -> int:
    model_precision_in_bits = 32  # Fallback: 32-bit floating-point precision.
    match = search(r"\d+", model.dtype)
    if match:
        model_precision_in_bits = int(match.group())
    return model_precision_in_bits


def get_model_input_shape(model: Model) -> tuple:
    model_input_shape = tuple(list(model.input_shape[1:]))
    return model_input_shape


def get_model_output_classes(model: Model) -> int:
    model_output_classes = model.output_shape[-1]
    return model_output_classes


def get_ndarray_num_parameters(nd_arrays: List[ndarray]) -> int:
    ndarray_num_parameters = sum(param.size for param in nd_arrays)
    return ndarray_num_parameters


def get_ndarray_precision_in_bits(nd_arrays: List[ndarray]) -> int:
    ndarray_precision_in_bits = 32  # Fallback: 32-bit floating-point precision.
    if nd_arrays:
        dtype = nd_arrays[0].dtype  # Assuming all arrays have the same dtype.
        ndarray_precision_in_bits = np_dtype(dtype).itemsize * 8  # Convert bytes to bits.
    return ndarray_precision_in_bits


def get_num_classes(y: ndarray) -> int:
    num_classes = unique(y).shape[0]
    return num_classes


def get_total_size_in_bits(values_dict: dict,
                           int_size: int = 32,
                           float_size: int = 64,
                           char_size: int = 8) -> int:
    total_size_in_bits = 0
    for value in values_dict.values():
        if isinstance(value, int):
            total_size_in_bits += int_size
        elif isinstance(value, float):
            total_size_in_bits += float_size
        elif isinstance(value, str):
            total_size_in_bits += len(value) * char_size
    return total_size_in_bits


def convert_duration(duration: float,
                     from_unit: str,
                     to_unit: str) -> float:
    unit_multipliers = {"s": 1, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9, "min": 60, "h": 3600}
    if from_unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'!\nSupported units: {1}"
                         .format(from_unit, ", ".join(unit_multipliers.keys())))
    if to_unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'!\nSupported units: {1}"
                         .format(to_unit, ", ".join(unit_multipliers.keys())))
    if from_unit == to_unit:
        return duration
    # Convert the duration to the base unit (seconds), then to the target unit.
    duration_in_seconds = duration * unit_multipliers[from_unit]
    converted_duration = duration_in_seconds / unit_multipliers[to_unit]
    return converted_duration


def measure_actual_round_trip_time(target_ip: str,
                                   target_port: int,
                                   time_unit: str = "s",
                                   connection_timeout: int = 10) -> float:
    rtt = inf
    try:
        rtt_start_timer = time()
        with create_connection(address=(target_ip, target_port), timeout=connection_timeout):
            rtt = (time() - rtt_start_timer)
            rtt = convert_duration(rtt, "s", time_unit)
    except (timeout, gaierror, ConnectionRefusedError, OSError):
        # timeout: the connection attempt took longer than the specified timeout.
        # gaierror: the DNS resolution failed (can't resolve the hostname to an IP).
        # ConnectionRefusedError: the host actively refused the connection on that port.
        # OSError: the network is unreachable, bad socket, file descriptor issues, etc.
        pass
    return rtt


def calculate_haversine_distance_in_kilometers(latitude1: float,
                                               longitude1: float,
                                               latitude2: float,
                                               longitude2: float) -> float:
    earth_radius_in_km = 6371  # Earth's radius in kilometers.
    delta_lat = radians(latitude2 - latitude1)  # Difference in latitude in radians.
    delta_lon = radians(longitude2 - longitude1)  # Difference in longitude in radians.
    lat1_rad = radians(latitude1)
    lat2_rad = radians(latitude2)
    haversine_component = sin(delta_lat / 2) ** 2 + \
                          cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    central_angle = 2 * atan2(sqrt(haversine_component), sqrt(1 - haversine_component))
    distance_km = earth_radius_in_km * central_angle
    return distance_km


def calculate_all_pairwise_locations_distances_in_kilometers(locations_coordinates: dict) -> dict:
    pairwise_locations_distances = {}
    for location1, (latitude1, longitude1) in locations_coordinates.items():
        for location2, (latitude2, longitude2) in locations_coordinates.items():
            if location1 != location2:
                distance_in_kilometers = calculate_haversine_distance_in_kilometers(latitude1,
                                                                                    longitude1,
                                                                                    latitude2,
                                                                                    longitude2)
                pairwise_locations_distances["{0} --> {1}".format(location1, location2)] = distance_in_kilometers
    return pairwise_locations_distances


def calculate_locations_distance_in_kilometers(origin_location: str,
                                               destination_location: str) -> float:
    distance_in_kilometers = 0
    if origin_location not in LOCATIONS_COORDINATES or destination_location not in LOCATIONS_COORDINATES:
        message = "Error: One or both locations not found: {0}, {1}".format(origin_location, destination_location)
        raise ValueError(message)
    latitude1, longitude1 = LOCATIONS_COORDINATES[origin_location]
    latitude2, longitude2 = LOCATIONS_COORDINATES[destination_location]
    if origin_location != destination_location:
        distance_in_kilometers = calculate_haversine_distance_in_kilometers(latitude1,
                                                                            longitude1,
                                                                            latitude2,
                                                                            longitude2)
    return distance_in_kilometers


def estimate_distance_based_latency(client_location: str,
                                    server_location: str,
                                    client_base_latency: float,
                                    client_base_latency_unit: str,
                                    time_unit: str) -> float:
    # Get the geographical distance between client i and the server (in kilometers).
    geo_distance_km = calculate_locations_distance_in_kilometers(client_location, server_location)
    # Approximate speed of light in fiber optics (km/ms).
    fiber_optics_light_speed_km_per_ms = 200
    # Cast the client base latency to milliseconds.
    client_base_latency_in_milliseconds = convert_duration(client_base_latency, client_base_latency_unit, "ms")
    # Estimate the distance-based latency between client i and the server.
    distance_based_latency = client_base_latency_in_milliseconds + ((2 * geo_distance_km) / fiber_optics_light_speed_km_per_ms)
    distance_based_latency = convert_duration(distance_based_latency, "ms", time_unit)
    return distance_based_latency


def convert_network_bandwidth(network_bandwidth: float,
                              from_unit: str,
                              to_unit: str) -> float:
    unit_multipliers = {"bps": 1, "Kbps": 1e3, "Mbps": 1e6, "Gbps": 1e9,
                        "Bps": 1/8, "KBps": 1e3/8, "MBps": 1e6/8, "GBps": 1e9/8}
    if from_unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'!\nSupported units: {1}"
                         .format(from_unit, ", ".join(unit_multipliers.keys())))
    if to_unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'!\nSupported units: {1}"
                         .format(to_unit, ", ".join(unit_multipliers.keys())))
    if from_unit == to_unit:
        return network_bandwidth
    # Convert the network bandwidth to the base unit (bps), then to the target unit.
    network_bandwidth_in_bps = network_bandwidth * unit_multipliers[from_unit]
    converted_network_bandwidth = network_bandwidth_in_bps / unit_multipliers[to_unit]
    return converted_network_bandwidth


def convert_data_size(data_size: float,
                      from_unit: str,
                      to_unit: str) -> float:
    unit_multipliers = {"bits": 1, "B": 8, "KB": 8 * 1e3, "MB": 8 * 1e6,
                        "GB": 8 * 1e9, "TB": 8 * 1e12, "Kbits": 1e3,
                        "Mbits": 1e6, "Gbits": 1e9, "Tbits": 1e12}
    if from_unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'!\nSupported units: {1}"
                         .format(from_unit, ", ".join(unit_multipliers.keys())))
    if to_unit not in unit_multipliers:
        raise ValueError("Unsupported unit '{0}'!\nSupported units: {1}"
                         .format(to_unit, ", ".join(unit_multipliers.keys())))
    if from_unit == to_unit:
        return data_size
    # Convert the data size to the base unit (bits), then to the target unit.
    data_size_in_bits = data_size * unit_multipliers[from_unit]
    converted_data_size = data_size_in_bits / unit_multipliers[to_unit]
    return converted_data_size


def estimate_number_of_tcp_acks(data_size_in_bytes: float,
                                bandwidth_in_bytes_per_second: float,
                                rtt_in_milliseconds: float,
                                mss_in_bytes: int,
                                tcp_receive_window_size_in_bytes: int = 65536) -> int:
    # Calculate the number of segments.
    num_segments = data_size_in_bytes / mss_in_bytes
    # Use delayed ACK ratio when bandwidth is high (>= 1MBps) and latency is low (< 50 ms).
    if rtt_in_milliseconds < 50 and bandwidth_in_bytes_per_second >= 1e6:
        ack_ratio = 0.5  # One ACK per 2 segments (Delayed ACK).
    else:
        ack_ratio = 1.0  # Regular ACK (1 per segment).
    # Calculate the number of TCP windows needed.
    num_tcp_windows = data_size_in_bytes / tcp_receive_window_size_in_bytes
    # Number of ACKs considering delayed ACKs and sliding window.
    num_tcp_acks = max(int(num_segments * ack_ratio / num_tcp_windows), 1)
    return num_tcp_acks


def estimate_transfer_time(data_size_in_bytes: float,
                           bandwidth_in_bytes_per_second: float,
                           num_tcp_acks: int,
                           rtt_in_milliseconds: float) -> float:
    # Convert RTT from milliseconds to seconds.
    rtt_in_seconds = convert_duration(rtt_in_milliseconds, "ms", "s")
    # Calculate the time for data transfer.
    data_transfer_time_in_seconds = data_size_in_bytes / bandwidth_in_bytes_per_second
    # Calculate the total time including TCP ACKs.
    estimated_transfer_time_in_seconds = data_transfer_time_in_seconds + (num_tcp_acks * rtt_in_seconds)
    return estimated_transfer_time_in_seconds


def calculate_initial_parameters_upload_time(client_attributes: dict,
                                             model: Model,
                                             packet_loss_event_occurred: bool = False,
                                             packet_loss_rate: float = 0) -> float:
    # Get the necessary attributes.
    upload_bandwidth = client_attributes["upload_bandwidth"]  # Upload bandwidth of the client i.
    upload_bandwidth_unit = client_attributes["upload_bandwidth_unit"]  # Upload bandwidth unit of the client i.
    base_latency = client_attributes["base_latency"]  # Base latency of the client i.
    base_latency_unit = client_attributes["base_latency_unit"]  # Base latency unit of the client i.
    mss_ipv4_in_bytes = client_attributes["mss_ipv4_in_bytes"]  # Maximum segment size IPv4 in bytes of the client i.
    server_location = client_attributes["server_location"]  # Geographic location of the server.
    client_location = client_attributes["client_location"]  # Geographic location of the client i.
    # Get the number of parameters in the model.
    np_m = get_model_num_parameters(model)
    # Get the model's floating-point precision (bits per parameter).
    fpp_m = get_model_precision_in_bits(model)
    # Calculate the data size in bytes (to be transferred).
    data_size_in_bits = np_m * fpp_m
    data_size_in_bytes = convert_data_size(data_size_in_bits, "bits", "B")
    # Convert the upload bandwidth to Bytes per second.
    upload_bandwidth_in_bytes_per_second = convert_network_bandwidth(upload_bandwidth,
                                                                     upload_bandwidth_unit,
                                                                     "Bps")
    # Estimate the distance-based latency between this client and the server.
    rtt_in_milliseconds = estimate_distance_based_latency(client_location,
                                                          server_location,
                                                          base_latency,
                                                          base_latency_unit,
                                                          "ms")
    # Estimate the number of TCP ACKs needed for this client-server upload interaction.
    num_tcp_acks_upload = estimate_number_of_tcp_acks(data_size_in_bytes,
                                                      upload_bandwidth_in_bytes_per_second,
                                                      rtt_in_milliseconds,
                                                      mss_ipv4_in_bytes)
    # Estimate the time needed to upload the local model parameters to the server.
    initial_parameters_upload_time_in_seconds = estimate_transfer_time(data_size_in_bytes,
                                                                       upload_bandwidth_in_bytes_per_second,
                                                                       num_tcp_acks_upload,
                                                                       rtt_in_milliseconds)
    # Calculate the number of packets.
    num_packets = int(data_size_in_bytes // mss_ipv4_in_bytes)
    if data_size_in_bytes % mss_ipv4_in_bytes != 0:
        num_packets += 1
    # Simulate the loss of packets.
    num_lost_packets = 0
    retransmission_time_in_seconds = 0
    if packet_loss_event_occurred and 0.0 < packet_loss_rate <= 1.0:
        for _ in range(num_packets):
            if random() < packet_loss_rate:
                num_lost_packets += 1
        # Calculate the retransmission time, in seconds.
        retransmission_time_in_seconds = (num_lost_packets * rtt_in_milliseconds) / 1000.0
    # Adjust the upload time considering the occurrence of packet losses.
    initial_parameters_upload_time_in_seconds += retransmission_time_in_seconds
    # Return the time needed to upload the local model parameters to the server.
    return initial_parameters_upload_time_in_seconds


def calculate_download_time(client_attributes: dict,
                            phase: str,
                            global_parameters: NDArrays = None,
                            phase_config: dict = None,
                            packet_loss_event_occurred: bool = False,
                            packet_loss_rate: float = 0) -> tuple:
    # Calculate the data size in bytes (to be transferred).
    if "np_m" in client_attributes:
        np_m_in_bytes = client_attributes["np_m"]
    else:
        # Get the number of parameters in the model.
        np_m = get_ndarray_num_parameters(global_parameters)
        np_m_in_bytes = convert_data_size(np_m, "bits", "B")
    if "fpp_m" in client_attributes:
        fpp_m_in_bytes = client_attributes["fpp_m"]
    else:
        # Get the model's floating-point precision (bits per parameter).
        fpp_m = get_ndarray_precision_in_bits(global_parameters)
        fpp_m_in_bytes = convert_data_size(fpp_m, "bits", "B")
    if "is_{0}_i".format(phase) in client_attributes:
        is_phase_in_bytes = client_attributes["is_{0}_i".format(phase)]
    else:
        # Get the phase instructions size in bits.
        is_phase = get_total_size_in_bits(phase_config)
        is_phase_in_bytes = convert_data_size(is_phase, "bits", "B")
    # Get the size of data to be downloaded, in bytes.
    data_size_in_bytes = (np_m_in_bytes * fpp_m_in_bytes) + is_phase_in_bytes
    if "bw_down_i" in client_attributes:
        download_bandwidth_in_bytes_per_second = client_attributes["bw_down_i"]
    else:
        # Convert the download bandwidth to Bytes per second.
        download_bandwidth = client_attributes["download_bandwidth"]
        download_bandwidth_unit = client_attributes["download_bandwidth_unit"]
        download_bandwidth_in_bytes_per_second = convert_network_bandwidth(download_bandwidth,
                                                                           download_bandwidth_unit,
                                                                           "Bps")
    if "rtt_down_i" in client_attributes:
        rtt_in_milliseconds = client_attributes["rtt_down_i"]
    else:
        # Estimate the distance-based latency between this client and the server.
        client_location = client_attributes["client_location"]
        server_location = client_attributes["server_location"]
        base_latency = client_attributes["base_latency"]
        base_latency_unit = client_attributes["base_latency_unit"]
        rtt_in_milliseconds = estimate_distance_based_latency(client_location,
                                                              server_location,
                                                              base_latency,
                                                              base_latency_unit,
                                                              "ms")
    if "ack_down" in client_attributes:
        num_tcp_acks_download = client_attributes["ack_down"]
    else:
        # Estimate the number of TCP ACKs needed for this client-server download interaction.
        mss_ipv4_in_bytes = client_attributes["mss_ipv4_in_bytes"]
        num_tcp_acks_download = estimate_number_of_tcp_acks(data_size_in_bytes,
                                                            download_bandwidth_in_bytes_per_second,
                                                            rtt_in_milliseconds,
                                                            mss_ipv4_in_bytes)
    # Estimate the time needed to download the global model parameters and instruction from the server.
    download_time_in_seconds = estimate_transfer_time(data_size_in_bytes,
                                                      download_bandwidth_in_bytes_per_second,
                                                      num_tcp_acks_download,
                                                      rtt_in_milliseconds)
    if "num_packets" in client_attributes:
        num_packets = client_attributes["num_packets"]
    else:
        # Calculate the number of packets.
        mss_ipv4_in_bytes = client_attributes["mss_ipv4_in_bytes"]
        num_packets = int(data_size_in_bytes // mss_ipv4_in_bytes)
        if data_size_in_bytes % mss_ipv4_in_bytes != 0:
            num_packets += 1
    if "num_lost_packets" in client_attributes and "retransmission_time_in_seconds" in client_attributes:
        num_lost_packets = client_attributes["num_lost_packets"]
        retransmission_time_in_seconds = client_attributes["retransmission_time_in_seconds"]
    else:
        # Simulate the loss of packets.
        num_lost_packets = 0
        retransmission_time_in_seconds = 0
        if packet_loss_event_occurred and 0.0 < packet_loss_rate <= 1.0:
            for _ in range(num_packets):
                if random() < packet_loss_rate:
                    num_lost_packets += 1
            # Calculate the retransmission time, in seconds.
            retransmission_time_in_seconds = (num_lost_packets * rtt_in_milliseconds) / 1000.0
    # Adjust the download time considering the occurrence of packet losses.
    download_time_in_seconds += retransmission_time_in_seconds
    # Initialize the set of client download metrics.
    client_down_metrics = {"np_m": np_m_in_bytes,
                           "fpp_m": fpp_m_in_bytes,
                           "is_{0}_i".format(phase): is_phase_in_bytes,
                           "bw_down_i": download_bandwidth_in_bytes_per_second,
                           "ack_down": num_tcp_acks_download,
                           "rtt_down_i": rtt_in_milliseconds,
                           "num_packets": num_packets,
                           "num_lost_packets": num_lost_packets,
                           "retransmission_time_in_seconds": retransmission_time_in_seconds}
    # Return the estimated download time (in seconds) and the set of client download metrics.
    return download_time_in_seconds, client_down_metrics


def calculate_total_floating_point_operations_i_r(client_attributes: dict,
                                                  phase: str) -> float:
    fpo_m = 0
    bs_i = 0
    ds_i = 0
    e_i = 0
    match phase:
        case "train":
            fpo_m = client_attributes["fpo_train_m"]  # Number of floating-point operations required to process a single training sample.
            bs_i = client_attributes["bs_train_i"]  # Batch size used by the client to train the model.
            ds_i = client_attributes["ds_train_i"]  # Number of training samples used by the client to train the model.
            e_i = client_attributes["e_i"]  # Number of epochs used by the client to train the model.
        case "test":
            fpo_m = client_attributes["fpo_test_m"]  # Number of floating-point operations required to process a single test sample.
            bs_i = client_attributes["bs_test_i"]  # Batch size used by the client to test the model.
            ds_i = client_attributes["ds_test_i"]  # Number of test samples used by the client to test the model.
            # Epochs are defined only when training the model (for test, a single pass of the whole test samples occurs)...
            e_i = 1
    # Calculate the total number of floating-point operations required for the client to train/test the model.
    tfpo_m_i = fpo_m * bs_i * ceil(ds_i / bs_i) * e_i
    # Return the total number of floating-point operations.
    return tfpo_m_i


def calculate_effective_flop_throughput_i_r(client_attributes: dict,
                                            phase: str) -> float:
    nc_cpu_i = client_attributes["nc_cpu_i"]  # Number of CPU cores of the client.
    fq_cpu_i = client_attributes["fq_cpu_i"]  # CPU frequency of the client (in Hertz).
    fpocc_m_i = 0
    match phase:
        case "train":
            # Mean number of floating-point operations executed by the client per cycle per CPU core when training the model.
            fpocc_m_i = client_attributes["fpocc_m_i_train"]
        case "test":
            # Mean number of floating-point operations executed by the client per cycle per CPU core when testing the model.
            fpocc_m_i = client_attributes["fpocc_m_i_test"]
    # Calculate the effective floating-point operations throughput of the client when training/testing the model (in flops per second).
    etp_m_i = nc_cpu_i * fpocc_m_i * fq_cpu_i
    # Return the effective floating-point operations throughput.
    return etp_m_i


def estimate_mt_m_device(mt_m_node: float,
                         mem_bw_node: float,
                         cpu_gflops_node: float,
                         mem_bw_device: float,
                         cpu_gflops_device: float) -> float:
    # Assuming similar memory access patterns and no drastic changes in model architecture...
    mem_bw_ratio = mem_bw_device / mem_bw_node
    cpu_gflops_ratio = cpu_gflops_device / cpu_gflops_node
    mt_m_device = mt_m_node * (mem_bw_ratio / cpu_gflops_ratio)
    return mt_m_device


def calculate_memory_slowdown_factor_i_r(client_attributes: dict,
                                         etp_m_i: float,
                                         phase: str) -> float:
    bw_mem_i = client_attributes["bw_mem_i"]  # Memory bandwidth of the client (in bytes per second).
    mt_m = 0
    match phase:
        case "train":
            # Mean data traffic of the client's main memory when training the model (in bytes per floating-point operation).
            mt_m = client_attributes["mt_m_train"]
        case "test":
            # Mean data traffic of the client's main memory when testing the model (in bytes per floating-point operation).
            mt_m = client_attributes["mt_m_test"]
    # Calculate the memory slowdown factor of the client when training/testing the model.
    msd_m_i = min(1, (bw_mem_i / (etp_m_i * mt_m)))
    # Return the memory slowdown factor.
    return msd_m_i


def calculate_computation_time(client_attributes: dict,
                               phase: str) -> float:
    # Estimate the total number of floating-point operations required for the client to train/test the model.
    tfpo_m_i = calculate_total_floating_point_operations_i_r(client_attributes, phase)
    # Estimate the effective floating-point operations throughput of the client when training/testing the model.
    etp_m_i = calculate_effective_flop_throughput_i_r(client_attributes, phase)
    # Estimate the memory slowdown factor of the client when training/testing the model.
    msd_m_i = calculate_memory_slowdown_factor_i_r(client_attributes, etp_m_i, phase)
    # Estimate the time spent with computation by a client i on round r (in seconds).
    computation_time_in_seconds = tfpo_m_i / (etp_m_i * msd_m_i)
    # Return the estimated computation time (in seconds).
    return computation_time_in_seconds


def calculate_upload_time(client_attributes: dict,
                          phase: str,
                          phase_metrics: dict = None,
                          packet_loss_event_occurred: bool = False,
                          packet_loss_rate: float = 0,
                          local_parameters: NDArrays = None) -> tuple:
    if "ms_{0}_i".format(phase) in client_attributes:
        ms_phase_in_bytes = client_attributes["ms_{0}_i".format(phase)]
    else:
        # Get the phase metrics size in bits.
        ms_phase = get_total_size_in_bits(phase_metrics)
        ms_phase_in_bytes = convert_data_size(ms_phase, "bits", "B")
    # Get the size of data to be uploaded, in bytes.
    data_size_in_bytes = 0
    np_m_in_bytes = 0
    fpp_m_in_bytes = 0
    match phase:
        case "train":
            if "np_m" in client_attributes:
                np_m_in_bytes = client_attributes["np_m"]
            else:
                # Get the number of parameters in the model.
                np_m = get_ndarray_num_parameters(local_parameters)
                np_m_in_bytes = convert_data_size(np_m, "bits", "B")
            if "fpp_m" in client_attributes:
                fpp_m_in_bytes = client_attributes["fpp_m"]
            else:
                # Get the model's floating-point precision (bits per parameter).
                fpp_m = get_ndarray_precision_in_bits(local_parameters)
                fpp_m_in_bytes = convert_data_size(fpp_m, "bits", "B")
            data_size_in_bytes = (np_m_in_bytes * fpp_m_in_bytes) + ms_phase_in_bytes
        case "test":
            data_size_in_bytes = ms_phase_in_bytes
    if "bw_up_i" in client_attributes:
        upload_bandwidth_in_bytes_per_second = client_attributes["bw_up_i"]
    else:
        # Convert the upload bandwidth to Bytes per second.
        upload_bandwidth = client_attributes["upload_bandwidth"]
        upload_bandwidth_unit = client_attributes["upload_bandwidth_unit"]
        upload_bandwidth_in_bytes_per_second = convert_network_bandwidth(upload_bandwidth,
                                                                         upload_bandwidth_unit,
                                                                         "Bps")
    if "rtt_up_i" in client_attributes:
        rtt_in_milliseconds = client_attributes["rtt_up_i"]
    else:
        # Estimate the distance-based latency between this client and the server.
        client_location = client_attributes["client_location"]
        server_location = client_attributes["server_location"]
        base_latency = client_attributes["base_latency"]
        base_latency_unit = client_attributes["base_latency_unit"]
        rtt_in_milliseconds = estimate_distance_based_latency(client_location,
                                                              server_location,
                                                              base_latency,
                                                              base_latency_unit,
                                                              "ms")
    if "ack_up" in client_attributes:
        num_tcp_acks_upload = client_attributes["ack_up"]
    else:
        # Estimate the number of TCP ACKs needed for this client-server upload interaction.
        mss_ipv4_in_bytes = client_attributes["mss_ipv4_in_bytes"]
        num_tcp_acks_upload = estimate_number_of_tcp_acks(data_size_in_bytes,
                                                          upload_bandwidth_in_bytes_per_second,
                                                          rtt_in_milliseconds,
                                                          mss_ipv4_in_bytes)
    # Estimate the time needed to upload the local model parameters and metrics to the server.
    upload_time_in_seconds = estimate_transfer_time(data_size_in_bytes,
                                                    upload_bandwidth_in_bytes_per_second,
                                                    num_tcp_acks_upload,
                                                    rtt_in_milliseconds)
    if "num_packets" in client_attributes:
        num_packets = client_attributes["num_packets"]
    else:
        # Calculate the number of packets.
        mss_ipv4_in_bytes = client_attributes["mss_ipv4_in_bytes"]
        num_packets = int(data_size_in_bytes // mss_ipv4_in_bytes)
        if data_size_in_bytes % mss_ipv4_in_bytes != 0:
            num_packets += 1
    if "num_lost_packets" in client_attributes and "retransmission_time_in_seconds" in client_attributes:
        num_lost_packets = client_attributes["num_lost_packets"]
        retransmission_time_in_seconds = client_attributes["retransmission_time_in_seconds"]
    else:
        # Simulate the loss of packets.
        num_lost_packets = 0
        retransmission_time_in_seconds = 0
        if packet_loss_event_occurred and 0.0 < packet_loss_rate <= 1.0:
            for _ in range(num_packets):
                if random() < packet_loss_rate:
                    num_lost_packets += 1
            # Calculate the retransmission time, in seconds.
            retransmission_time_in_seconds = (num_lost_packets * rtt_in_milliseconds) / 1000.0
    # Adjust the upload time considering the occurrence of packet losses.
    upload_time_in_seconds += retransmission_time_in_seconds
    # Initialize the set of client upload metrics.
    client_up_metrics = {}
    if phase == "train":
        client_up_metrics.update({"np_m": np_m_in_bytes, "fpp_m": fpp_m_in_bytes})
    client_up_metrics = {"ms_{0}_i".format(phase): ms_phase_in_bytes,
                         "bw_up_i": upload_bandwidth_in_bytes_per_second,
                         "ack_up": num_tcp_acks_upload,
                         "rtt_up_i": rtt_in_milliseconds,
                         "num_packets": num_packets,
                         "num_lost_packets": num_lost_packets,
                         "retransmission_time_in_seconds": retransmission_time_in_seconds}
    # Return the estimated upload time (in seconds) and the set of client upload metrics.
    return upload_time_in_seconds, client_up_metrics


def calculate_initialization_energy(client_attributes: dict,
                                    initialization_time_in_seconds: float) -> float:
    # Get the necessary attributes.
    base_latency = client_attributes["base_latency"]
    base_latency_unit = client_attributes["base_latency_unit"]
    server_location = client_attributes["server_location"]
    client_location = client_attributes["client_location"]
    # Mean power consumption of the client i when under a heavy computational load (in watts).
    mpc_comp_i = client_attributes["mpc_comp_i"]
    # Mean power consumption of the client i when sending data (in watts).
    mpc_send_i = client_attributes["mpc_send_i"]
    # Estimate the distance-based latency between this client and the server.
    rtt_in_seconds = estimate_distance_based_latency(client_location,
                                                     server_location,
                                                     base_latency,
                                                     base_latency_unit,
                                                     "s")
    # Get the time spent by the client i with local loading (e.g., model and dataset).
    loading_time_in_seconds = initialization_time_in_seconds - rtt_in_seconds
    # Estimate the energy consumed by the client i during the initialization event.
    initialization_energy_in_joules = (loading_time_in_seconds * mpc_comp_i) + (rtt_in_seconds * mpc_send_i)
    # Return the estimated initialization energy (in joules).
    return initialization_energy_in_joules


def calculate_download_energy(client_attributes: dict,
                              download_time_in_seconds: float) -> float:
    # Mean power consumption of the client i when receiving data (in watts).
    mpc_recv_i = client_attributes["mpc_recv_i"]
    # Estimate the energy consumed by the client i during a download event.
    download_energy_in_joules = download_time_in_seconds * mpc_recv_i
    # Return the estimated download energy (in joules).
    return download_energy_in_joules


def calculate_computation_energy(client_attributes: dict,
                                 computation_time_in_seconds: float) -> float:
    # Mean power consumption of the client i when under a heavy computational load (in watts).
    mpc_comp_i = client_attributes["mpc_comp_i"]
    # Estimate the energy consumed by the client i during a computation event.
    computation_energy_in_joules = computation_time_in_seconds * mpc_comp_i
    # Return the estimated computation energy (in joules).
    return computation_energy_in_joules


def calculate_upload_energy(client_attributes: dict,
                            upload_time_in_seconds: float) -> float:
    # Mean power consumption of the client i when sending data (in watts).
    mpc_send_i = client_attributes["mpc_send_i"]
    # Estimate the energy consumed by the client i during an upload event.
    upload_energy_in_joules = upload_time_in_seconds * mpc_send_i
    # Return the estimated upload energy (in joules).
    return upload_energy_in_joules


def calculate_idle_energy(client_attributes: dict,
                          idle_time_in_seconds: float) -> float:
    # Mean power consumption of the client i when in idle mode (in watts).
    mpc_idle_i = client_attributes["mpc_idle_i"]
    # Estimate the energy consumed by the client i during an idle event.
    idle_energy_in_joules = idle_time_in_seconds * mpc_idle_i
    # Return the estimated idle energy (in joules).
    return idle_energy_in_joules
