from pathlib import Path


def calculate_fpocc(simd_lanes: int,
                    fma_units: int,
                    fma_per_cycle: int,
                    utilization_range: tuple) -> tuple:
    # simd_lanes: number of parallel lanes in SIMD (Single Instruction Multiple Data) unit.
    # fma_units: number of Fused Multiply-Add (FMA) units per core (or floating-point execution units).
    # fma_per_cycle: number of FMA instructions the core can issue per cycle.
    # utilization_range: the fraction of peak theoretical computational resources that are actually used.
    #  - during the training, lower effective fpocc (more complex operations, higher memory traffic, parameter updates).
    #  - during the inference, higher effective fpocc (static computation graph, batch processing, less memory overhead).
    base_fpocc = simd_lanes * max(1, fma_units) * max(1, fma_per_cycle)
    min_fpocc = base_fpocc * utilization_range[0]
    max_fpocc = base_fpocc * utilization_range[1]
    return min_fpocc, max_fpocc


def calculate_peak_cpu_gflops(cpu_num_cores: int,
                              cpu_frequency_in_hertz: float,
                              simd_lanes: int,
                              fma_units: int,
                              fma_per_cycle: int) -> float:
    peak_cpu_gflops = round((cpu_num_cores * cpu_frequency_in_hertz * simd_lanes * max(1, fma_units) * max(1, fma_per_cycle)) / 1e9, 2)
    return peak_cpu_gflops


def calculate_battery_maximum_stored_energy(battery_voltage_in_volts: float,
                                            battery_capacity_in_milli_ampere_hours: float,
                                            battery_stored_energy_unit: str) -> float:
    battery_capacity_in_ah = battery_capacity_in_milli_ampere_hours / 1000  # Convert mAh to Ah.
    match battery_stored_energy_unit:
        case u if u in ["watt-hours", "Wh"]:
            return battery_voltage_in_volts * battery_capacity_in_ah
        case u if u in ["joules", "J"]:
            return battery_voltage_in_volts * battery_capacity_in_ah * 3600
    raise ValueError("Unsupported unit: {0}".format(battery_stored_energy_unit))


def emulate_intel_celeron_performance(device_name: str,
                                      power_source_connection_scenario: bool) -> dict:
    # Intel Celeron N3350
    # https://ark.intel.com/products/95598/
    # https://ark.intel.com/content/www/us/en/ark/products/95598/intel-celeron-processor-n3350-2m-cache-up-to-2-40-ghz.html
    # https://www.cpubenchmark.net/cpu.php?cpu=Intel+Celeron+N3350+%40+1.10GHz
    cpu_num_cores = 0
    cpu_frequency_in_hertz = 0
    simd_lanes = 0
    fma_units = 0
    fma_per_cycle = 0
    effective_utilization_range_for_training = ()
    effective_utilization_range_for_inference = ()
    memory_size_in_gigabytes = 0
    memory_speed_in_megahertz = 0
    memory_type = "Unknown"
    peak_memory_bandwidth_in_bytes_per_second = 0
    mean_power_consumption_data_reception_in_watts = 0
    mean_power_consumption_heavy_computational_load_in_watts = 0
    mean_power_consumption_data_transmission_in_watts = 0
    mean_power_consumption_idle_in_watts = 0
    battery_capacity_in_milli_ampere_hours = 0
    battery_voltage_in_volts = 0
    battery_type = "Unknown"
    match device_name:
        case "Intel Celeron N3350":
            cpu_num_cores = 2
            cpu_frequency_in_hertz = 1.1 * 1e9
            simd_lanes = 4
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.4, 0.7)
            effective_utilization_range_for_inference = (0.7, 0.9)
            memory_size_in_gigabytes = 4
            memory_speed_in_megahertz = 1866
            memory_type = "LPDDR4"
            peak_memory_bandwidth_in_bytes_per_second = 29.8 * 1e9
            mean_power_consumption_data_reception_in_watts = 4.5
            mean_power_consumption_heavy_computational_load_in_watts = 6.0
            mean_power_consumption_data_transmission_in_watts = 5.0
            mean_power_consumption_idle_in_watts = 2.0
            battery_capacity_in_milli_ampere_hours = 3000
            battery_voltage_in_volts = 5
            battery_type = "Lithium-Ion"
    peak_cpu_gflops = calculate_peak_cpu_gflops(cpu_num_cores, cpu_frequency_in_hertz, simd_lanes, fma_units, fma_per_cycle)
    fpocc_training = calculate_fpocc(simd_lanes, fma_units, fma_per_cycle, effective_utilization_range_for_training)
    fpocc_inference = calculate_fpocc(simd_lanes, fma_units, fma_per_cycle, effective_utilization_range_for_inference)
    battery_maximum_stored_energy_in_joules \
        = calculate_battery_maximum_stored_energy(battery_voltage_in_volts,
                                                  battery_capacity_in_milli_ampere_hours,
                                                  "joules")
    device_connected_to_a_power_source = True if power_source_connection_scenario == "connected_to_a_power_source" else False
    performance_dict = {"device_name": device_name,
                        "cpu_num_cores": cpu_num_cores,
                        "cpu_frequency_in_hertz": cpu_frequency_in_hertz,
                        "peak_cpu_gflops": peak_cpu_gflops,
                        "fpocc_training": fpocc_training,
                        "fpocc_training_min": fpocc_training[0],
                        "fpocc_training_max": fpocc_training[1],
                        "fpocc_inference": fpocc_inference,
                        "fpocc_inference_min": fpocc_inference[0],
                        "fpocc_inference_max": fpocc_inference[1],
                        "memory_size_in_gigabytes": memory_size_in_gigabytes,
                        "memory_speed_in_megahertz": memory_speed_in_megahertz,
                        "memory_type": memory_type,
                        "peak_memory_bandwidth_in_gigabytes_per_second": peak_memory_bandwidth_in_bytes_per_second / 1e9,
                        "peak_memory_bandwidth_in_bytes_per_second": peak_memory_bandwidth_in_bytes_per_second,
                        "mean_power_consumption_data_reception_in_watts": mean_power_consumption_data_reception_in_watts,
                        "mean_power_consumption_heavy_computational_load_in_watts": mean_power_consumption_heavy_computational_load_in_watts,
                        "mean_power_consumption_data_transmission_in_watts": mean_power_consumption_data_transmission_in_watts,
                        "mean_power_consumption_idle_in_watts": mean_power_consumption_idle_in_watts,
                        "battery_capacity_in_milli_ampere_hours": battery_capacity_in_milli_ampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_type": battery_type,
                        "battery_maximum_stored_energy_in_joules": battery_maximum_stored_energy_in_joules,
                        "device_connected_to_a_power_source": device_connected_to_a_power_source}
    return performance_dict


def emulate_intel_core_i_series_performance(device_name: str,
                                            power_source_connection_scenario: bool) -> dict:
    # Intel Core i3 10th Gen, Intel Core i5 10th Gen, Intel Core i7 10th Gen
    # https://ark.intel.com/products/199276/
    # https://ark.intel.com/products/195436/
    # https://ark.intel.com/products/196451/
    # https://ark.intel.com/content/www/us/en/ark/products/codename/90385/comet-lake.html
    # https://www.anandtech.com/show/14664/intel-comet-lake-10th-gen-14nm-desktop-cpu-review
    cpu_num_cores = 0
    cpu_frequency_in_hertz = 0
    simd_lanes = 0
    fma_units = 0
    fma_per_cycle = 0
    effective_utilization_range_for_training = ()
    effective_utilization_range_for_inference = ()
    memory_size_in_gigabytes = 0
    memory_speed_in_megahertz = 0
    memory_type = "Unknown"
    peak_memory_bandwidth_in_bytes_per_second = 0
    mean_power_consumption_data_reception_in_watts = 0
    mean_power_consumption_heavy_computational_load_in_watts = 0
    mean_power_consumption_data_transmission_in_watts = 0
    mean_power_consumption_idle_in_watts = 0
    battery_capacity_in_milli_ampere_hours = 0
    battery_voltage_in_volts = 0
    battery_type = "Unknown"
    match device_name:
        case "Intel Core i3 10th Gen":
            cpu_num_cores = 2
            cpu_frequency_in_hertz = 2.1 * 1e9
            simd_lanes = 8
            fma_units = 2
            fma_per_cycle = 2
            effective_utilization_range_for_training = (0.5, 0.85)
            effective_utilization_range_for_inference = (0.85, 0.99)
            memory_size_in_gigabytes = 8
            memory_speed_in_megahertz = 2666
            memory_type = "DDR4"
            peak_memory_bandwidth_in_bytes_per_second = 42.6 * 1e9
            mean_power_consumption_data_reception_in_watts = 8.0
            mean_power_consumption_heavy_computational_load_in_watts = 15.0
            mean_power_consumption_data_transmission_in_watts = 10.0
            mean_power_consumption_idle_in_watts = 3.0
            battery_capacity_in_milli_ampere_hours = 50000
            battery_voltage_in_volts = 11.1
            battery_type = "Lithium-Ion"
        case "Intel Core i5 10th Gen":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 1.6 * 1e9
            simd_lanes = 8
            fma_units = 2
            fma_per_cycle = 2
            effective_utilization_range_for_training = (0.5, 0.85)
            effective_utilization_range_for_inference = (0.85, 0.99)
            memory_size_in_gigabytes = 16
            memory_speed_in_megahertz = 2666
            memory_type = "DDR4"
            peak_memory_bandwidth_in_bytes_per_second = 42.6 * 1e9
            mean_power_consumption_data_reception_in_watts = 14.0
            mean_power_consumption_heavy_computational_load_in_watts = 25.0
            mean_power_consumption_data_transmission_in_watts = 18.0
            mean_power_consumption_idle_in_watts = 4.0
            battery_capacity_in_milli_ampere_hours = 50000
            battery_voltage_in_volts = 11.1
            battery_type = "Lithium-Ion"
        case "Intel Core i7 10th Gen":
            cpu_num_cores = 6
            cpu_frequency_in_hertz = 1.1 * 1e9
            simd_lanes = 8
            fma_units = 2
            fma_per_cycle = 2
            effective_utilization_range_for_training = (0.5, 0.85)
            effective_utilization_range_for_inference = (0.85, 0.99)
            memory_size_in_gigabytes = 32
            memory_speed_in_megahertz = 2666
            memory_type = "DDR4"
            peak_memory_bandwidth_in_bytes_per_second = 42.6 * 1e9
            mean_power_consumption_data_reception_in_watts = 25.0
            mean_power_consumption_heavy_computational_load_in_watts = 45.0
            mean_power_consumption_data_transmission_in_watts = 30.0
            mean_power_consumption_idle_in_watts = 5.0
            battery_capacity_in_milli_ampere_hours = 50000
            battery_voltage_in_volts = 11.1
            battery_type = "Lithium-Ion"
    peak_cpu_gflops = calculate_peak_cpu_gflops(cpu_num_cores, cpu_frequency_in_hertz, simd_lanes, fma_units, fma_per_cycle)
    fpocc_training = calculate_fpocc(simd_lanes, fma_units, fma_per_cycle, effective_utilization_range_for_training)
    fpocc_inference = calculate_fpocc(simd_lanes, fma_units, fma_per_cycle, effective_utilization_range_for_inference)
    battery_maximum_stored_energy_in_joules \
        = calculate_battery_maximum_stored_energy(battery_voltage_in_volts,
                                                  battery_capacity_in_milli_ampere_hours,
                                                  "joules")
    device_connected_to_a_power_source = True if power_source_connection_scenario == "connected_to_a_power_source" else False
    performance_dict = {"device_name": device_name,
                        "cpu_num_cores": cpu_num_cores,
                        "cpu_frequency_in_hertz": cpu_frequency_in_hertz,
                        "peak_cpu_gflops": peak_cpu_gflops,
                        "fpocc_training": fpocc_training,
                        "fpocc_training_min": fpocc_training[0],
                        "fpocc_training_max": fpocc_training[1],
                        "fpocc_inference": fpocc_inference,
                        "fpocc_inference_min": fpocc_inference[0],
                        "fpocc_inference_max": fpocc_inference[1],
                        "memory_size_in_gigabytes": memory_size_in_gigabytes,
                        "memory_speed_in_megahertz": memory_speed_in_megahertz,
                        "memory_type": memory_type,
                        "peak_memory_bandwidth_in_gigabytes_per_second": peak_memory_bandwidth_in_bytes_per_second / 1e9,
                        "peak_memory_bandwidth_in_bytes_per_second": peak_memory_bandwidth_in_bytes_per_second,
                        "mean_power_consumption_data_reception_in_watts": mean_power_consumption_data_reception_in_watts,
                        "mean_power_consumption_heavy_computational_load_in_watts": mean_power_consumption_heavy_computational_load_in_watts,
                        "mean_power_consumption_data_transmission_in_watts": mean_power_consumption_data_transmission_in_watts,
                        "mean_power_consumption_idle_in_watts": mean_power_consumption_idle_in_watts,
                        "battery_capacity_in_milli_ampere_hours": battery_capacity_in_milli_ampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_type": battery_type,
                        "battery_maximum_stored_energy_in_joules": battery_maximum_stored_energy_in_joules,
                        "device_connected_to_a_power_source": device_connected_to_a_power_source}
    return performance_dict


def generate_pc_class_devices(power_source_connection_scenarios: list) -> dict:
    pc_class_devices = {}
    for power_source_connection_scenario in power_source_connection_scenarios:
        pc_class_devices.update({"intel_celeron_n3350_device_{0}".format(power_source_connection_scenario): emulate_intel_celeron_performance("Intel Celeron N3350", power_source_connection_scenario),
                                 "intel_core_i3_10th_gen_device_{0}".format(power_source_connection_scenario): emulate_intel_core_i_series_performance("Intel Core i3 10th Gen", power_source_connection_scenario),
                                 "intel_core_i5_10th_gen_device_{0}".format(power_source_connection_scenario): emulate_intel_core_i_series_performance("Intel Core i5 10th Gen", power_source_connection_scenario),
                                 "intel_core_i7_10th_gen_device_{0}".format(power_source_connection_scenario): emulate_intel_core_i_series_performance("Intel Core i7 10th Gen", power_source_connection_scenario)})
    return pc_class_devices


def generate_pc_class_devices_configuration_file(output_file: Path,
                                                 power_source_connection_scenarios: list) -> None:
    # Create the parents directories of the output file (if not exist yet).
    output_file.parent.mkdir(exist_ok=True, parents=True)
    # Emulate PC-class devices performance.
    pc_class_devices = generate_pc_class_devices(power_source_connection_scenarios)
    with open(file=output_file, mode="a", encoding="utf-8") as file:
        # Get the dictionary of PC-class devices settings from the first device.
        _, pc_class_device_settings_dict = next(iter(pc_class_devices.items()))
        # Set and write the header line.
        header_line = ",".join(list(pc_class_device_settings_dict.keys()))
        file.write(header_line + "\n")
        # Set and write the data lines.
        data_lines = []
        for _, pc_class_device_settings_dict in pc_class_devices.items():
            data_line = ",".join(str(value) for value in pc_class_device_settings_dict.values())
            data_lines.append(data_line)
        file.writelines("\n".join(str(data_line) for data_line in data_lines))
        file.write("\n")
