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


def emulate_raspberry_pi_performance(device_name: str,
                                     power_source_connection_scenario: bool) -> dict:
    # Raspberry Pi 3, Raspberry Pi 4
    # https://www.raspberrypi.com/products/raspberry-pi-3-model-b/
    # https://www.raspberrypi.com/products/raspberry-pi-4-model-b/
    # https://www.jeffgeerling.com/blogs/jeff-geerling/benchmarking-raspberry-pi-3-b
    # https://developer.arm.com/documentation/ddi0500/e/
    # https://www.tomshardware.com/reviews/raspberry-pi-4-b,6193.html
    # https://developer.arm.com/documentation/ddi0489/latest
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
        case "Raspberry Pi 3":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 1.2 * 1e9
            simd_lanes = 2
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.3, 0.6)
            effective_utilization_range_for_inference = (0.6, 0.9)
            memory_size_in_gigabytes = 1
            memory_speed_in_megahertz = 900
            memory_type = "LPDDR2"
            peak_memory_bandwidth_in_bytes_per_second = 7.2 * 1e9
            mean_power_consumption_data_reception_in_watts = 2.5
            mean_power_consumption_heavy_computational_load_in_watts = 3.5
            mean_power_consumption_data_transmission_in_watts = 2.8
            mean_power_consumption_idle_in_watts = 1.5
            battery_capacity_in_milli_ampere_hours = 5000
            battery_voltage_in_volts = 5
            battery_type = "External Powerbank (Li-Ion)"
        case "Raspberry Pi 4":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 1.5 * 1e9
            simd_lanes = 4
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.4, 0.7)
            effective_utilization_range_for_inference = (0.7, 0.95)
            memory_size_in_gigabytes = 4
            memory_speed_in_megahertz = 3200
            memory_type = "LPDDR4"
            peak_memory_bandwidth_in_bytes_per_second = 25.6 * 1e9
            mean_power_consumption_data_reception_in_watts = 4.8
            mean_power_consumption_heavy_computational_load_in_watts = 6.4
            mean_power_consumption_data_transmission_in_watts = 5.2
            mean_power_consumption_idle_in_watts = 2.5
            battery_capacity_in_milli_ampere_hours = 5000
            battery_voltage_in_volts = 5
            battery_type = "External Powerbank (Li-Ion)"
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


def emulate_nvidia_jetson_performance(device_name: str,
                                      power_source_connection_scenario: bool) -> dict:
    # NVIDIA Jetson TX1, NVIDIA Jetson TX2, NVIDIA Jetson Nano, NVIDIA Jetson Xavier
    # https://developer.nvidia.com/embedded/jetson-tx1
    # https://developer.nvidia.com/embedded/jetson-tx2
    # https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/
    # https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-nx/
    # https://developer.nvidia.com/embedded/jetson-tx1
    # https://mlcommons.org/en/inference-edge-10/
    # https://developer.nvidia.com/embedded/dlc/jetson-tx2-module-datasheet
    # https://mlcommons.org/en/inference-edge-10/
    # https://developer.nvidia.com/embedded/jetson-nano
    # https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3271/index.html
    # https://developer.nvidia.com/embedded/dlc/jetson-agx-xavier-datasheet
    # https://mlcommons.org/en/inference-edge-10/
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
        case "NVIDIA Jetson TX1":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 1.9 * 1e9
            simd_lanes = 4
            fma_units = 2
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.4, 0.75)
            effective_utilization_range_for_inference = (0.75, 0.95)
            memory_size_in_gigabytes = 4
            memory_speed_in_megahertz = 1600
            memory_type = "LPDDR4"
            peak_memory_bandwidth_in_bytes_per_second = 25.6 * 1e9
            mean_power_consumption_data_reception_in_watts = 7.0
            mean_power_consumption_heavy_computational_load_in_watts = 10.0
            mean_power_consumption_data_transmission_in_watts = 8.0
            mean_power_consumption_idle_in_watts = 3.0
            battery_capacity_in_milli_ampere_hours = 20000
            battery_voltage_in_volts = 12
            battery_type = "Lithium-Polymer (Li-Po)"
        case "NVIDIA Jetson TX2":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 2.0 * 1e9
            simd_lanes = 4
            fma_units = 2
            fma_per_cycle = 2
            effective_utilization_range_for_training = (0.4, 0.8)
            effective_utilization_range_for_inference = (0.8, 0.98)
            memory_size_in_gigabytes = 8
            memory_speed_in_megahertz = 1866
            memory_type = "LPDDR4"
            peak_memory_bandwidth_in_bytes_per_second = 29.8 * 1e9
            mean_power_consumption_data_reception_in_watts = 8.0
            mean_power_consumption_heavy_computational_load_in_watts = 15.0
            mean_power_consumption_data_transmission_in_watts = 10.0
            mean_power_consumption_idle_in_watts = 2.5
            battery_capacity_in_milli_ampere_hours = 20000
            battery_voltage_in_volts = 12
            battery_type = "Lithium-Polymer (Li-Po)"
        case "NVIDIA Jetson Nano":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 1.43 * 1e9
            simd_lanes = 4
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.4, 0.7)
            effective_utilization_range_for_inference = (0.7, 0.95)
            memory_size_in_gigabytes = 4
            memory_speed_in_megahertz = 1600
            memory_type = "LPDDR4"
            peak_memory_bandwidth_in_bytes_per_second = 25.6 * 1e9
            mean_power_consumption_data_reception_in_watts = 5.0
            mean_power_consumption_heavy_computational_load_in_watts = 10.0
            mean_power_consumption_data_transmission_in_watts = 7.0
            mean_power_consumption_idle_in_watts = 2.0
            battery_capacity_in_milli_ampere_hours = 5000
            battery_voltage_in_volts = 3.7
            battery_type = "Lithium-Polymer (Li-Po) or Powerbank (Li-Ion)"
        case "NVIDIA Jetson Xavier":
            cpu_num_cores = 6
            cpu_frequency_in_hertz = 2.2 * 1e9
            simd_lanes = 8
            fma_units = 2
            fma_per_cycle = 2
            effective_utilization_range_for_training = (0.5, 0.85)
            effective_utilization_range_for_inference = (0.85, 0.99)
            memory_size_in_gigabytes = 8
            memory_speed_in_megahertz = 2133
            memory_type = "LPDDR4x"
            peak_memory_bandwidth_in_bytes_per_second = 137 * 1e9
            mean_power_consumption_data_reception_in_watts = 20.0
            mean_power_consumption_heavy_computational_load_in_watts = 30.0
            mean_power_consumption_data_transmission_in_watts = 25.0
            mean_power_consumption_idle_in_watts = 4.0
            battery_capacity_in_milli_ampere_hours = 20000
            battery_voltage_in_volts = 12
            battery_type = "Lithium-Polymer (Li-Po)"
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


def emulate_intel_atom_performance(device_name: str,
                                   power_source_connection_scenario: bool) -> dict:
    # Intel Atom x5-Z8350 2 Cores, Intel Atom x5-Z8350 4 Cores, Intel Atom x7-E3950
    # https://ark.intel.com/products/93361/
    # https://ark.intel.com/products/95592/
    # https://www.cpubenchmark.net/cpu.php?cpu=Intel+Atom+x5-Z8350+%40+1.44GHz&id=2837
    # https://ark.intel.com/content/www/us/en/ark/products/93361/intel-atom-x5-z8350-processor-2m-cache-up-to-1-92-ghz.html
    # https://ark.intel.com/content/www/us/en/ark/products/95592/intel-atom-x7-e3950-processor-2m-cache-up-to-1-80-ghz.html
    # https://www.cpubenchmark.net/cpu.php?cpu=Intel+Atom+x7-E3950+%40+1.60GHz
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
        case "Intel Atom x5-Z8350 2 Cores":
            cpu_num_cores = 2
            cpu_frequency_in_hertz = 1.44 * 1e9
            simd_lanes = 4
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.35, 0.65)
            effective_utilization_range_for_inference = (0.65, 0.9)
            memory_size_in_gigabytes = 2
            memory_speed_in_megahertz = 1600
            memory_type = "LPDDR3"
            peak_memory_bandwidth_in_bytes_per_second = 12.8 * 1e9
            mean_power_consumption_data_reception_in_watts = 3.0
            mean_power_consumption_heavy_computational_load_in_watts = 4.0
            mean_power_consumption_data_transmission_in_watts = 3.5
            mean_power_consumption_idle_in_watts = 1.5
            battery_capacity_in_milli_ampere_hours = 2500
            battery_voltage_in_volts = 3.7
            battery_type = "Lithium-Ion"
        case "Intel Atom x5-Z8350 4 Cores":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 1.44 * 1e9
            simd_lanes = 4
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.35, 0.65)
            effective_utilization_range_for_inference = (0.65, 0.9)
            memory_size_in_gigabytes = 4
            memory_speed_in_megahertz = 1600
            memory_type = "LPDDR3"
            peak_memory_bandwidth_in_bytes_per_second = 12.8 * 1e9
            mean_power_consumption_data_reception_in_watts = 4.5
            mean_power_consumption_heavy_computational_load_in_watts = 6.0
            mean_power_consumption_data_transmission_in_watts = 5.0
            mean_power_consumption_idle_in_watts = 2.0
            battery_capacity_in_milli_ampere_hours = 5000
            battery_voltage_in_volts = 5
            battery_type = "Lithium-Ion"
        case "Intel Atom x7-E3950":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 2.0 * 1e9
            simd_lanes = 4
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.4, 0.7)
            effective_utilization_range_for_inference = (0.7, 0.9)
            memory_size_in_gigabytes = 8
            memory_speed_in_megahertz = 1866
            memory_type = "LPDDR4"
            peak_memory_bandwidth_in_bytes_per_second = 29.8 * 1e9
            mean_power_consumption_data_reception_in_watts = 7.0
            mean_power_consumption_heavy_computational_load_in_watts = 10.0
            mean_power_consumption_data_transmission_in_watts = 8.0
            mean_power_consumption_idle_in_watts = 3.0
            battery_capacity_in_milli_ampere_hours = 5000
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


def emulate_qualcomm_snapdragon_performance(device_name: str,
                                            power_source_connection_scenario: bool) -> dict:
    # Qualcomm Snapdragon 410E, Qualcomm Snapdragon 820E
    # https://www.qualcomm.com/products/snapdragon-410e-processor
    # https://www.qualcomm.com/products/snapdragon-820e-embedded-platform
    # https://www.qualcomm.com/products/snapdragon-410e-processor
    # https://www.anandtech.com/show/11111/quick-look-at-the-qualcomm-snapdragon-835
    # https://www.qualcomm.com/products/snapdragon-820e-embedded-platform
    # https://www.anandtech.com/show/10386/snapdragon-820-performance-preview
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
        case "Qualcomm Snapdragon 410E":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 1.2 * 1e9
            simd_lanes = 4
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.35, 0.6)
            effective_utilization_range_for_inference = (0.6, 0.85)
            memory_size_in_gigabytes = 2
            memory_speed_in_megahertz = 533
            memory_type = "LPDDR3"
            peak_memory_bandwidth_in_bytes_per_second = 4.2 * 1e9
            mean_power_consumption_data_reception_in_watts = 2.8
            mean_power_consumption_heavy_computational_load_in_watts = 4.0
            mean_power_consumption_data_transmission_in_watts = 3.2
            mean_power_consumption_idle_in_watts = 1.0
            battery_capacity_in_milli_ampere_hours = 6000
            battery_voltage_in_volts = 3.7
            battery_type = "Lithium-Polymer (Li-Po)"
        case "Qualcomm Snapdragon 820E":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 2.2 * 1e9
            simd_lanes = 4
            fma_units = 2
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.4, 0.75)
            effective_utilization_range_for_inference = (0.75, 0.95)
            memory_size_in_gigabytes = 4
            memory_speed_in_megahertz = 1866
            memory_type = "LPDDR4"
            peak_memory_bandwidth_in_bytes_per_second = 29.8 * 1e9
            mean_power_consumption_data_reception_in_watts = 5.5
            mean_power_consumption_heavy_computational_load_in_watts = 7.0
            mean_power_consumption_data_transmission_in_watts = 6.0
            mean_power_consumption_idle_in_watts = 2.5
            battery_capacity_in_milli_ampere_hours = 6000
            battery_voltage_in_volts = 3.7
            battery_type = "Lithium-Polymer (Li-Po)"
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


def emulate_google_coral_edge_tpu_performance(device_name: str,
                                              power_source_connection_scenario: bool) -> dict:
    # Google Coral Edge TPU Cortex-A53
    # https://coral.ai/products/dev-board/
    # https://coral.ai/docs/edgetpu/benchmarks/
    # https://mlcommons.org/en/inference-edge-10/
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
        case "Google Coral Edge TPU Cortex-A53":
            cpu_num_cores = 4
            cpu_frequency_in_hertz = 1.2 * 1e9
            simd_lanes = 4
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.4, 0.75)
            effective_utilization_range_for_inference = (0.8, 0.99)
            memory_size_in_gigabytes = 4
            memory_speed_in_megahertz = 1600
            memory_type = "LPDDR4"
            peak_memory_bandwidth_in_bytes_per_second = 12.8 * 1e9
            mean_power_consumption_data_reception_in_watts = 4.5
            mean_power_consumption_heavy_computational_load_in_watts = 6.0
            mean_power_consumption_data_transmission_in_watts = 5.0
            mean_power_consumption_idle_in_watts = 2.0
            battery_capacity_in_milli_ampere_hours = 10000
            battery_voltage_in_volts = 3.7
            battery_type = "Lithium-Polymer (Li-Po)"
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


def emulate_beaglebone_black_performance(device_name: str,
                                         power_source_connection_scenario: bool) -> dict:
    # BeagleBone Black Cortex-A8
    # https://beagleboard.org/black
    # https://developer.arm.com/documentation/ddi0344/k/
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
        case "BeagleBone Black Cortex-A8":
            cpu_num_cores = 1
            cpu_frequency_in_hertz = 1.0 * 1e9
            simd_lanes = 2
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.3, 0.5)
            effective_utilization_range_for_inference = (0.5, 0.8)
            memory_size_in_gigabytes = 0.5
            memory_speed_in_megahertz = 800
            memory_type = "DDR3"
            peak_memory_bandwidth_in_bytes_per_second = 6.4 * 1e9
            mean_power_consumption_data_reception_in_watts = 1.5
            mean_power_consumption_heavy_computational_load_in_watts = 2.0
            mean_power_consumption_data_transmission_in_watts = 1.6
            mean_power_consumption_idle_in_watts = 0.8
            battery_capacity_in_milli_ampere_hours = 3000
            battery_voltage_in_volts = 3.7
            battery_type = "Lithium-Polymer (Li-Po)"
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


def emulate_arm_cortex_m_series_performance(device_name: str,
                                            power_source_connection_scenario: bool) -> dict:
    # ARM Cortex-M0, ARM Cortex-M3, ARM Cortex-M4, ARM Cortex-M7
    # https://developer.arm.com/ip-products/processors/cortex-m/cortex-m0
    # https://developer.arm.com/ip-products/processors/cortex-m/cortex-m3
    # https://developer.arm.com/ip-products/processors/cortex-m/cortex-m4
    # https://developer.arm.com/ip-products/processors/cortex-m/cortex-m7
    # https://developer.arm.com/documentation/100706/0100/
    # https://www.eembc.org/benchmark/automotive-sl/
    # https://www.keil.com/pack/doc/CMSIS/DSP/html/index.html
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
        case "ARM Cortex-M0":
            cpu_num_cores = 1
            cpu_frequency_in_hertz = 0.048 * 1e9
            simd_lanes = 1
            fma_units = 0
            fma_per_cycle = 0
            effective_utilization_range_for_training = (0.1, 0.2)
            effective_utilization_range_for_inference = (0.2, 0.3)
            memory_size_in_gigabytes = 0.000032  # 32 KB
            memory_speed_in_megahertz = 48,
            memory_type = "SRAM"
            peak_memory_bandwidth_in_bytes_per_second = 0.1 * 1e9  # Estimated
            mean_power_consumption_data_reception_in_watts = 0.015  # 15 milli-watts (BLE RX)
            mean_power_consumption_heavy_computational_load_in_watts = 0.0001  # 100 microwatts
            mean_power_consumption_data_transmission_in_watts = 0.02  # 20 milli-watts (BLE TX)
            mean_power_consumption_idle_in_watts = 0.00005  # 50 microwatts
            battery_capacity_in_milli_ampere_hours = 100
            battery_voltage_in_volts = 1.8
            battery_type = "Lithium-Polymer (Li-Po) or Coin Cell (Lithium)"
        case "ARM Cortex-M3":
            cpu_num_cores = 1
            cpu_frequency_in_hertz = 0.1 * 1e9
            simd_lanes = 1
            fma_units = 0
            fma_per_cycle = 0
            effective_utilization_range_for_training = (0.1, 0.2)
            effective_utilization_range_for_inference = (0.2, 0.3)
            memory_size_in_gigabytes = 0.000064  # 64 KB
            memory_speed_in_megahertz = 72
            memory_type = "SRAM"
            peak_memory_bandwidth_in_bytes_per_second = 0.2 * 1e9
            mean_power_consumption_data_reception_in_watts =  0.02
            mean_power_consumption_heavy_computational_load_in_watts = 0.0002
            mean_power_consumption_data_transmission_in_watts = 0.025
            mean_power_consumption_idle_in_watts = 0.0001
            battery_capacity_in_milli_ampere_hours = 200
            battery_voltage_in_volts = 2.0
            battery_type = "Lithium-Polymer (Li-Po) or Coin Cell (Lithium)"
        case "ARM Cortex-M4":
            cpu_num_cores = 1
            cpu_frequency_in_hertz = 0.15 * 1e9
            simd_lanes = 1
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.2, 0.4)
            effective_utilization_range_for_inference = (0.4, 0.6)
            memory_size_in_gigabytes = 0.000128  # 128 KB
            memory_speed_in_megahertz = 120
            memory_type = "SRAM"
            peak_memory_bandwidth_in_bytes_per_second = 0.5 * 1e9
            mean_power_consumption_data_reception_in_watts = 0.025
            mean_power_consumption_heavy_computational_load_in_watts = 0.0002
            mean_power_consumption_data_transmission_in_watts = 0.03
            mean_power_consumption_idle_in_watts = 0.0001
            battery_capacity_in_milli_ampere_hours = 300
            battery_voltage_in_volts = 1.8
            battery_type = "Lithium-Polymer (Li-Po) or Coin Cell (Lithium)"
        case "ARM Cortex-M7":
            cpu_num_cores = 1
            cpu_frequency_in_hertz = 0.4 * 1e9
            simd_lanes = 1
            fma_units = 1
            fma_per_cycle = 1
            effective_utilization_range_for_training = (0.2, 0.4)
            effective_utilization_range_for_inference = (0.4, 0.6)
            memory_size_in_gigabytes = 0.000256  # 256 KB
            memory_speed_in_megahertz = 216
            memory_type = "SRAM"
            peak_memory_bandwidth_in_bytes_per_second = 1.0 * 1e9
            mean_power_consumption_data_reception_in_watts = 0.03
            mean_power_consumption_heavy_computational_load_in_watts = 0.0004
            mean_power_consumption_data_transmission_in_watts = 0.035
            mean_power_consumption_idle_in_watts = 0.0002
            battery_capacity_in_milli_ampere_hours = 500
            battery_voltage_in_volts = 1.8
            battery_type = "Lithium-Polymer (Li-Po)"
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


def emulate_microchip_sam_microcontrollers_performance(device_name: str,
                                                       power_source_connection_scenario: bool) -> dict:
    # Microchip SAM D, Microchip SAM E
    # https://www.microchip.com/en-us/product/ATSAMD21G18A
    # https://www.microchip.com/en-us/product/SAME54P20A
    # https://www.microchip.com/en-us/product-family/32-bit-mcus/sam-d
    # https://www.eembc.org/coremark/
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
        case "Microchip SAM D":
            cpu_num_cores = 1
            cpu_frequency_in_hertz = 0.048 * 1e9
            simd_lanes = 1
            fma_units = 0
            fma_per_cycle = 0
            effective_utilization_range_for_training = (0.1, 0.2)
            effective_utilization_range_for_inference = (0.2, 0.3)
            memory_size_in_gigabytes = 0.000032  # 32 KB
            memory_speed_in_megahertz = 48
            memory_type = "SRAM"
            peak_memory_bandwidth_in_bytes_per_second = 0.1 * 1e9
            mean_power_consumption_data_reception_in_watts = 0.015
            mean_power_consumption_heavy_computational_load_in_watts = 0.0002
            mean_power_consumption_data_transmission_in_watts = 0.02
            mean_power_consumption_idle_in_watts = 0.0001
            battery_capacity_in_milli_ampere_hours = 300
            battery_voltage_in_volts = 3.7
            battery_type = "Lithium-Polymer (Li-Po) or Coin Cell (Lithium)"
        case "Microchip SAM E":
            cpu_num_cores = 1
            cpu_frequency_in_hertz = 0.12 * 1e9
            simd_lanes = 1
            fma_units = 0
            fma_per_cycle = 0
            effective_utilization_range_for_training = (0.1, 0.2)
            effective_utilization_range_for_inference = (0.2, 0.3)
            memory_size_in_gigabytes = 0.000256  # 256 KB
            memory_speed_in_megahertz = 300
            memory_type = "SRAM"
            peak_memory_bandwidth_in_bytes_per_second = 1.5 * 1e9
            mean_power_consumption_data_reception_in_watts = 0.025
            mean_power_consumption_heavy_computational_load_in_watts = 0.0003
            mean_power_consumption_data_transmission_in_watts = 0.03
            mean_power_consumption_idle_in_watts = 0.0002
            battery_capacity_in_milli_ampere_hours = 300
            battery_voltage_in_volts = 3.7
            battery_type = "Lithium-Polymer (Li-Po)"
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


def generate_edge_devices(power_source_connection_scenarios: list) -> dict:
    edge_devices = {}
    for power_source_connection_scenario in power_source_connection_scenarios:
        edge_devices.update({"raspberry_pi_3_device_{0}".format(power_source_connection_scenario): emulate_raspberry_pi_performance("Raspberry Pi 3", power_source_connection_scenario),
                             "raspberry_pi_4_device_{0}".format(power_source_connection_scenario): emulate_raspberry_pi_performance("Raspberry Pi 4", power_source_connection_scenario),
                             "nvidia_jetson_tx1_device_{0}".format(power_source_connection_scenario): emulate_nvidia_jetson_performance("NVIDIA Jetson TX1", power_source_connection_scenario),
                             "nvidia_jetson_tx2_device_{0}".format(power_source_connection_scenario): emulate_nvidia_jetson_performance("NVIDIA Jetson TX2", power_source_connection_scenario),
                             "nvidia_jetson_nano_device_{0}".format(power_source_connection_scenario): emulate_nvidia_jetson_performance("NVIDIA Jetson Nano", power_source_connection_scenario),
                             "nvidia_jetson_xavier_device_{0}".format(power_source_connection_scenario): emulate_nvidia_jetson_performance("NVIDIA Jetson Xavier", power_source_connection_scenario),
                             "intel_atom_x5_z8350_2c_device_{0}".format(power_source_connection_scenario): emulate_intel_atom_performance("Intel Atom x5-Z8350 2 Cores", power_source_connection_scenario),
                             "intel_atom_x5_z8350_4c_device_{0}".format(power_source_connection_scenario): emulate_intel_atom_performance("Intel Atom x5-Z8350 4 Cores", power_source_connection_scenario),
                             "intel_atom_x7_e3950_device_{0}".format(power_source_connection_scenario): emulate_intel_atom_performance("Intel Atom x7-E3950", power_source_connection_scenario),
                             "qualcomm_snapdragon_410e_device_{0}".format(power_source_connection_scenario): emulate_qualcomm_snapdragon_performance("Qualcomm Snapdragon 410E", power_source_connection_scenario),
                             "qualcomm_snapdragon_820e_device_{0}".format(power_source_connection_scenario): emulate_qualcomm_snapdragon_performance("Qualcomm Snapdragon 820E", power_source_connection_scenario),
                             "google_coral_edge_tpu_cortex_a53_device_{0}".format(power_source_connection_scenario): emulate_google_coral_edge_tpu_performance("Google Coral Edge TPU Cortex-A53", power_source_connection_scenario),
                             "beaglebone_black_cortex_a8_device_{0}".format(power_source_connection_scenario): emulate_beaglebone_black_performance("BeagleBone Black Cortex-A8", power_source_connection_scenario),
                             "arm_cortex_m0_device_{0}".format(power_source_connection_scenario): emulate_arm_cortex_m_series_performance("ARM Cortex-M0", power_source_connection_scenario),
                             "arm_cortex_m3_device_{0}".format(power_source_connection_scenario): emulate_arm_cortex_m_series_performance("ARM Cortex-M3", power_source_connection_scenario),
                             "arm_cortex_m4_device_{0}".format(power_source_connection_scenario): emulate_arm_cortex_m_series_performance("ARM Cortex-M4", power_source_connection_scenario),
                             "arm_cortex_m7_device_{0}".format(power_source_connection_scenario): emulate_arm_cortex_m_series_performance("ARM Cortex-M7", power_source_connection_scenario),
                             "microchip_sam_d_microcontroller_device_{0}".format(power_source_connection_scenario): emulate_microchip_sam_microcontrollers_performance("Microchip SAM D", power_source_connection_scenario),
                             "microchip_sam_e_microcontroller_device_{0}".format(power_source_connection_scenario): emulate_microchip_sam_microcontrollers_performance("Microchip SAM E", power_source_connection_scenario)})
    return edge_devices


def generate_edge_devices_configuration_file(output_file: Path,
                                             power_source_connection_scenarios: list) -> None:
    # Create the parents directories of the output file (if not exist yet).
    output_file.parent.mkdir(exist_ok=True, parents=True)
    # Emulate edge devices performance.
    edge_devices = generate_edge_devices(power_source_connection_scenarios)
    with open(file=output_file, mode="a", encoding="utf-8") as file:
        # Get the dictionary of edge devices settings from the first device.
        _, edge_device_settings_dict = next(iter(edge_devices.items()))
        # Set and write the header line.
        header_line = ",".join(list(edge_device_settings_dict.keys()))
        file.write(header_line + "\n")
        # Set and write the data lines.
        data_lines = []
        for _, edge_device_settings_dict in edge_devices.items():
            data_line = ",".join(str(value) for value in edge_device_settings_dict.values())
            data_lines.append(data_line)
        file.writelines("\n".join(str(data_line) for data_line in data_lines))
        file.write("\n")
