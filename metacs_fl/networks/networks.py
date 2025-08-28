from pathlib import Path


def calculate_maximum_segment_size_in_bytes(mtu_in_bytes: int,
                                            protocol: str) -> float:
    mss_in_bytes = 0
    match protocol:
        case "IPv4":
            ipv4_header = 40  # IPv4 header is 40 bytes (20 IP + 20 TCP).
            mss_in_bytes = mtu_in_bytes - ipv4_header
        case "IPv6":
            ipv6_header = 60  # IPv6 header is 60 bytes (40 IP + 20 TCP).
            mss_in_bytes = mtu_in_bytes - ipv6_header
    return mss_in_bytes


def generate_network_dict_based_on_quality_scenario(base_upload_bandwidth: float,
                                                    base_download_bandwidth: float,
                                                    base_latency: float,
                                                    network_quality_scenario: str) -> dict:
    network_dict = {}
    match network_quality_scenario:
        case "excellent":
            network_dict = {"upload_bandwidth_mean": round(base_upload_bandwidth * 0.9, 2),
                            "upload_bandwidth_std": round(base_upload_bandwidth * 0.05, 2),
                            "upload_bandwidth_min": round(base_upload_bandwidth * 0.7, 2),
                            "download_bandwidth_mean": round(base_download_bandwidth * 0.9, 2),
                            "download_bandwidth_std": round(base_download_bandwidth * 0.05, 2),
                            "download_bandwidth_min": round(base_download_bandwidth * 0.7, 2),
                            "base_latency": round(base_latency * 0.8, 2),
                            "latency_jitter": round(base_latency * 0.1, 2),
                            "packet_loss_rate": 0.001}
        case "moderate":
            network_dict = {"upload_bandwidth_mean": round(base_upload_bandwidth * 0.6, 2),
                            "upload_bandwidth_std": round(base_upload_bandwidth * 0.1, 2),
                            "upload_bandwidth_min": round(base_upload_bandwidth * 0.4, 2),
                            "download_bandwidth_mean": round(base_download_bandwidth * 0.6, 2),
                            "download_bandwidth_std": round(base_download_bandwidth * 0.1, 2),
                            "download_bandwidth_min": round(base_download_bandwidth * 0.4, 2),
                            "base_latency": round(base_latency * 1.0, 2),
                            "latency_jitter": round(base_latency * 0.2, 2),
                            "packet_loss_rate": 0.01}
        case "poor":
            network_dict = {"upload_bandwidth_mean": round(base_upload_bandwidth * 0.3, 2),
                            "upload_bandwidth_std": round(base_upload_bandwidth * 0.15, 2),
                            "upload_bandwidth_min": round(base_upload_bandwidth * 0.1, 2),
                            "download_bandwidth_mean": round(base_download_bandwidth * 0.3, 2),
                            "download_bandwidth_std": round(base_download_bandwidth * 0.15, 2),
                            "download_bandwidth_min": round(base_download_bandwidth * 0.1, 2),
                            "base_latency": round(base_latency * 1.5, 2),
                            "latency_jitter": round(base_latency * 0.3, 2),
                            "packet_loss_rate": 0.05}
    return network_dict


def emulate_3g_network(network_quality_scenario: str) -> dict:
    # 3G Network.
    # https://pratexo.com/wp-content/uploads/2024/02/Common-Mistakes-in-Edge-Computing-White-Paper.pdf
    # https://www.researchgate.net/publication/221244486_An_Empirical_Study_on_3G_Network_Capacity_and_Performance
    # https://www.diva-portal.org/smash/get/diva2%3A829687/FULLTEXT01.pdf
    network_name = "3G"
    base_upload_bandwidth = 0.5
    upload_bandwidth_unit = "Mbps"
    base_download_bandwidth = 2.0
    download_bandwidth_unit = "Mbps"
    base_latency = 100
    base_latency_unit = "ms"
    mtu_in_bytes = 1500  # Maximum transmission unit in bytes.
    mss_ipv4_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv4")
    mss_ipv6_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv6")
    network_dict = generate_network_dict_based_on_quality_scenario(base_upload_bandwidth,
                                                                   base_download_bandwidth,
                                                                   base_latency,
                                                                   network_quality_scenario)
    network_dict.update({"network_name": network_name,
                         "network_quality_scenario": network_quality_scenario,
                         "upload_bandwidth_unit": upload_bandwidth_unit,
                         "download_bandwidth_unit": download_bandwidth_unit,
                         "base_latency_unit": base_latency_unit,
                         "mss_ipv4_in_bytes": mss_ipv4_in_bytes,
                         "mss_ipv6_in_bytes": mss_ipv6_in_bytes})
    return network_dict


def emulate_4g_lte_network(network_quality_scenario: str) -> dict:
    # 4G LTE Network.
    # https://www.amplicon.com/actions/viewDoc.cfm?doc=AnyG_to_4G_Whitepaper.pdf
    # https://www.researchgate.net/publication/340126116_Performance_Evaluation_of_LTE_Networks
    # https://onlinelibrary.wiley.com/doi/10.1155/2023/6205689
    network_name = "4G LTE"
    base_upload_bandwidth = 10
    upload_bandwidth_unit = "Mbps"
    base_download_bandwidth = 50
    download_bandwidth_unit = "Mbps"
    base_latency = 50
    base_latency_unit = "ms"
    mtu_in_bytes = 1500  # Maximum transmission unit in bytes.
    mss_ipv4_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv4")
    mss_ipv6_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv6")
    network_dict = generate_network_dict_based_on_quality_scenario(base_upload_bandwidth,
                                                                   base_download_bandwidth,
                                                                   base_latency,
                                                                   network_quality_scenario)
    network_dict.update({"network_name": network_name,
                         "network_quality_scenario": network_quality_scenario,
                         "upload_bandwidth_unit": upload_bandwidth_unit,
                         "download_bandwidth_unit": download_bandwidth_unit,
                         "base_latency_unit": base_latency_unit,
                         "mss_ipv4_in_bytes": mss_ipv4_in_bytes,
                         "mss_ipv6_in_bytes": mss_ipv6_in_bytes})
    return network_dict


def emulate_5g_network(network_quality_scenario: str) -> dict:
    # 5G Network.
    # https://www.cisco.com/c/en/us/solutions/collateral/executive-perspectives/annual-internet-report/white-paper-c11-741490.html
    # https://patentpc.com/blog/5g-network-speeds-vs-4g-performance-stats
    # https://www.researchgate.net/publication/387000586_ASSESSING_LATENCY_AND_THROUGHPUT_IN_5G_NETWORKS_FOR_DATA_COMMUNICATION
    network_name = "5G"
    base_upload_bandwidth = 100
    upload_bandwidth_unit = "Mbps"
    base_download_bandwidth = 575
    download_bandwidth_unit = "Mbps"
    base_latency = 10
    base_latency_unit = "ms"
    mtu_in_bytes = 1500  # Maximum transmission unit in bytes.
    mss_ipv4_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv4")
    mss_ipv6_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv6")
    network_dict = generate_network_dict_based_on_quality_scenario(base_upload_bandwidth,
                                                                   base_download_bandwidth,
                                                                   base_latency,
                                                                   network_quality_scenario)
    network_dict.update({"network_name": network_name,
                         "network_quality_scenario": network_quality_scenario,
                         "upload_bandwidth_unit": upload_bandwidth_unit,
                         "download_bandwidth_unit": download_bandwidth_unit,
                         "base_latency_unit": base_latency_unit,
                         "mss_ipv4_in_bytes": mss_ipv4_in_bytes,
                         "mss_ipv6_in_bytes": mss_ipv6_in_bytes})
    return network_dict


def emulate_wifi_network(network_quality_scenario: str) -> dict:
    # Wi-Fi Network.
    # https://www.cisco.com/c/en/us/solutions/collateral/executive-perspectives/annual-internet-report/white-paper-c11-741490.html
    # https://www.netally.com/tech-tips/how-to-check-wifi-quality
    # https://www.fusionconnect.com/speed-test-plus/internet-quality-test
    network_name = "Wi-Fi"
    base_upload_bandwidth = 30
    upload_bandwidth_unit = "Mbps"
    base_download_bandwidth = 92
    download_bandwidth_unit = "Mbps"
    base_latency = 20
    base_latency_unit = "ms"
    mtu_in_bytes = 1500  # Maximum transmission unit in bytes.
    mss_ipv4_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv4")
    mss_ipv6_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv6")
    network_dict = generate_network_dict_based_on_quality_scenario(base_upload_bandwidth,
                                                                   base_download_bandwidth,
                                                                   base_latency,
                                                                   network_quality_scenario)
    network_dict.update({"network_name": network_name,
                         "network_quality_scenario": network_quality_scenario,
                         "upload_bandwidth_unit": upload_bandwidth_unit,
                         "download_bandwidth_unit": download_bandwidth_unit,
                         "base_latency_unit": base_latency_unit,
                         "mss_ipv4_in_bytes": mss_ipv4_in_bytes,
                         "mss_ipv6_in_bytes": mss_ipv6_in_bytes})
    return network_dict


def emulate_fixed_broadband_network(network_quality_scenario: str) -> dict:
    # Fixed Broadband Network.
    # https://www.cisco.com/c/en/us/solutions/collateral/executive-perspectives/annual-internet-report/white-paper-c11-741490.html
    # https://www.fcc.gov/reports-research/reports/measuring-broadband-america/measuring-fixed-broadband-thirteenth-report
    # https://www.fcc.gov/reports-research/reports/measuring-broadband-america/charts-measuring-fixed-broadband-thirteenth
    network_name = "Fixed Broadband"
    base_upload_bandwidth = 20
    upload_bandwidth_unit = "Mbps"
    base_download_bandwidth = 110
    download_bandwidth_unit = "Mbps"
    base_latency = 20
    base_latency_unit = "ms"
    mtu_in_bytes = 1500  # Maximum transmission unit in bytes.
    mss_ipv4_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv4")
    mss_ipv6_in_bytes = calculate_maximum_segment_size_in_bytes(mtu_in_bytes, "IPv6")
    network_dict = generate_network_dict_based_on_quality_scenario(base_upload_bandwidth,
                                                                   base_download_bandwidth,
                                                                   base_latency,
                                                                   network_quality_scenario)
    network_dict.update({"network_name": network_name,
                         "network_quality_scenario": network_quality_scenario,
                         "upload_bandwidth_unit": upload_bandwidth_unit,
                         "download_bandwidth_unit": download_bandwidth_unit,
                         "base_latency_unit": base_latency_unit,
                         "mss_ipv4_in_bytes": mss_ipv4_in_bytes,
                         "mss_ipv6_in_bytes": mss_ipv6_in_bytes})
    return network_dict


def generate_networks(network_quality_scenarios: list) -> dict:
    networks = {}
    for network_quality_scenario in network_quality_scenarios:
        networks.update({"3g_network_{0}".format(network_quality_scenario): emulate_3g_network(network_quality_scenario),
                         "4g_lte_network_{0}".format(network_quality_scenario): emulate_4g_lte_network(network_quality_scenario),
                         "5g_network_{0}".format(network_quality_scenario): emulate_5g_network(network_quality_scenario),
                         "wifi_network_{0}".format(network_quality_scenario): emulate_wifi_network(network_quality_scenario),
                         "fixed_broadband_network_{0}".format(network_quality_scenario): emulate_fixed_broadband_network(network_quality_scenario)})
    return networks


def generate_networks_configuration_file(output_file: Path,
                                         network_quality_scenarios: list) -> None:
    # Create the parents directories of the output file (if not exist yet).
    output_file.parent.mkdir(exist_ok=True, parents=True)
    # Emulate networks performance.
    networks = generate_networks(network_quality_scenarios)
    with open(file=output_file, mode="a", encoding="utf-8") as file:
        # Get the dictionary of network settings from the first network.
        _, network_settings_dict = next(iter(networks.items()))
        # Set and write the header line.
        header_line = ",".join(list(network_settings_dict.keys()))
        file.write(header_line + "\n")
        # Set and write the data lines.
        data_lines = []
        for _, network_settings_dict in networks.items():
            data_line = ",".join(str(value) for value in network_settings_dict.values())
            data_lines.append(data_line)
        file.writelines("\n".join(str(data_line) for data_line in data_lines))
        file.write("\n")
