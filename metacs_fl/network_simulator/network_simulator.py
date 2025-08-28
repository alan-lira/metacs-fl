from collections import deque
from numpy import random


class NetworkSimulator:

    def __init__(self,
                 upload_bandwidth_mean: float,
                 upload_bandwidth_std: float,
                 upload_bandwidth_min: float,
                 download_bandwidth_mean: float,
                 download_bandwidth_std: float,
                 download_bandwidth_min: float,
                 base_latency: float,
                 latency_jitter: float,
                 packet_loss_rate: float,
                 phi: list) -> None:
        # Initialize the bandwidth stats.
        self._upload_bandwidth_mean = upload_bandwidth_mean
        self._upload_bandwidth_std = upload_bandwidth_std
        self._upload_bandwidth_min = upload_bandwidth_min
        self._download_bandwidth_mean = download_bandwidth_mean
        self._download_bandwidth_std = download_bandwidth_std
        self._download_bandwidth_min = download_bandwidth_min
        # Initialize the latency (base and jitter) and packet loss rate.
        self._base_latency = base_latency
        self._latency_jitter = latency_jitter
        self._packet_loss_rate = packet_loss_rate
        # Initialize AR (autoregressive process) coefficients.
        self._phi = phi
        self._p = len(phi)  # Order of AR(p)
        # Initialize history deque for upload/download with p elements.
        initial_upload = random.normal(upload_bandwidth_mean, upload_bandwidth_std)
        initial_download = random.normal(download_bandwidth_mean, download_bandwidth_std)
        self._upload_history = deque([initial_upload] * self._p, maxlen=self._p)
        self._download_history = deque([initial_download] * self._p, maxlen=self._p)

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def simulate_latency_change(self) -> float:
        # Get the necessary attributes.
        base_latency = self.get_attribute("_base_latency")
        latency_jitter = self.get_attribute("_latency_jitter")
        jitter = random.uniform(-latency_jitter, latency_jitter)
        current_latency = max(0, base_latency + jitter)
        # Return the current latency performance.
        return current_latency

    def simulate_packet_loss_event(self) -> bool:
        # Get the necessary attributes.
        packet_loss_rate = self.get_attribute("_packet_loss_rate")
        packet_loss_event = random.random() < packet_loss_rate
        return packet_loss_event

    def _calculate_ar_step(self,
                           history: deque,
                           mean: float,
                           std: float) -> float:
        # Get the necessary attributes.
        phi = self.get_attribute("_phi")
        # Compute AR(p) step.
        ar_sum = sum(phi_i * (history[-(i+1)] - mean) for i, phi_i in enumerate(phi))
        noise = random.normal(0, std)
        ar_step = mean + ar_sum + noise
        return ar_step

    def simulate_network_performance_change(self) -> tuple:
        # Get the necessary attributes.
        upload_bandwidth_mean = self.get_attribute("_upload_bandwidth_mean")
        upload_bandwidth_std = self.get_attribute("_upload_bandwidth_std")
        upload_bandwidth_min = self.get_attribute("_upload_bandwidth_min")
        download_bandwidth_mean = self.get_attribute("_download_bandwidth_mean")
        download_bandwidth_std = self.get_attribute("_download_bandwidth_std")
        download_bandwidth_min = self.get_attribute("_download_bandwidth_min")
        # Get current histories.
        upload_history = self.get_attribute("_upload_history")
        download_history = self.get_attribute("_download_history")
        # Calculate AR(p) steps.
        ar_step_upload = self._calculate_ar_step(upload_history, upload_bandwidth_mean, upload_bandwidth_std)
        ar_step_download = self._calculate_ar_step(download_history, download_bandwidth_mean, download_bandwidth_std)
        # Apply minimum bounds.
        current_upload_bandwidth = max(ar_step_upload, upload_bandwidth_min)
        current_download_bandwidth = max(ar_step_download, download_bandwidth_min)
        # Update histories.
        upload_history.append(current_upload_bandwidth)
        download_history.append(current_download_bandwidth)
        self._set_attribute("_upload_history", upload_history)
        self._set_attribute("_download_history", download_history)
        # Return the current upload and download bandwidth performances.
        return current_upload_bandwidth, current_download_bandwidth
