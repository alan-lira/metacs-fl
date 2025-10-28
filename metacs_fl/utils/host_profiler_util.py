from copy import deepcopy
from json import dump
from numpy import dot, mean, random
from os import environ
from pathlib import Path
from psutil import cpu_count, cpu_freq, cpu_percent
from socket import gethostname
from subprocess import CalledProcessError, check_output, STDOUT
from time import time, sleep


class HostProfiler:

    def __init__(self,
                 output_dir: Path) -> None:
        self.output_dir = Path(output_dir).joinpath("host_profile")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hostname = gethostname().lower()

    @staticmethod
    def _run_command(command_text: str) -> str:
        try:
            return check_output(args=command_text, shell=True, text=True).strip()
        except CalledProcessError:
            return ""

    @staticmethod
    def _measure_gflops() -> float:
        environ["OPENBLAS_NUM_THREADS"] = str(cpu_count(logical=False))
        n = 2000
        a = random.randn(n, n)
        b = random.randn(n, n)
        t0 = time()
        dot(a, b)
        t = time() - t0
        return round((2.0 * n ** 3) / (t * 1e9), 2)

    @staticmethod
    def _measure_memory_bandwidth() -> float:
        memory_bandwidth_in_gbs = 5.0  # Conservative fallback: assume ~5 GB/s — baseline for older DDR3/DDR4 systems.
        try:
            result = check_output(["stress-ng", "--stream", "4", "--timeout", "10", "--metrics-brief"],
                                  stderr=STDOUT,
                                  text=True)
            bw_values = [float(x) for x in result.split() if x.replace(".", "", 1).isdigit()]
            return round(bw_values[-1], 2) if bw_values else memory_bandwidth_in_gbs
        except FileNotFoundError:
            return memory_bandwidth_in_gbs

    @staticmethod
    def _measure_disk_speed() -> float:
        disk_speed_in_mbs = 100.0  # Conservative fallback: assume 100 MB/s — roughly HDD baseline.
        try:
            result = check_output(args=["fio", "--name=write_test", "--filename=/tmp/testfile", "--size=512M", "--bs=1M",
                                        "--rw=write", "--direct=1", "--numjobs=1", "--time_based", "--runtime=5", "--group_reporting"],
                                  text=True,
                                  stderr=STDOUT)
            for line in result.splitlines():
                if "WRITE:" in line and "MiB/s" in line:
                    val = line.split("MiB/s")[-2].split()[-1]
                    return float(val)
            return disk_speed_in_mbs
        except FileNotFoundError:
            return disk_speed_in_mbs

    @staticmethod
    def _measure_power() -> dict:
        try:
            import pyRAPL
        except ImportError:
            pyRAPL = None
        if pyRAPL:
            try:
                pyRAPL.setup()
                m = pyRAPL.Measurement("pkg")
                m.begin()
                sleep(3)
                m.end()
                return {d.domain.name: d.energy for d in m.result}
            except Exception:
                pass
        p0 = cpu_percent(interval=None)
        sleep(3)
        p1 = cpu_percent(interval=None)
        avg_util = (p0 + p1) / 2
        est_pkg_joules = avg_util * cpu_count() * 0.3
        return {"PKG": round(est_pkg_joules, 2), "DRAM": 0}

    def _profile_host(self) -> dict:
        cpu_info = {"model": self._run_command("lscpu | grep 'Model name' | head -n1 | cut -d: -f2- | xargs"),
                    "sockets": int(self._run_command("lscpu | grep 'Socket(s)' | awk '{print $2}'") or 1),
                    "cores_per_socket": int(self._run_command("lscpu | grep 'Core(s) per socket' | awk '{print $4}'") or cpu_count(logical=False)),
                    "threads": cpu_count(logical=True),
                    "freq_mhz": round(cpu_freq().max, 0) if cpu_freq() else 0}
        compute_profile = {"sustained_gflops": self._measure_gflops()}
        memory_profile = {"bandwidth_gbs": self._measure_memory_bandwidth()}
        storage_profile = {"seq_write_mbs": self._measure_disk_speed()}
        power_profile = self._measure_power()
        host_profile = {"node": self.hostname,
                        "cpu": cpu_info,
                        "compute": compute_profile,
                        "memory": memory_profile,
                        "storage": storage_profile,
                        "power": power_profile}
        return host_profile

    def _recursive_average(self,
                           dicts: list) -> dict:
        result = deepcopy(dicts[0])
        for key, value in result.items():
            if isinstance(value, dict):
                nested_dicts = [d[key] for d in dicts if key in d]
                result[key] = self._recursive_average(nested_dicts)
            elif isinstance(value, (int, float)):
                numeric_values = [d[key] for d in dicts if isinstance(d.get(key, None), (int, float))]
                if numeric_values:
                    result[key] = float(mean(numeric_values))
        return result

    def _round_floats(self,
                      obj: dict | list | float,
                      decimals: int = 2) -> dict | list | float:
        if isinstance(obj, dict):
            return {k: self._round_floats(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_floats(v, decimals) for v in obj]
        elif isinstance(obj, float):
            return round(obj, decimals)
        else:
            return obj

    def _save_host_profile_result(self,
                                  host_profile: dict,
                                  client_id: int) -> None:
        host_profile_result_file = self.output_dir.joinpath("{0}_client_{1}.json".format(self.hostname, client_id))
        with open(file=host_profile_result_file, mode="w") as o_f:
            dump(host_profile, o_f, indent=2)
        print("Output file: {0}".format(host_profile_result_file))

    def profile_host_n_times(self,
                             client_id: int,
                             n: int = 10) -> dict:
        print("Profiling the host '{0}' for client '{1}' ({2} {3})..."
              .format(self.hostname, client_id, n, "times" if n != 1 else "time"))
        profiles = []
        for i in range(n):
            print("- Measurement {0}/{1} for client '{2}'...".format(i + 1, n, client_id))
            profile = self._profile_host()
            profiles.append(profile)
        host_profile_avg = self._recursive_average(profiles)
        host_profile_avg = self._round_floats(host_profile_avg)
        self._save_host_profile_result(host_profile_avg, client_id)
        return host_profile_avg
