from threading import Lock
from time import sleep, time

from metacs_fl.utils.system_modeler_util import get_cpu_cores_frequencies_via_sysfs, set_cpu_cores_frequencies, \
    restore_original_cpu_cores_frequencies, get_cpu_governors, set_cpu_governors, restore_original_cpu_cores_governors


class CPUCoresAllocator:

    def __init__(self) -> None:
        self._all_cpu_cores_available = []
        self._cpu_cores_acquired_map = {}
        self._next_core_index = 0
        self._original_cpu_cores_frequencies = {}
        self._original_cpu_governors = {}
        self._lock = Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = Lock()

    def initialize(self,
                   all_cpu_cores_available: list,
                   frequencies_unit: str = "MHz",
                   cpu_governor: str = "performance") -> None:
        self._all_cpu_cores_available = sorted(all_cpu_cores_available)
        self._original_cpu_cores_frequencies = get_cpu_cores_frequencies_via_sysfs(all_cpu_cores_available, frequencies_unit)
        self._original_cpu_governors = get_cpu_governors(all_cpu_cores_available)
        # Set the CPU cores governors.
        set_cpu_governors(all_cpu_cores_available, cpu_governor)

    def acquire_cpu_cores(self,
                          actor_id: str,
                          num_cpu_cores_to_acquire: int,
                          cpu_frequency_min: float = None,
                          cpu_frequency_max: float = None,
                          cpu_frequency_unit: str = "MHz",
                          lock_timeout: float = None) -> list:
        start_time = time()
        actor_acquired_cpu_cores = []
        while True:
            with self._lock:
                acquired = False
                cpu_cores_in_use = {core for cores in self._cpu_cores_acquired_map.values() for core in cores}
                cpu_cores_available = [core for core in self._all_cpu_cores_available if core not in cpu_cores_in_use]
                if len(cpu_cores_available) >= num_cpu_cores_to_acquire:
                    # Round-robin CPU cores allocation.
                    count = 0
                    i = self._next_core_index
                    while count < num_cpu_cores_to_acquire:
                        core = self._all_cpu_cores_available[i % len(self._all_cpu_cores_available)]
                        if core not in cpu_cores_in_use and core not in actor_acquired_cpu_cores:
                            actor_acquired_cpu_cores.append(core)
                            count += 1
                        i += 1
                    self._next_core_index = i % len(self._all_cpu_cores_available)
                    self._cpu_cores_acquired_map[actor_id] = actor_acquired_cpu_cores
                    acquired = True
            if acquired:
                # Set the CPU cores frequencies, if requested.
                if cpu_frequency_min is not None and cpu_frequency_max is not None:
                    set_cpu_cores_frequencies(actor_acquired_cpu_cores,
                                              cpu_frequency_min,
                                              cpu_frequency_max,
                                              cpu_frequency_unit)
                return actor_acquired_cpu_cores
            if lock_timeout is not None and (time() - start_time) >= lock_timeout:
                raise TimeoutError("Failed to acquire CPU cores within the timeout.")
            sleep(0.1)

    def release_cpu_cores(self,
                          actor_id: str) -> None:
        with self._lock:
            if actor_id in self._cpu_cores_acquired_map:
                actor_acquired_cpu_cores = self._cpu_cores_acquired_map[actor_id]
                # Restore the CPU cores frequencies to their original values.
                restore_original_cpu_cores_frequencies(actor_acquired_cpu_cores, self._original_cpu_cores_frequencies)
                # Restore the CPU cores governors to their original values.
                restore_original_cpu_cores_governors(actor_acquired_cpu_cores, self._original_cpu_governors)
                del self._cpu_cores_acquired_map[actor_id]
