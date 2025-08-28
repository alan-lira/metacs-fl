from asyncio import sleep
from ray import get_actor, remote
from ray.exceptions import RayActorError

from metacs_fl.flower_simulator.lock_actor import LockActor
from metacs_fl.utils.system_modeler_util import get_cpu_cores_frequencies_via_sysfs, set_cpu_cores_frequencies, \
    restore_original_cpu_cores_frequencies, get_cpu_governors, set_cpu_governors, restore_original_cpu_cores_governors


@remote
class CPUCoresAllocatorActor:

    def __init__(self) -> None:
        self._all_cpu_cores_available = []
        self._cpu_cores_acquired_map = {}
        self._next_core_index = 0
        self._original_cpu_cores_frequencies = {}
        self._original_cpu_governors = {}
        self._lock_actor = None

    async def initialize(self,
                         all_cpu_cores_available: list,
                         frequencies_unit: str = "MHz",
                         cpu_governor: str = "performance",
                         use_lock_actor: bool = True,
                         lock_actor_namespace: str = None,
                         lock_actor_name: str = None,
                         lock_actor_lifetime: str = None,
                         lock_actor_scheduling_strategy: any = None) -> None:
        self._all_cpu_cores_available = sorted(all_cpu_cores_available)
        self._original_cpu_cores_frequencies = get_cpu_cores_frequencies_via_sysfs(all_cpu_cores_available,
                                                                                   frequencies_unit)
        self._original_cpu_governors = get_cpu_governors(all_cpu_cores_available)
        # Set the CPU cores governors.
        set_cpu_governors(all_cpu_cores_available, cpu_governor)
        # Get or create the LockActor, if requested.
        if use_lock_actor:
            lock_actor_name = "lock_{0}".format(lock_actor_name)
            try:
                self._lock_actor = get_actor(name=lock_actor_name, namespace=lock_actor_namespace)
            except (ValueError, RayActorError):
                try:
                    self._lock_actor = LockActor.options(name=lock_actor_name,
                                                         namespace=lock_actor_namespace,
                                                         lifetime=lock_actor_lifetime,
                                                         scheduling_strategy=lock_actor_scheduling_strategy).remote()
                except ValueError:
                    self._lock_actor = get_actor(name=lock_actor_name, namespace=lock_actor_namespace)

    async def _acquire_lock(self,
                            timeout: float = None) -> None:
        if self._lock_actor:
            while True:
                if timeout is None:
                    lock_acquired = await self._lock_actor.acquire.remote()
                else:
                    lock_acquired = await self._lock_actor.acquire.remote(timeout=timeout)
                if lock_acquired:
                    break
                print("Waiting to acquire lock...")
                await sleep(0.1)

    async def _release_lock(self) -> None:
        if self._lock_actor:
            await self._lock_actor.release.remote()

    async def acquire_cpu_cores(self,
                                actor_id: str,
                                actor_num_cpu_cores_to_acquire: int,
                                actor_cpu_frequency_min: float = None,
                                actor_cpu_frequency_max: float = None,
                                actor_cpu_frequency_unit: str = "MHz",
                                lock_timeout: float = None) -> list:
        await self._acquire_lock(timeout=lock_timeout)
        try:
            while True:
                cpu_cores_in_use = {core for cores in self._cpu_cores_acquired_map.values() for core in cores}
                cpu_cores_available = [core for core in self._all_cpu_cores_available if core not in cpu_cores_in_use]
                if len(cpu_cores_available) >= actor_num_cpu_cores_to_acquire:
                    # Round-robin CPU cores allocation.
                    actor_acquired_cpu_cores = []
                    count = 0
                    i = self._next_core_index
                    while count < actor_num_cpu_cores_to_acquire:
                        core = self._all_cpu_cores_available[i % len(self._all_cpu_cores_available)]
                        if core not in cpu_cores_in_use and core not in actor_acquired_cpu_cores:
                            actor_acquired_cpu_cores.append(core)
                            count += 1
                        i += 1
                    self._next_core_index = i % len(self._all_cpu_cores_available)
                    self._cpu_cores_acquired_map[actor_id] = actor_acquired_cpu_cores
                    # Set the CPU cores frequencies, if requested.
                    if actor_cpu_frequency_min is not None and actor_cpu_frequency_max is not None:
                        set_cpu_cores_frequencies(actor_acquired_cpu_cores,
                                                  actor_cpu_frequency_min,
                                                  actor_cpu_frequency_max,
                                                  actor_cpu_frequency_unit)
                    return actor_acquired_cpu_cores
                await sleep(0.1)
        finally:
            await self._release_lock()

    async def release_cpu_cores(self,
                                actor_id: str,
                                lock_timeout: float = None) -> None:
        await self._acquire_lock(timeout=lock_timeout)
        try:
            if actor_id in self._cpu_cores_acquired_map:
                actor_acquired_cpu_cores = self._cpu_cores_acquired_map[actor_id]
                # Restore the CPU cores frequencies to their original values.
                restore_original_cpu_cores_frequencies(actor_acquired_cpu_cores, self._original_cpu_cores_frequencies)
                # Restore the CPU cores governors to their original values.
                restore_original_cpu_cores_governors(actor_acquired_cpu_cores, self._original_cpu_governors)
                self._cpu_cores_acquired_map.pop(actor_id)
        finally:
            await self._release_lock()

    @staticmethod
    def ping() -> str:
        return "pong"
