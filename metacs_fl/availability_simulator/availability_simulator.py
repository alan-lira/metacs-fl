from collections import deque
from numpy.random import default_rng, Generator


class AvailabilitySimulator:
    """
    Per-client intermittent availability simulator.

    The simulator models client availability as a two-state Markov process:

        available   -> unavailable with probability p_off
        unavailable -> available   with probability p_on

    This creates temporal persistence: if a client is unavailable in round r,
    it is more likely to remain unavailable in round r+1 when p_on is small.

    Each client should own one independent AvailabilitySimulator instance.
    """

    def __init__(self,
                 enabled: bool = False,
                 initial_available: bool = True,
                 p_off: float = 0.05,
                 p_on: float = 0.50,
                 seed: int | None = None,
                 profile_name: str = "intermittent",
                 history_size: int = 100,
                 force_available_during_profiling: bool = True) -> None:
        self._enabled = bool(enabled)
        self._currently_available = bool(initial_available)
        self._p_off = self._validate_probability(p_off, "p_off")
        self._p_on = self._validate_probability(p_on, "p_on")
        self._profile_name = profile_name
        self._rng: Generator = default_rng(seed)
        self._history = deque(maxlen=history_size)
        self._round_cache = {}
        self._last_simulated_round = None
        self._force_available_during_profiling = bool(force_available_during_profiling)

    @staticmethod
    def _validate_probability(value: float,
                              name: str) -> float:
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("{0} must be in [0, 1], got {1}".format(name, value))
        return value

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def _step_once(self) -> bool:
        """
        Advance the Markov chain by one round.
        """
        currently_available = self.get_attribute("_currently_available")
        p_off = self.get_attribute("_p_off")
        p_on = self.get_attribute("_p_on")
        rng = self.get_attribute("_rng")
        if currently_available:
            # Available clients may become unavailable.
            next_available = not (rng.random() < p_off)
        else:
            # Unavailable clients may return.
            next_available = rng.random() < p_on
        self._set_attribute("_currently_available", next_available)
        return next_available

    def simulate_availability_change(self,
                                     comm_round: int,
                                     is_profiling_round: bool = False) -> bool:
        """
        Return this client's availability for comm_round.

        The result is cached per round so repeated get_properties calls in the
        same round do not advance the process multiple times.
        """
        enabled = self.get_attribute("_enabled")
        force_available_during_profiling = self.get_attribute("_force_available_during_profiling")
        if not enabled:
            return True
        if is_profiling_round and force_available_during_profiling:
            return True
        comm_round = int(comm_round)
        round_cache = self.get_attribute("_round_cache")
        if comm_round in round_cache:
            return round_cache[comm_round]
        last_simulated_round = self.get_attribute("_last_simulated_round")
        # First observed round: use the initial state, do not immediately transition away from it.
        # This is useful for experiments where all clients initially exist and start available.
        if last_simulated_round is None:
            current_available = self.get_attribute("_currently_available")
            self._set_attribute("_last_simulated_round", comm_round)
        else:
            current_available = self.get_attribute("_currently_available")
            # Advance one Markov step for each missing round.
            # This keeps the state meaningful even if a client is not queried every round.
            steps = max(1, comm_round - last_simulated_round)
            for _ in range(steps):
                current_available = self._step_once()
            self._set_attribute("_last_simulated_round", comm_round)
        round_cache[comm_round] = current_available
        self._set_attribute("_round_cache", round_cache)
        history = self.get_attribute("_history")
        history.append((comm_round, current_available))
        self._set_attribute("_history", history)
        return current_available

    def get_history_as_string(self) -> str:
        history = self.get_attribute("_history")
        return "|".join(["{0}:{1}".format(comm_round, int(available)) for comm_round, available in history])

    def get_state_as_dict(self) -> dict:
        return {"availability_profile": self.get_attribute("_profile_name"),
                "availability_enabled": self.get_attribute("_enabled"),
                "availability_current_state": self.get_attribute("_currently_available"),
                "availability_p_off": self.get_attribute("_p_off"),
                "availability_p_on": self.get_attribute("_p_on"),
                "availability_history": self.get_history_as_string()}
