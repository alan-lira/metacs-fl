from pandas import read_csv
from pandas.errors import EmptyDataError
from pathlib import Path


def get_remaining_battery_energy_from_file(remaining_battery_energy_file: Path) -> float:
    remaining_battery_energy_in_joules = 0
    if remaining_battery_energy_file.is_file():
        try:
            with open(file=remaining_battery_energy_file, mode="r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith("comm_round"):
                        remaining_battery_energy_in_joules = float(line.split(",")[3])
                        break
        except (IndexError, ValueError):
            pass  # Return 0 if any issue occurs during parsing
    return remaining_battery_energy_in_joules


def initialize_remaining_battery_energy_file(remaining_battery_energy_file: Path,
                                             device_emulation_settings: dict) -> float:
    if remaining_battery_energy_file.is_file():
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
    else:
        battery_maximum_stored_energy_in_joules = float(device_emulation_settings["battery_maximum_stored_energy_in_joules"])
        if "battery_stored_energy_in_joules" in device_emulation_settings:
            remaining_battery_energy_in_joules = float(device_emulation_settings["battery_stored_energy_in_joules"])
        else:
            remaining_battery_energy_in_joules = battery_maximum_stored_energy_in_joules
        remaining_battery_energy_in_percentage = round(remaining_battery_energy_in_joules / battery_maximum_stored_energy_in_joules, 4)
        remaining_battery_energy_file.parent.mkdir(exist_ok=True, parents=True)
        with open(file=remaining_battery_energy_file, mode="a", encoding="utf-8") as file:
            file.write("comm_round,phase,event,remaining_battery_level_in_joules,remaining_battery_level_in_percentage\n")
            file.write("{0},{1},{2},{3},{4}\n"
                       .format(0,
                               "initialization",
                               "before_establishing_connection_to_the_server",
                               remaining_battery_energy_in_joules,
                               remaining_battery_energy_in_percentage))
    return remaining_battery_energy_in_joules


def check_if_event_already_exists(remaining_battery_energy_file: Path,
                                  comm_round: int,
                                  phase: str,
                                  event: str) -> bool:
    try:
        df = read_csv(remaining_battery_energy_file)
        # Check if the required columns exist.
        if not {"comm_round", "phase", "event"}.issubset(df.columns):
            return False
        # Filter by comm_round and phase, then check if the event exists.
        filtered_df = df[(df["comm_round"] == comm_round) & (df["phase"] == phase)]
        return event in filtered_df["event"].values
    except (FileNotFoundError, EmptyDataError, KeyError):
        return False  # The file doesn't exist, is empty, or lacks the necessary columns.


def update_remaining_battery_energy_file(remaining_battery_energy_file: Path,
                                         comm_round: int,
                                         phase: str,
                                         event: str,
                                         consumed_energy_in_joules: float,
                                         battery_maximum_stored_energy_in_joules: float,
                                         device_connected_to_a_power_source: bool = False) -> None:
    event_exists = check_if_event_already_exists(remaining_battery_energy_file, comm_round, phase, event)
    if not event_exists:
        remaining_battery_energy_in_joules = get_remaining_battery_energy_from_file(remaining_battery_energy_file)
        if remaining_battery_energy_in_joules > 0:
            if not device_connected_to_a_power_source:
                remaining_battery_energy_in_joules = max(0.0, remaining_battery_energy_in_joules - consumed_energy_in_joules)
            else:
                pass  # TODO: Implement the battery charging logic here (possibly).
            remaining_battery_energy_in_percentage = round(remaining_battery_energy_in_joules / battery_maximum_stored_energy_in_joules, 4)
            with open(file=remaining_battery_energy_file, mode="a", encoding="utf-8") as file:
                file.write("{0},{1},{2},{3},{4}\n".format(comm_round,
                                                          phase,
                                                          event,
                                                          remaining_battery_energy_in_joules,
                                                          remaining_battery_energy_in_percentage))
