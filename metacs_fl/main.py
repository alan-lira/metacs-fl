import sys
from os import environ

# Suppress TensorFlow C++ log messages without silencing Python tracebacks.
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS
from logging import Logger
from pathlib import Path
from sys import argv
from time import perf_counter

from metacs_fl.client_launcher.flower_client_launcher import FlowerClientLauncher
from metacs_fl.result_analyzer.result_analyzer import ResultAnalyzer
from metacs_fl.server_launcher.flower_server_launcher import FlowerServerLauncher
from metacs_fl.flower_executor.flower_executor import FlowerExecutor
from metacs_fl.flower_simulator.flower_simulator import FlowerSimulator
from metacs_fl.utils.logger_util import load_logger, log_message
from metacs_fl.utils.setup_tools_util import get_version

# Paths.
__BASE_PATH = Path(__file__).parent.resolve()
__VERSION_FILE = __BASE_PATH.joinpath("VERSION")
__FLOWER_CLIENT_CONFIG_FILE = __BASE_PATH.joinpath("client/config/flower_client.cfg")
__FLOWER_SERVER_CONFIG_FILE = __BASE_PATH.joinpath("server/config/flower_server.cfg")
__FLOWER_EXECUTOR_CONFIG_FILE = __BASE_PATH.joinpath("flower_executor/config/flower_executor.cfg")
__FLOWER_SIMULATOR_CONFIG_FILE = __BASE_PATH.joinpath("flower_simulator/config/flower_simulator.cfg")
__RESULT_ANALYZER_CONFIG_FILE = __BASE_PATH.joinpath("result_analyzer/config/results_analyzer.cfg")

# List of implemented tools.
__AVAILABLE_TOOLS = [{"name": "MetaCS-FL",
                      "description": "",
                      "actions": [{"launch_server": "launches a FL server instance"},
                                  {"launch_client": "launches a FL client instance"},
                                  {"execute_fl_with_flower": "runs a FL execution using Flower"},
                                  {"execute_fl_with_flower_simulation_engine": "runs a FL execution using Flower's Simulation Engine"},
                                  {"analyze_results": "analyzes the results"}]}]


def _get_tools_info() -> list:
    tools_info = []
    for available_tool in __AVAILABLE_TOOLS:
        tool_actions = ""
        for action in available_tool["actions"]:
            tool_action = list(action.keys())[0]
            tool_action_description = list(action.values())[0]
            tool_actions += tool_action.ljust(25) + tool_action_description + "\n"
        tool_info = "{0}".format(tool_actions)
        tools_info.append(tool_info)
    return tools_info


def _get_tools_names() -> list:
    tools_names = []
    for available_tool in __AVAILABLE_TOOLS:
        tool_name = available_tool["name"]
        tools_names.append(tool_name)
    return tools_names


def _verify_if_tool_name_is_valid(tool_name: str) -> None:
    tools_names = _get_tools_names()
    if tool_name not in tools_names:
        error_message = "'{0}' is not a valid tool.\nAvailable tools: {1}".format(tool_name, ", ".join(tools_names))
        raise ValueError(error_message)


def _get_tool_actions(tool_name: str) -> list:
    tool_actions = []
    for available_tool in __AVAILABLE_TOOLS:
        if available_tool["name"] == tool_name:
            for action in available_tool["actions"]:
                tool_actions.extend(list(action.keys()))
            break
    return tool_actions


def _verify_if_tool_action_is_valid(tool_name: str,
                                    tool_action: str) -> None:
    tool_actions = _get_tool_actions(tool_name)
    if tool_action not in tool_actions:
        error_message = ("'{0}' is not a valid action for the {1} tool.\nAvailable actions: {2}"
                         .format(tool_action, tool_name, ", ".join(tool_actions)))
        raise ValueError(error_message)


def _set_logger() -> Logger:
    logging_settings = {"enable_logging": True,
                        "log_to_file": False,
                        "log_to_console": True,
                        "file_name": None,
                        "file_mode": None,
                        "encoding": None,
                        "level": "INFO",
                        "format_str": "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        "date_format": "%Y/%m/%d %H:%M:%S"}
    logger_name = "MetaCS-FL_Logger"
    logger = load_logger(logging_settings, logger_name)
    return logger


def _verify_if_config_file_is_valid(config_file: Path) -> None:
    if not config_file.is_file():
        error_message = "The '{0}' config file was not found!".format(config_file)
        raise FileNotFoundError(error_message)


def _str_to_bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError("Invalid boolean value: {0}".format(value))


def main() -> None:
    # Begin.
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Get the version.
    version = get_version(__VERSION_FILE)
    # Parse the MetaCS-FL arguments.
    usage = "metacs-fl [-v | --version] [-h | --help] <action> [args]"
    description = '''Runs real-world FL experiments for the evaluation of client selection algorithms.\n'''
    epilog = ('''{0}'''.format("\n".join([tool_info for tool_info in _get_tools_info()])))
    ap = ArgumentParser(usage=usage,
                        description=description,
                        epilog=epilog,
                        formatter_class=RawTextHelpFormatter)
    ap.add_argument("-v", "--version",
                    action="version",
                    help="show version number and exit",
                    version="%(prog)s version {0}".format(version))
    ap.add_argument("action",
                    type=str,
                    help="action to perform with MetaCS-FL tool")
    default_config_file = None
    if "launch_server" in argv:
        default_config_file = __FLOWER_SERVER_CONFIG_FILE
        ap.add_argument("--id",
                        type=int,
                        required=True,
                        help=SUPPRESS)
    elif "launch_client" in argv:
        default_config_file = __FLOWER_CLIENT_CONFIG_FILE
        ap.add_argument("--id",
                        type=int,
                        required=True,
                        help=SUPPRESS)
    elif "execute_fl_with_flower" in argv:
        default_config_file = __FLOWER_EXECUTOR_CONFIG_FILE
        ap.add_argument("--repetitions",
                        type=int,
                        required=False,
                        default=1,
                        help=SUPPRESS)
        ap.add_argument("--hostfile",
                        type=Path,
                        required=False,
                        default=None,
                        help=SUPPRESS)
        ap.add_argument("--gather-outputs",
                        type=str,
                        required=False,
                        default=None,
                        help=SUPPRESS)
        ap.add_argument("--output-collector-ip",
                        type=str,
                        required=False,
                        default=None,
                        help=SUPPRESS)
        ap.add_argument("--output-collector-user",
                        type=str,
                        required=False,
                        default=None,
                        help=SUPPRESS)
        ap.add_argument("--output-collector-folder",
                        type=Path,
                        required=False,
                        default=None,
                        help=SUPPRESS)
        ap.add_argument("--output-collection-method",
                        type=str,
                        required=False,
                        default=None,
                        help=SUPPRESS)
        ap.add_argument("--fail-on-output-collection-error",
                        type=str,
                        required=False,
                        default=None,
                        help=SUPPRESS)
        ap.add_argument("--gather-single-node-execution",
                        type=str,
                        required=False,
                        default=None,
                        help=SUPPRESS)
        ap.add_argument("--output-node-identifier",
                        type=str,
                        required=False,
                        default=None,
                        help=SUPPRESS)
    elif "execute_fl_with_flower_simulation_engine" in argv:
        default_config_file = __FLOWER_SIMULATOR_CONFIG_FILE
    elif "analyze_results" in argv:
        default_config_file = __RESULT_ANALYZER_CONFIG_FILE
    ap.add_argument("--config-file",
                    type=Path,
                    required=False,
                    default=default_config_file,
                    help=SUPPRESS)
    parsed_args = ap.parse_args()
    # Get the user-provided arguments.
    action = str(parsed_args.action)
    # Verify if the user-provided action is valid.
    _verify_if_tool_action_is_valid("MetaCS-FL", action)
    # Set the logger.
    logger = _set_logger()
    if action == "launch_server":
        id_ = int(parsed_args.id)
        config_file = Path(parsed_args.config_file)
        # Verify if the user-provided config file is valid.
        _verify_if_config_file_is_valid(config_file)
        fs = FlowerServerLauncher(id_, config_file)
        fs.launch_server()
    elif action == "launch_client":
        id_ = int(parsed_args.id)
        config_file = Path(parsed_args.config_file)
        # Verify if the user-provided config file is valid.
        _verify_if_config_file_is_valid(config_file)
        fc = FlowerClientLauncher(id_, config_file)
        fc.launch_client()
    elif action == "execute_fl_with_flower":
        config_file = Path(parsed_args.config_file)
        repetitions = int(parsed_args.repetitions)
        hostfile = parsed_args.hostfile
        if hostfile is not None:
            hostfile = Path(hostfile)
        output_collector_folder = parsed_args.output_collector_folder
        if output_collector_folder is not None:
            output_collector_folder = Path(output_collector_folder)
        output_gathering_settings = {"gather_outputs": _str_to_bool_or_none(parsed_args.gather_outputs),
                                     "output_collector_ip": parsed_args.output_collector_ip,
                                     "output_collector_user": parsed_args.output_collector_user,
                                     "output_collector_folder": output_collector_folder,
                                     "output_collection_method": parsed_args.output_collection_method,
                                     "fail_on_output_collection_error": _str_to_bool_or_none(parsed_args.fail_on_output_collection_error),
                                     "gather_single_node_execution": _str_to_bool_or_none(parsed_args.gather_single_node_execution),
                                     "output_node_identifier": parsed_args.output_node_identifier}
        output_gathering_settings = {k: v for k, v in output_gathering_settings.items() if v is not None}
        _verify_if_config_file_is_valid(config_file)
        if hostfile is not None:
            _verify_if_config_file_is_valid(hostfile)
        fe = FlowerExecutor(config_file,
                            repetitions,
                            hostfile=hostfile,
                            output_gathering_settings=output_gathering_settings)
        fe.execute_fl_with_flower()
    elif action == "execute_fl_with_flower_simulation_engine":
        config_file = Path(parsed_args.config_file)
        # Verify if the user-provided config file is valid.
        _verify_if_config_file_is_valid(config_file)
        fs = FlowerSimulator(config_file)
        fs.execute_fl_with_flower_simulation_engine()
    elif action == "analyze_results":
        config_file = Path(parsed_args.config_file)
        # Verify if the user-provided config file is valid.
        _verify_if_config_file_is_valid(config_file)
        ra = ResultAnalyzer(config_file)
        ra.analyze_results()
    # Stop the performance counter.
    perf_counter_stop = perf_counter()
    # Log an 'elapsed time in seconds' message.
    elapsed_time_seconds = round((perf_counter_stop - perf_counter_start), 2)
    message = "Elapsed time: {0} {1}.".format(elapsed_time_seconds,
                                              "seconds" if elapsed_time_seconds != 1 else "second")
    log_message(logger, message, "INFO")
    # End.
    exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        from traceback import format_exc
        print("[FATAL] main.py failed: {0}".format(e), file=sys.stdout, flush=True)
        print(format_exc(), file=sys.stdout, flush=True)
        print("[FATAL] main.py failed: {0}".format(e), file=sys.stderr, flush=True)
        print(format_exc(), file=sys.stderr, flush=True)
        raise
