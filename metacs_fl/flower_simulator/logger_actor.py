from ray import remote

from metacs_fl.utils.logger_util import load_logger


@remote
class LoggerActor:

    def __init__(self) -> None:
        self._logger = None

    def initialize(self,
                   logging_settings: dict,
                   logger_name: str) -> None:
        self._logger = load_logger(logging_settings, logger_name)

    @staticmethod
    def ping() -> str:
        return "pong"

    def log(self,
            level: str,
            msg: str):
        log_method = getattr(self._logger, level.lower(), None)
        if log_method is not None:
            log_method(msg)

    def flush(self):
        for handler in self._logger.handlers:
            handler.flush()

    def getEffectiveLevel(self) -> int:
        return self._logger.getEffectiveLevel()


class RemoteLoggerAdapter:

    def __init__(self,
                 logger_actor: any) -> None:
        self._logger_actor = logger_actor

    def info(self,
             msg: str) -> None:
        self._logger_actor.log.remote("info", msg)

    def debug(self,
              msg: str) -> None:
        self._logger_actor.log.remote("debug", msg)

    def warning(self,
                msg: str) -> None:
        self._logger_actor.log.remote("warning", msg)

    def error(self,
              msg: str) -> None:
        self._logger_actor.log.remote("error", msg)

    def critical(self,
                 msg: str) -> None:
        self._logger_actor.log.remote("critical", msg)

    def flush(self) -> None:
        self._logger_actor.flush.remote()

    def getEffectiveLevel(self) -> int:
        return self._logger_actor.getEffectiveLevel.remote()
