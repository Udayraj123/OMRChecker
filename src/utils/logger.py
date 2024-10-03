import logging

from rich.console import Console
from rich.logging import RichHandler

FORMAT = "%(message)s"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

DEFAULT_LOG_LEVEL_MAP = {
    "critical": True,
    "error": True,
    "warning": True,
    "info": True,
    "debug": True,
}


class Logger:
    def __init__(
        self,
        name,
        message_format="%(message)s",
        date_format="[%X]",
    ):
        self.log = logging.getLogger(name)
        self.log.setLevel(logging.DEBUG)
        self.log.__format__ = message_format
        self.log.__date_format__ = date_format
        self.reset_log_levels()

    def set_log_levels(self, show_logs_by_type):
        self.show_logs_by_type = {**DEFAULT_LOG_LEVEL_MAP, **show_logs_by_type}

    def reset_log_levels(self):
        self.show_logs_by_type = DEFAULT_LOG_LEVEL_MAP

    def debug(self, *msg: object, sep=" "):
        return self.logutil("debug", *msg, sep=sep)

    def info(self, *msg: object, sep=" "):
        return self.logutil("info", *msg, sep=sep)

    def warning(self, *msg: object, sep=" "):
        return self.logutil("warning", *msg, sep=sep)

    def error(self, *msg: object, sep=" "):
        return self.logutil("error", *msg, sep=sep)

    def critical(self, *msg: object, sep=" "):
        return self.logutil("critical", *msg, sep=sep)

    def stringify(func):
        def inner(self, method_type: str, *msg: object, sep=" "):
            nmsg = []
            for v in msg:
                if not isinstance(v, str):
                    v = str(v)
                nmsg.append(v)
            return func(self, method_type, *nmsg, sep=sep)

        return inner

    # set stack level to 3 so that the caller of this function is logged, not this function itself.
    # stack-frame - self.log.debug - logutil - stringify - log method - caller
    @stringify
    def logutil(self, method_type: str, *msg: object, sep=" ") -> None:
        if self.show_logs_by_type[method_type] is False:
            return
        logger_func = getattr(self.log, method_type, None)
        if not logger_func:
            raise AttributeError(f"Logger has no method {method_type}")
        return logger_func(sep.join(msg), stacklevel=4)


logger = Logger(__name__)
console = Console()
