import logging
from typing import Union

from rich.console import Console
from rich.logging import RichHandler

FORMAT = "%(message)s"

# TODO: set logging level from config.json dynamically
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


class Logger:
    def __init__(
        self,
        name,
        level: Union[int, str] = logging.NOTSET,
        message_format="%(message)s",
        date_format="[%X]",
    ):
        self.log = logging.getLogger(name)
        self.log.setLevel(level)
        self.log.__format__ = message_format
        self.log.__date_format__ = date_format

    def debug(self, *msg: object, sep=" ", end="\n") -> None:
        return self.logutil("debug", *msg, sep=sep)

    def info(self, *msg: object, sep=" ", end="\n") -> None:
        return self.logutil("info", *msg, sep=sep)

    def warning(self, *msg: object, sep=" ", end="\n") -> None:
        return self.logutil("warning", *msg, sep=sep)

    def error(self, *msg: object, sep=" ", end="\n") -> None:
        return self.logutil("error", *msg, sep=sep)

    def critical(self, *msg: object, sep=" ", end="\n") -> None:
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
        func = getattr(self.log, method_type, None)
        if not func:
            raise AttributeError(f"Logger has no method {method_type}")
        return func(sep.join(msg), stacklevel=4)


logger = Logger(__name__)
console = Console()
