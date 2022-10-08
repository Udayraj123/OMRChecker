import logging
from typing import Union
from rich.logging import RichHandler
from .config import CONFIG_DEFAULTS as config

FORMAT = "%(message)s"

logging.basicConfig(
                level=logging.NOTSET, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=config.outputs.rich_tracebacks)]
                )

class Logger:
    def __init__(self, name, level: Union[int, str] = logging.NOTSET,  message_format = "%(message)s", date_format = "[%X]"):
        self.log = logging.getLogger(name)
        self.log.setLevel(level)
        self.log.__format__ = message_format
        self.log.__date_format__ = date_format
        print(self.log.name)
        
    
    def stringify(func):
        def inner(self, *msg, sep=' ', end='\n'):
            nmsg = []
            for v in msg:
                if not isinstance(v, str):
                    v = str(v)
                nmsg.append(v)
            func(self, *nmsg, sep=sep, end=end)
        return inner
                
    @stringify
    def debug(self, *msg, sep=' ', end='\n'):
        self.log.debug(sep.join(msg), stacklevel=3)  # set stack level to 3 so that the caller of this function is logged, not this function itself. stack-frame - self.log.debug - stringify:28 - caller
        
    @stringify
    def info(self, *msg, sep=' ', end='\n'):
        self.log.info(sep.join(msg), stacklevel=3)
        
    @stringify
    def warning(self, *msg, sep=' ', end='\n'):
        self.log.warning(sep.join(msg), stacklevel=3)
        
    @stringify
    def error(self, *msg, sep=' ', end='\n'):
        self.log.error(sep.join(msg), stacklevel=3)
    
    @stringify
    def critical(self, *msg, sep=' ', end='\n'):
        self.log.critical(sep.join(msg), stacklevel=3)
        
logger = Logger(__name__)