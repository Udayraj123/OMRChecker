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
        
    
    def stringify(func):
        def inner(self, method_type, *msg, sep=' '):
            nmsg = []
            for v in msg:
                if not isinstance(v, str):
                    v = str(v)
                nmsg.append(v)
            func(self, method_type, *nmsg, sep=sep)
        return inner
    
    # set stack level to 3 so that the caller of this function is logged, not this function itself.
    # stack-frame - self.log.debug - logutil - stringify - log method - caller
    @stringify 
    def logutil(self, method_type, *msg, sep):
        func = getattr(self.log, method_type, __default = None)
        if not func:
            return
        func(sep.join(msg), stacklevel=4)
    
    def debug(self, *msg, sep=' ', end='\n'):
        self.logutil('debug', *msg, sep=sep)
        
    def info(self, *msg, sep=' ', end='\n'):
        self.logutil('info', *msg, sep=sep)
        
    def warning(self, *msg, sep=' ', end='\n'):
        self.logutil('warning', *msg, sep=sep)
        
    def error(self, *msg, sep=' ', end='\n'):
        self.logutil('error', *msg, sep=sep)
    
    def critical(self, *msg, sep=' ', end='\n'):
        self.logutil('critical', *msg, sep=sep)
        
logger = Logger(__name__)
