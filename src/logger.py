import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"

class Logger:
    def __init__(self):
        self.__FORMAT = "%(message)s"
        logging.basicConfig(
                level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
                )
        self.__logger = logging.getLogger("rich")
        
    def stringfy(func):
        def inner(self, *msg, sep=' ', end='\n'):
            nmsg = []
            for v in msg:
                if not isinstance(v, str):
                    v = str(v)
                nmsg.append(v)
            func(self, *nmsg, sep=sep, end=end)
        return inner
                
    @stringfy
    def debug(self, *msg, sep=' ', end='\n'):
        self.__logger.debug(sep.join(msg))
        
    @stringfy
    def info(self, *msg, sep=' ', end='\n'):
        self.__logger.info(sep.join(msg))
        
    @stringfy
    def warning(self, *msg, sep=' ', end='\n'):
        self.__logger.warning(sep.join(msg))
        
    @stringfy
    def error(self, *msg, sep=' ', end='\n'):
        self.__logger.error(sep.join(msg))
    
    @stringfy
    def critical(self, *msg, sep=' ', end='\n'):
        self.__logger.critical(sep.join(msg))
        
logger = Logger()