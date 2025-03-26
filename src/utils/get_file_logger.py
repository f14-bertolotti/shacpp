import logging
import json

def get_file_logger(path:str) -> logging.Logger:
    """ Create a logger that writes to a file """
    logger  = logging.  getLogger(path)
    handler = logging.FileHandler(path)
    logger .setLevel(logging.INFO)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(json.dumps(json.loads("""
        {
            "data"    : "%(asctime)s", 
            "level"   : "%(levelname)s", 
            "process" : { 
                "id"   : "%(process)d", 
                "name" : "%(processName)s"
            }, 
            "thread"  : {
                "id"   : "%(thread)d", 
                "name" : "%(threadName)s"
            }, 
            "message" : "MESSAGE" 
        }
    """), indent=None, separators=(",",":")).replace("\"MESSAGE\"","%(message)s"))

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


