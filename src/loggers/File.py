from loggers import ContextFilter
import logging, click

class File:
    def __init__(self, path):
        self.format = "{\"label\":\"%(levelname)s\", \"time\":\"%(asctime)s\", %(message)s}"
        self.logger = logging.getLogger(path)
        self.logger.addFilter(ContextFilter())
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter(self.format))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logid = 0

    def log(self, value):
        self.logger.info(f"\"id\": {self.logid}, \"message\":" + str(value).replace("'","\""))
        self.logid += 1

