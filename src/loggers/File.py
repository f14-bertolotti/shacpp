from loggers import logger
from loggers import ContextFilter
import logging, click

class File:
    def __init__(self, path):
        self.format = "{\"lebel\":\"%(levelname)s\", \"id\": %(record_number)s, \"time\":\"%(asctime)s\", \"message\":{%(message)s}}"
        self.logger = logging.getLogger(path)
        self.logger.addFilter(ContextFilter())
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter(self.format))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logid = 0

    def log(self, value):
        self.logger.info(str(value).replace("'","\""))

@logger.group(invoke_without_command=True)
@click.option("--path", "path", type=click.Path(), default="./train.log")
@click.pass_obj
def file(trainer, path):
    trainer.set_logger(
        File(
            path = path
        )
    )

