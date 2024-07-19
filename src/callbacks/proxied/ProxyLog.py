from callbacks.proxied import proxied
from callbacks import Callback
from loggers import File
import click

class ProxyLog(Callback):
    def __init__(self, trainer, path="train.log"):
        self.logger  = File(path)
        self.trainer = trainer
        self.path    =    path

    def end_epoch_proxy(self, data):

        self.logger.log({
            "update"   : data["self"].updates ,
            "epoch"    : data["epoch"]        ,
            "step"     : data["step"]         ,
            "loss"     : data["loss"].item()  ,
            "accuracy" : data["accuracy"]
        })

@proxied.group()
@click.option("--path" , "path" , type=click.Path() , default="proxy_train.log" , help="logger path" )
@click.pass_obj
def proxylog(trainer, path):
    trainer.add_callback(
        ProxyLog(
            trainer = trainer,
            path    =    path,
        )
    )

