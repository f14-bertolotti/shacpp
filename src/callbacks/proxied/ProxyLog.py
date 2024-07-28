from callbacks.proxied import proxied
from callbacks import Callback
from loggers import File
import torch, click

class ProxyLog(Callback):
    def __init__(self, trainer, path="train.log"):
        self.logger  = File(path)
        self.trainer = trainer
        self.path    =    path

    def end_epoch_proxy(self, data):
        nonempty = data["self"].mask.sum().item()

        self.logger.log({
            "update"   : data["self"].updates ,
            "filled"   : nonempty,
            "epoch"    : data["epoch"]        ,
            "step"     : data["step"]         ,
            "loss"     : data["loss"].item()  ,
            "dist"     : torch.nn.functional.avg_pool1d(data["self"].rewards[data["self"].mask].unsqueeze(0), kernel_size=nonempty//10, stride=nonempty//10).tolist()[0] ,
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

