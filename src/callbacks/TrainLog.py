from callbacks import callback, Callback
from loggers import File
import time, click

class TrainLog(Callback):
    def __init__(self, trainer, path="train.log"):
        self.logger    = File(path)
        self.trainer   = trainer
        self.path      =    path
        self.starttime = time.time()

    def end_step(self, data):
        episode, epoch, step = data["episode"], data["epoch"], data["step"]
        loss   = data["lossval"].item()
        reward = data["trajectories"]["rewards"].sum().item() / data["trajectories"]["rewards"].size(0)
        self.logger.log({
            "episode"   : episode                                                ,
            "epoch"     : epoch                                                  ,
            "step"      : step                                                   ,
            "reward"    : reward                                                 ,
            "loss"      : loss                                                   ,
            "lr"        : self.trainer.algorithm.optimizer.param_groups[0]["lr"] ,
            "time"      : time.time() - self.starttime                           , 
        })

@callback.group()
@click.option("--path" , "path" , type=click.Path() , default="train.log" , help="logger path" )
@click.pass_obj
def trainlog(trainer, path):
    trainer.add_callback(
        TrainLog(
            trainer = trainer,
            path    =    path,
        )
    )

