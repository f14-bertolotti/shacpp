from callbacks.proxied import proxied
from callbacks import Callback
import torch, click

class CheckpointProxy(Callback):
    def __init__(self, trainer, etc = 100, path="proxy.pkl"):
        self.trainer = trainer
        self.path    =    path
        self.etc     =     etc

    def end_episode(self, data):
        if data["episode"] % self.etc == 0:
            torch.save({
                    "agentsd"  : self.trainer.environment.proxynn.state_dict(),
                    "episode"  : data["episode"]
                }, self.path)


@proxied.group()
@click.option("--path" , "path" , type=click.Path() , default="proxy.pkl", help="path in which the checkpoint is saved" )
@click.option("--etc"  , "etc"  , type=int          , default=10         , help="epochs to checkpoint"                  )
@click.pass_obj
def checkpoint_proxy(trainer, path, etc):
    trainer.add_callback(
        CheckpointProxy(
            trainer = trainer,
            path    =    path,
            etc     =     etc,
        )
    )

