from callbacks import callback, Callback
import torch, click

class CheckpointAgent(Callback):
    def __init__(self, trainer, etc = 100, path="agent.pkl"):
        self.trainer = trainer
        self.path    =    path
        self.etc     =     etc

    def end_episode(self, data):
        if data["episode"] % self.etc == 0:
            torch.save({
                    "agentsd"  : self.trainer.agent.state_dict(),
                    "episode"  : data["episode"]
                }, self.path)


@callback.group()
@click.option("--path" , "path" , type=click.Path() , default="agent.pkl" , help="path in which the checkpoint is saved" )
@click.option("--etc"  , "etc"  , type=int          , default=10          , help="epochs to checkpoint"                  )
@click.pass_obj
def checkpoint_agent(trainer, path, etc):
    trainer.add_callback(
        CheckpointAgent(
            trainer = trainer,
            path    =    path,
            etc     =     etc,
        )
    )

