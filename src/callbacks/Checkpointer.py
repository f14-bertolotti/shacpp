from callbacks import callback
import torch, click


class Checkpointer():
    def __init__(self, trainer, path="agent.pkl", utc=10):
        self.trainer, self.utc, self.path = trainer, utc, path
    def __call__(self, update, epoch, step, **kwargs):
        if update % self.utc == 0 and epoch == 0 and step == 0: torch.save({"agentsd" : self.trainer.agent.state_dict()}, self.path)

@callback.group(invoke_without_command=True)
@click.option("--path", "path", type=click.Path(), default="agent.pkl")
@click.option("--utc" , "utc" , type=int         , default=10)
@click.pass_obj
def checkpointer(trainer, path, utc):
    trainer.add_callback(
        Checkpointer(
            path    = path,
            utc     = utc,
            trainer = trainer
        )
    )







            
