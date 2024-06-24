from losses import loss
import click

class SHACLoss:
    def __init__(self):
        pass

    def __call__(self, new, old, **kwargs):
        return {"loss" : ((new["values"] - old["target_values"])**2).mean()}

    def __str__(self): return "SHACLoss()"


@loss.group(invoke_without_command=True)
@click.pass_obj
def shac(trainer):
    trainer.set_loss(SHACLoss())

