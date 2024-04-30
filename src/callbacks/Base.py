from callbacks import callback
import click


class Base():
    def __init__(self, trainer):
        self.trainer = trainer
    def __call__(self, tbar, batch, update, epoch, step, losses, **kwargs):
        loss = losses["loss"].item()
        avg_reward = batch["rewards"].mean().item()
        tbar.set_description(f"{update}-{epoch}-{step}, lr:{self.trainer.scheduler.get_last_lr()[0]:7.6f}, l:{loss:7.4f}, r:{avg_reward:7.4f}")
        self.trainer.logger.log({
            "loss"   : loss.item(),
            "reward" : avg_reward,
            "update" : update,
            "epoch"  : epoch,
            "step"   : step,
        })

@callback.group(invoke_without_command=True)
@click.pass_obj
def base(trainer):
    trainer.add_callback(
        Base(trainer=trainer)
    )





