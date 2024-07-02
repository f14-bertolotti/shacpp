from callbacks import callback
import click


class Dispersion():
    def __init__(self, trainer):
        self.trainer = trainer
    def __call__(self, tbar, storage, update, epoch, step, lossval, **kwargs):
        loss = lossval.item()
        avg_reward = storage.dictionary["rewards"].sum().item() / (self.trainer.environment.envs)
        lr = self.trainer.algorithm.optimizer.param_groups[0]["lr"]
        tbar.set_description(f"{update}-{epoch}-{step}, lr:{lr:7.6f}, l:{loss:7.4f}, r:{avg_reward:7.4f}")
        self.trainer.logger.log({
            "loss"   : loss,
            "reward" : avg_reward,
            "update" : update,
            "epoch"  : epoch,
            "step"   : step,
        })

@callback.group(invoke_without_command=True)
@click.pass_obj
def dispersion(trainer):
    trainer.add_callback(
        Dispersion(trainer=trainer)
    )





