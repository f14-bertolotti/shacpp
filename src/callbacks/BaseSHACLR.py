from callbacks import callback
import click


class BaseSHACLR():
    def __init__(self, trainer):
        self.trainer = trainer
    def __call__(self, tbar, batch, update, epoch, step, losses, trajectory_result, **kwargs):
        environment_loss = self.trainer.environment.train_step(**batch)
        loss = losses["loss"].item()
        avg_reward = batch["rewards"].mean().item()
        avg_real_reward = batch["real_rewards"].mean().item()
        trajectory_loss = trajectory_result["loss"].item()
        tbar.set_description(f"{update}-{epoch}-{step}, lr:{self.trainer.scheduler.get_last_lr()[0]:7.6f}, l:{loss:7.4f}, tl:{trajectory_loss:7.4f}, el:{environment_loss:7.4f}, rr:{avg_real_reward:7.4f}, r:{avg_reward:7.4f}")
        self.trainer.logger.log({
            "loss"             : loss,
            "trajectory_loss"  : trajectory_loss,
            "environment_loss" : environment_loss,
            "reward"           : avg_reward,
            "real_reward"      : avg_real_reward,
            "update"           : update,
            "epoch"            : epoch,
            "step"             : step,
        })

@callback.group(invoke_without_command=True)
@click.pass_obj
def base_shac_learnable_reward(trainer):
    trainer.add_callback(
        BaseSHACLR(trainer=trainer)
    )


