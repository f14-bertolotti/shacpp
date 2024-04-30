from losses import policy_loss, value_loss, loss
import click

class PPOLRLoss:
    def __init__(self, clipcoef=.2, vfcoef=.5, entcoef=.1):
        self.clipcoef, self.vfcoef, self.entcoef = clipcoef, vfcoef, entcoef

    def __call__(self, new, old, **kwargs):
        advantages = (old["advantages"] - old["advantages"].mean()) / (old["advantages"].std() + 1e-8)
        ploss = policy_loss(advantages, (new["logprobs"] - old["logprobs"]).exp(), self.clipcoef)
        vloss = value_loss(new["values"], old["values"], old["returns"], self.clipcoef)
        eloss = old["entropy"].mean()
        loss = ploss + vloss * self.vfcoef - self.entcoef * eloss 
        return {
            "loss"  : loss,
            "ploss" : ploss,
            "vloss" : vloss,
            "eloss" : eloss,
        }

@loss.group(invoke_without_command=True)
@click.option("--clip-coef" , "clipcoef" , type=float , default=.2)
@click.option("--vf-coef"   , "vfcoef"   , type=float , default=.5)
@click.option("--ent-coef"  , "entcoef"  , type=float , default=.1)
@click.pass_obj
def ppo_learnable_reward(trainer, clipcoef, vfcoef, entcoef):
    trainer.set_loss(
        PPOLRLoss(
            clipcoef = clipcoef,
            vfcoef   =   vfcoef,
            entcoef  =  entcoef
        )
    )
 
