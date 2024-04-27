from losses import policy_loss, value_loss, loss
import click

class PPOLoss:
    def __init__(self, clipcoef=.2, vfcoef=.5, entcoef=.1):
        self.clipcoef, self.vfcoef, self.entcoef = clipcoef, vfcoef, entcoef

    def __call__(self, advantages, returns, value, oldvalues, logprobs, oldlogprobs, entropy, **kwargs):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ploss = policy_loss(advantages, (logprobs - oldlogprobs).exp(), self.clipcoef)
        vloss = value_loss(value, oldvalues, returns, self.clipcoef)
        eloss = entropy.mean()
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
def ppo(trainer, clipcoef, vfcoef, entcoef):
    trainer.set_loss(
        PPOLoss(
            clipcoef = clipcoef,
            vfcoef   =   vfcoef,
            entcoef  =  entcoef
        )
    )
 
