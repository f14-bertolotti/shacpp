from algorithms import Algorithm, algorithm
from algorithms.ppo import loss
from optimizers import add_adam_command
from schedulers import add_constant_command, add_cosine_command
import click, torch

class PPO(Algorithm):
    def __init__(self, 
        trainer, 
        max_grad_norm = .5,
        epochs        = 64,
        batch_size    = 64,
        clipcoef      = .2, 
        vfcoef        = .5, 
        entcoef       = .1
    ): 
        self.trainer       = trainer
        self.max_grad_norm = max_grad_norm
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.clipcoef      = clipcoef
        self.vfcoef        = vfcoef
        self.entcoef       = entcoef

    def set_optimizer (self, value): self.optimizer  = value
    def set_scheduler (self, value): self.scheduler  = value
    def set_trajectory(self, value): self.trajectory = value

    def start(self): pass
    def end  (self): pass

    def step (self, update, tbar):

        storage = self.trajectory(
            agent       = self.trainer.agent,
            environment = self.trainer.environment,
            storage     = self.trainer.storage
        )

        dataloader = torch.utils.data.DataLoader(
            storage,
            collate_fn = storage.collate_fn,
            batch_size = self.batch_size,
            shuffle    = True
        )
        
        for step in range(self.epochs):
            for epoch, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
        
                result  = self.trainer.agent.get_action_and_value(observation = batch["observations"], action = batch["actions"])
                result  = {"actions" : result[0], "logprobs":result[1], "entropy":result[2], "values":result[3].squeeze(-1)}
                
        
                lossval = loss(new = result, old = batch, clipcoef = self.clipcoef, vfcoef = self.vfcoef, entcoef = self.entcoef)
                lossval.backward()

                torch.nn.utils.clip_grad_norm_(self.trainer.agent.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
                for callback in self.trainer.callbacks: callback(**{k:v for k,v in locals().items() if k != "self"})
        
            self.scheduler.step()


@algorithm.group(invoke_without_command=True)
@click.option("--batch-size"    , "batch_size"    , type=int   , default=256)
@click.option("--max-grad-norm" , "max_grad_norm" , type=float , default=1)
@click.option("--epochs"        , "epochs"        , type=int   , default=4)
@click.option("--clip-coef"     , "clipcoef"      , type=float , default=.2)
@click.option("--vf-coef"       , "vfcoef"        , type=float , default=.5)
@click.option("--ent-coef"      , "entcoef"       , type=float , default=0)
@click.pass_obj
def ppo(trainer, batch_size, epochs, max_grad_norm, clipcoef, vfcoef, entcoef):
    if not hasattr(trainer, "algorithm"):
        trainer.set_algorithm(
            PPO(
                trainer       = trainer,
                batch_size    = batch_size,
                epochs        = epochs,
                clipcoef      = clipcoef,
                vfcoef        = vfcoef,
                entcoef       = entcoef,
                max_grad_norm = max_grad_norm
            )
        )

@ppo.group()
def optimizer(): pass
add_adam_command(optimizer, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.agent)

@ppo.group()
def scheduler(): pass
add_cosine_command(scheduler, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.algorithm)
add_constant_command(scheduler, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.algorithm)

