from algorithms import Algorithm, algorithm
from algorithms.ppo import loss
from optimizers import add_adam_command
from schedulers import add_constant_command, add_cosine_command
import tqdm, time, click, math, torch

class PPO(Algorithm):
    def __init__(self, 
        trainer, 
        max_grad_norm = .5,
        epochs        = 64,
        batch_size    = 64,
        eval_steps    = 64,
        eval_size     = 512,
        clipcoef      = .2, 
        vfcoef        = .5, 
        entcoef       = .0,
        vclip         = False
    ): 
        self.trainer       = trainer
        self.max_grad_norm = max_grad_norm
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.vclip         = vclip
        self.clipcoef      = clipcoef
        self.vfcoef        = vfcoef
        self.entcoef       = entcoef
        self.eval_steps    = eval_steps
        self.eval_size     = eval_size
        self.start_time    = time.time()

    def set_optimizer (self, value): self.optimizer  = value
    def set_scheduler (self, value): self.scheduler  = value
    def set_trajectory(self, value): self.trajectory = value

    def start(self): pass
    def end  (self): pass

    def step (self, episode):

        trajectories = self.trajectory(
            agent       = self.trainer.agent,
            environment = self.trainer.environment,
        )

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                trajectories["observations"].flatten(0,1),
                trajectories["logprobs"    ].flatten(0,1),
                trajectories["actions"     ].flatten(0,1),
                trajectories["values"      ].flatten(0,1),
                trajectories["advantages"  ].flatten(0,1),
                trajectories["returns"     ].flatten(0,1),
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = self.batch_size,
            shuffle    = True
        )

        results = []
        for epoch in range(self.epochs):
            for step, (observations, logprobs, actions, values, advantages, returns) in enumerate(dataloader):
                self.optimizer.zero_grad()
        
                agent_result  = self.trainer.agent.get_action_and_value(observation = observations, action = actions)
        
                lossval = loss(
                    new_values   = agent_result["values"],
                    old_values   = values,
                    new_logprobs = agent_result["logprobs"],
                    old_logprobs = logprobs,
                    advantages   = advantages,
                    returns      = returns,
                    entropy      = agent_result["entropy"],
                    vclip        = self.vclip,
                    clipcoef     = self.clipcoef,
                    vfcoef       = self.vfcoef  ,
                    entcoef      = self.entcoef
                )

                lossval.backward()
                torch.nn.utils.clip_grad_norm_(self.trainer.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                results.append({
                    "episode" : episode,
                    "epoch"   : epoch,
                    "step"    : step,
                    "loss"    : lossval.item(),
                    "reward"  : trajectories["rewards"].sum().item() / trajectories["rewards"].size(0),
                    "lr"      : self.trainer.algorithm.optimizer.param_groups[0]["lr"],
                    "return"  : trajectories["returns"].mean().item(),
                    "time"    : time.time() - self.start_time
                })
        
            self.scheduler.step()

        return results

@algorithm.group(invoke_without_command=True)
@click.option("--batch-size"    , "batch_size"    , type=int   , default=256)
@click.option("--max-grad-norm" , "max_grad_norm" , type=float , default=.5)
@click.option("--epochs"        , "epochs"        , type=int   , default=4)
@click.option("--clip-coef"     , "clipcoef"      , type=float , default=.2)
@click.option("--vf-coef"       , "vfcoef"        , type=float , default=.5)
@click.option("--ent-coef"      , "entcoef"       , type=float , default=.0)
@click.option("--vclip"         , "vclip"         , type=bool  , default=None, help="True if clipping is to be applied to the value function")
@click.option("--eval-steps"    , "eval_steps"    , type=int   , default=64)
@click.option("--eval-size"     , "eval_size"     , type=int   , default=512, help="number of runs for evaluation")
@click.pass_obj
def ppo(trainer, batch_size, epochs, eval_size, eval_steps, max_grad_norm, clipcoef, vfcoef, entcoef, vclip):
    if not hasattr(trainer, "algorithm"):
        trainer.set_algorithm(
            PPO(
                trainer       = trainer,
                batch_size    = batch_size,
                epochs        = epochs,
                clipcoef      = clipcoef,
                vfcoef        = vfcoef,
                entcoef       = entcoef,
                vclip         = vclip,
                eval_steps    = eval_steps,
                eval_size     = eval_size,
                max_grad_norm = max_grad_norm
            )
        )

@ppo.group()
def optimizer(): pass
add_adam_command(optimizer, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.agent)

@ppo.group()
def scheduler(): pass
add_cosine_command  (scheduler, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.algorithm)
add_constant_command(scheduler, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.algorithm)

