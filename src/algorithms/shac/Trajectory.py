from algorithms.shac import compute_shac_values, shac
from algorithms import Options
from optimizers import add_adam_command
from schedulers import add_constant_command, add_cosine_command
import click, torch

from utils import hash_tensors

class Trajectory:

    def __init__(self, trainer, gamma=.99, gaelm=.95, steps=64, utr=1, max_grad_norm=1):

        self.actor_optimizer = trainer.algorithm.actor_optimizer
        self.actor_scheduler = trainer.algorithm.actor_scheduler

        self.device = trainer.environment.device
        self.envirs = trainer.environment.envirs
        self.agents = trainer.environment.agents

        # compute gammas vector ###################
        # self.gammas = [[1],[γ],[γ²],[γ³],...] ###
        self.gammas = torch.ones(steps, device=self.device, dtype=torch.float)
        self.gammas[1:] = gamma
        self.gammas = self.gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.agents)

        self.max_grad_norm = max_grad_norm
        self.gamma         =         gamma
        self.gaelm         =         gaelm
        self.steps         =         steps
        self.utr           =           utr
        self.rollouts      =             0

        self.obs      = []
        self.actions  = []
        self.rewards  = []
        self.dones    = []
        self.values   = []

    @torch.no_grad()
    def reset_storage(self):
        """ reset storage into a full zero state """
        self.obs     .clear()
        self.actions .clear()
        self.rewards .clear()
        self.dones   .clear()
        self.values  .clear()

    def __call__(self, agent, target_agent, environment):

        environment.world.world.zero_grad()
        next_obs  = environment.reset().detach()                 if self.rollouts % self.utr == 0 else self.obs  [-1].detach().clone()
        next_done = torch.zeros(self.envirs, device=self.device) if self.rollouts % self.utr == 0 else self.dones[-1].detach().clone()
        self.rollouts += 1
        self.reset_storage()

        # unroll trajectories
        for step in range(0, self.steps):
            self.obs  .append(next_obs )
            self.dones.append(next_done)
            
            actor_result  = agent.get_action(next_obs)
            critic_result = target_agent.get_value(next_obs)

            self.values  .append(critic_result["values"  ])
            self.actions .append( actor_result["actions" ])
            envir_result = environment.step(oldobs=self.obs[-1], action=actor_result["actions"])
            self.rewards .append(envir_result["reward"  ])

            next_obs, next_done = envir_result["observation"], envir_result["done"]


        rewards = torch.stack(self.rewards)
        values  = torch.stack(self.values)
        actions = torch.stack(self.actions)
        obs     = torch.stack(self.obs)
        dones   = torch.stack(self.dones)

        obs = environment.normalize(obs)

        target_values = compute_shac_values(
             steps   = self.steps  ,
             envirs  = self.envirs ,
             agents  = self.agents ,
             values  = values      ,
             rewards = rewards     ,
             slam    = self.gaelm  ,
             gamma   = self.gamma  ,
             device  = self.device
         )

            
        self.actor_optimizer.zero_grad()
        loss = -(
            (rewards * self.gammas).sum(0) + 
            (self.gamma ** self.steps) * values
        ).sum() / (self.steps * self.envirs)


        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_scheduler.step()


        environment.update_statistics(obs)

        return {
            "observations"  : obs           ,
            "rewards"       : rewards       ,
            "actions"       : actions       ,
            "values"        : values        ,
            "target_values" : target_values ,
            "dones"         : self.dones    ,
            "loss"          : loss.item()
        }

@shac.group()
def trajectory(): pass
 
@trajectory.group(invoke_without_command=True)
@Options.trajectory
@click.option("--max-grad-norm" , "max_grad_norm" , type=float , default=1)
@click.pass_obj
def default(trainer, gamma, gaelambda, steps, utr, max_grad_norm):
    trainer.algorithm.set_trajectory(
        Trajectory( 
            trainer       = trainer       ,
            steps         = steps         ,
            gamma         = gamma         ,
            gaelm         = gaelambda     ,
            utr           = utr           ,
            max_grad_norm = max_grad_norm ,
        )
    )


