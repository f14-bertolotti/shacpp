from algorithms.shac import shac
import utils, click, torch
from optimizers import add_adam_command
from schedulers import add_constant_command, add_cosine_command

class Trajectory:
    def __init__(self, trainer, steps=64, gamma=.99, gaelambda=.95, max_grad_norm=1, device="cuda:0"):
        self.steps, self.gamma, self.gaelambda, self.max_grad_norm, self.device = steps, gamma, gaelambda, max_grad_norm, device

        # compute gammas vector ###################
        # self.gammas = [[1],[γ],[γ²],[γ³],...] ###
        self.gammas = torch.ones(steps, device=device, dtype=torch.float)
        self.gammas[1:] = gamma
        self.gammas = self.gammas.cumprod(0).unsqueeze(-1)

    def set_optimizer(self, value): self.optimizer = value
    def set_scheduler(self, value): self.scheduler = value

    def __call__(self, agent, environment, storage):
        storage.clear()
 
        observation = environment.reset()
        for _ in range(0, self.steps):

            agent_result = agent.get_action_and_value(observation = observation)

            envir_result = environment.step(observation, agent_result["actions"])
            observation = envir_result["next_observations"]
            storage.append(envir_result)
            storage.append(agent_result)
            
        storage.stack() 

        storage.dictionary["target_values"] = utils.compute_shac_values(
            steps   = self.steps,
            envs    = environment.envs,
            values  = storage.dictionary["values"],
            rewards = storage.dictionary["rewards"],
            slam    = self.gaelambda,
            gamma   = self.gamma,
            device  = self.device
        )
        self.optimizer.zero_grad()
        loss = -(
            (storage.dictionary["rewards"] * self.gammas).sum(0) + 
            (self.gamma ** self.steps) * agent_result["values"]
        ).sum() / (self.steps * environment.envs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        
        return storage.flatten().detach(), loss

@shac.group()
def trajectory(): pass
 
@trajectory.group(invoke_without_command=True)
@click.option("--gamma"         , "gamma"         , type=float , default=.99)
@click.option("--gaelambda"     , "gaelambda"     , type=float , default=.95)
@click.option("--steps"         , "steps"         , type=int   , default=64)
@click.option("--max-grad-norm" , "max_grad_norm" , type=float , default=1)
@click.option("--device"        , "device"        , type=str   , default="cuda:0")
@click.pass_obj
def default(trainer, gamma, gaelambda, steps, max_grad_norm, device):
    trainer.algorithm.set_trajectory(
        Trajectory( 
            trainer       = trainer,
            steps         = steps,
            gamma         = gamma,
            gaelambda     = gaelambda,
            max_grad_norm = max_grad_norm,
            device        = device
        )
    )

@trajectory.group()
def optimizer(): pass
add_adam_command(optimizer, srcnav=lambda x:x.algorithm.trajectory, tgtnav=lambda x:x.agent.actor)

@trajectory.group()
def scheduler(): pass
add_cosine_command(scheduler, srcnav=lambda x:x.algorithm.trajectory, tgtnav=lambda x:x.algorithm.trajectory)
add_constant_command(scheduler, srcnav=lambda x:x.algorithm.trajectory, tgtnav=lambda x:x.algorithm.trajectory)


