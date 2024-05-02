from trajectories import trajectory
import click, utils, torch

class SHACTrajectory:
    def __init__(self, trainer, steps=64, gamma=.99, gaelambda=.95, device="cuda:0"):
        self.steps, self.gamma, self.gaelambda, self.device = steps, gamma, gaelambda, device
        self.gammas = torch.ones(steps, device=device, dtype=torch.float)
        self.gammas[1:] = gamma
        self.gammas = self.gammas.cumprod(0).unsqueeze(-1)
        self.optimizer = torch.optim.Adam(trainer.agent.actor.parameters(), lr=0.0001)

    def __call__(self, agent, environment, storage, prev_trajectories=None):
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
        loss = -((storage.dictionary["rewards"] * self.gammas).sum(1) + (self.gamma ** self.steps) * agent_result["values"]).sum() / (self.steps * environment.envs)
        loss.backward()
        self.optimizer.step()
        
        storage.flatten().detach()

        return {
            "storage" : storage, 
            "loss"    : loss
        }
            
@trajectory.group(invoke_without_command=True)
@click.option("--gamma"     , "gamma"     , type=float , default=.99)
@click.option("--gaelambda" , "gaelambda" , type=float , default=.95)
@click.option("--steps"     , "steps"     , type=int   , default=64)
@click.option("--device"    , "device"    , type=str   , default="cuda:0")
@click.pass_obj
def shac(trainer, gamma, gaelambda, steps, device):
    trainer.set_trajectory(
        SHACTrajectory( 
            trainer   = trainer,
            steps     = steps,
            gamma     = gamma,
            gaelambda = gaelambda,
            device    = device
        )
    )
