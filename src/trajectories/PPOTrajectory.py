from trajectories import trajectory
import click, utils, torch

class PPOTrajectory:
    def __init__(self, steps=64, gamma=.99, gaelambda=.95):
        self.steps, self.gamma, self.gaelambda = steps, gamma, gaelambda
    
    def __call__(self, agent, environment, storage):
        storage.clear()
        observation = environment.reset()
        for _ in range(0, self.steps):
        
            with torch.no_grad():
                agent_result = agent.get_action_and_value(observation = observation)
            
            envir_result = environment.step(observation, agent_result["actions"])
            observation = envir_result["next_observations"]
            storage.append(envir_result)
            storage.append(agent_result)

        storage.stack()
        
        storage.dictionary["advantages"] = utils.compute_advantages(
            observation = observation,
            values      = storage.dictionary["values"],
            rewards     = storage.dictionary["rewards"],
            agent       = agent,
            gamma       = self.gamma,
            gaelambda   = self.gaelambda
        )
        
        storage.dictionary["returns"] = utils.compute_returns(
            values     = storage.dictionary["values"], 
            advantages = storage.dictionary["advantages"]
        )

        storage.flatten()

        return storage


@trajectory.group(invoke_without_command=True)
@click.option("--gamma"     , "gamma"     , type=float , default=.99)
@click.option("--gaelambda" , "gaelambda" , type=float , default=.95)
@click.option("--steps"     , "steps"     , type=int   , default=64)
@click.pass_obj
def ppo(trainer, gamma, gaelambda, steps):
    trainer.set_trajectory(
        PPOTrajectory( 
            steps     = steps,
            gamma     = gamma,
            gaelambda = gaelambda
        )
    )
