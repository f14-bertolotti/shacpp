from trajectories import trajectory
import click, utils, torch

class PPOTrajectory:
    def __init__(self, steps=64, gamma=.99, gaelambda=.95):
        self.steps, self.gamma, self.gaelambda = steps, gamma, gaelambda
    
    def __call__(self, agent, environment, storage):
        observation = environment.reset()
        for step in range(0, self.steps):
        
            storage.observations[step] = observation
        
            with torch.no_grad():
                result = agent.get_action_and_value(observation = observation)
                storage.values[step] = result["value"]
            
            observation, reward = environment.step(observation, result["action"])
            storage.actions  [step] = result["action"]
            storage.logprobs [step] = result["logprobs"]
            storage.rewards  [step] = reward
        
        storage.advantages[:] = utils.compute_advantages(
            observation = observation,
            values      = storage.values,
            rewards     = storage.rewards,
            agent       = agent,
            gamma       = self.gamma,
            gaelambda   = self.gaelambda
        )
        
        storage.returns[:] = utils.compute_returns(
            values     = storage.values, 
            advantages = storage.advantages
        )
        
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
