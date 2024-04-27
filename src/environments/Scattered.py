from environments import environment
import torch, click

class Scattered:
    def __init__(self, envs=64, agents=9, device="cuda:0"):
        self.envs, self.agents, self.device = envs, agents, device
        self.points = torch.tensor([[(i/9)*20-10,(j/9)*20-10] for j in range(10) for i in range(10)],dtype=torch.float32,device=device)

    def reset(self):
        return torch.rand((self.envs,self.agents,2), device=self.device)*20-10

    def step(self, observation, action):
        next_observation = observation + action
        return next_observation, self.reward(next_observation)

    def reward(self, observation):
        dists = torch.cdist(self.points, observation)
        return (dists.min(-1).values  < 3.333).float().mean(-1) - \
               (dists.min(-1).values >= 3.333).float().mean(-1)



@environment.group(invoke_without_command=True)
@click.option("--envs"   , "envs"   , type=int , default=64)
@click.option("--agents" , "agents" , type=int , default=9)
@click.option("--device" , "device" , type=str , default="cuda:0")
@click.pass_obj
def scattered(trainer, envs, agents, device):
    trainer.set_environment(
        Scattered(
            envs   =   envs,
            agents = agents,
            device = device
        )
    )


