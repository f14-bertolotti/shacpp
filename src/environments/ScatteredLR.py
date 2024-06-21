from environments.rewards import DummyReward
from environments import environment
import torch, click



class ScatteredLR:
    def __init__(self, envs=64, agents=9, device="cuda:0"):
        self.envs, self.agents, self.device = envs, agents, device
        self.points = torch.tensor([[(i/9)*2-1,(j/9)*2-1] for j in range(10) for i in range(10)],dtype=torch.float32,device=device)
        self.loss = torch.nn.MSELoss()
        self.rewardnn = DummyReward()

    def set_reward_nn(self, value):
        self.rewardnn = value
        self.optimizer = torch.optim.Adam(self.rewardnn.parameters(), lr=0.0001)


    def reset(self):
        return torch.rand((self.envs,self.agents,2), device=self.device)*2-1

    def step(self, observation, action):
        next_observation = observation + action
        return {
            "observations"      : observation, 
            "next_observations" : next_observation, 
            "rewards"           : self.rewardnn(next_observation),
            "real_rewards"      : self.reward(next_observation),
        }

    def reward(self, observation):
        dists = torch.cdist(self.points, observation)
        return (dists.min(-1).values  < 0.333).float().mean(-1) - \
               (dists.min(-1).values >= 0.333).float().mean(-1)

    def train_step(self, next_observations, epochs=1, **kwargs):
        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            mask = (torch.randperm(next_observations.size(0)) < int(next_observations.size(0) * .2)).to(self.device)
            batch = next_observations.clone().to(self.device)
            batch[mask] = torch.rand(int(next_observations.size(0) * .2), self.agents, 2, device=self.device)*2-1
            loss = self.loss(self.reward(batch), self.rewardnn(batch))
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        return sum(losses)/len(losses)



@environment.group(invoke_without_command=True)
@click.option("--envs"   , "envs"   , type=int , default=64)
@click.option("--agents" , "agents" , type=int , default=9)
@click.option("--device" , "device" , type=str , default="cuda:0")
@click.pass_obj
def scattered_learnable_reward(trainer, envs, agents, device):
    trainer.set_environment(
        ScatteredLR(
            envs   =   envs,
            agents = agents,
            device = device
        )
    )


