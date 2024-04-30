from environments import environment
import torch, click

class Reward(torch.nn.Module):
    def __init__(self, layers = 3, embedding_size = 64, heads=2, feedforward_size=256, activation="gelu", device="cuda:0"):
        super().__init__()
        self.embedding = torch.nn.Linear(2, embedding_size, device=device)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model         = embedding_size,
                nhead           = heads,
                dim_feedforward = feedforward_size,
                activation      = activation,
                device          = device,
                batch_first     = True
            ), 
            num_layers = layers
        )
        self.fin       = torch.nn.Linear(embedding_size, 1, device=device)

    def forward(self, observation):
        embeddings = self.embedding(observation)
        encoded = self.encoder(embeddings).mean(1)
        reward = self.fin(encoded)
        return reward.squeeze(-1)
        

class ScatteredLR:
    def __init__(self, envs=64, agents=9, device="cuda:0"):
        self.envs, self.agents, self.device = envs, agents, device
        self.points = torch.tensor([[(i/9)*2-1,(j/9)*2-1] for j in range(10) for i in range(10)],dtype=torch.float32,device=device)
        self.rewardnn = Reward(device=device)
        self.optimizer = torch.optim.Adam(self.rewardnn.parameters(), lr=0.0001)
        self.loss = torch.nn.MSELoss()

    def reset(self):
        return torch.rand((self.envs,self.agents,2), device=self.device)*2-1

    def step(self, observation, action):
        next_observation = observation + action
        return {
            "observations"      : observation, 
            "next_observations" : next_observation, 
            "rewards"           : self.rewardnn(next_observation).detach(),
            "real_rewards"      : self.reward(next_observation),
        }

    def reward(self, observation):
        dists = torch.cdist(self.points, observation)
        return (dists.min(-1).values  < 0.333).float().mean(-1) - \
               (dists.min(-1).values >= 0.333).float().mean(-1)

    def train_step(self, next_observations, **kwargs):
        self.optimizer.zero_grad()
        loss = self.loss(self.reward(next_observations), self.rewardnn(next_observations))
        loss.backward()
        self.optimizer.step()
        return loss



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


