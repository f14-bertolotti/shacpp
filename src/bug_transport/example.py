from vmas.scenarios import transport
import torch
import vmas

world = vmas.simulator.environment.Environment(
    transport.Scenario() ,
    n_agents           = 3     ,
    num_envs           = 32    ,
    device             = "cpu" ,
    grad_enabled       = True  ,
    continuous_actions = True  ,
    seed               = 42    ,
)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(11,2)
        self.act = torch.nn.Tanh()
    def forward(self, obs):
        return self.act(self.lin(obs))

policy = NN()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
optimizer.zero_grad()

observations = world.reset()
for step in range(16):
    actions = [policy(observation) for observation in observations]
    observations, rewards, dones, _ = world.step(actions)
 
    for agent_observ in observations: 
        agent_observ.retain_grad()

    # RuntimeError: can't retain_grad on Tensor that has requires_grad=False
    for agent_reward in rewards: 
        agent_reward.retain_grad()

