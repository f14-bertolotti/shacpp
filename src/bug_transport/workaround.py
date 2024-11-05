from vmas.simulator.utils import Color
from vmas.scenarios import transport
import torch
import vmas

class Transport(transport.Scenario):

    def diffreward(self, prevs, nexts):
        prevs_dist_to_goal = [torch.linalg.vector_norm(prev[:,4:6], dim=-1) for prev in prevs]
        nexts_dist_to_goal = [torch.linalg.vector_norm(next[:,4:6], dim=-1) for next in nexts]
        rewards = [(prev_dist - next_dist)*100 for prev_dist, next_dist in zip(prevs_dist_to_goal, nexts_dist_to_goal)]
        return rewards

world = vmas.simulator.environment.Environment(
    Transport()               ,
    n_agents           = 3    ,
    num_envs           = 32   ,
    device             = "cpu",
    grad_enabled       = True ,
    continuous_actions = True ,
    seed               = 42   ,
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
action_cache = []
reward_cache = []
observ_cache = []
for step in range(16):
    actions = [policy(observation) for observation in observations]
    prev = observations
    observations, _, dones, _ = world.step(actions)
    rewards = world.scenario.diffreward(observations, prev)
 
    action_cache.append(actions)
    reward_cache.append(rewards)
    observ_cache.append(observations)

    for agent_obs in observations: agent_obs.retain_grad()
    for agent_rew in rewards     : agent_rew.retain_grad()
    for agent_act in actions     : agent_act.retain_grad()

    
loss = sum(rewards).sum()
loss.backward()
optimizer.step()

for observations in observ_cache: 
    for observation in observations: 
        print(observation.grad.mean().item() if observation.grad is not None else None, end=" ")
    print()


