
import vmas
import torch

env = vmas.make_env(
    scenario           = "transport" ,
    num_envs           = 32          ,
    device             = "cpu"       ,
    continuous_actions = True        ,
    seed               = 42          ,
    grad_enabled       = True        ,
)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(11, 2)
    def forward(self, obs):
        return self.lin(obs)

policy = NN()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

observations = env.reset()
rewards_by_step = []
for step in range(32):
    actions = [policy(observation) for observation in observations]
    observations, rewards, dones, _ = env.step(actions)
    rewards_by_step.append(sum(rewards).sum())

loss = -sum(rewards_by_step)
loss.backward()
optimizer.step()

### NEXT STEP ###
optimizer.zero_grad()
env.world.zero_grad()
observations = [observation.detach() for observation in observations]

### WORKAROUND ###
for package in env.scenario.packages:
    package.global_shaping = (
        torch.linalg.vector_norm(
            package.state.pos - package.goal.state.pos, dim=1
        )
        * env.scenario.shaping_factor
    )
##################

rewards_by_step.clear()

for step in range(32):
    actions = [policy(observation) for observation in observations]
    observations, rewards, dones, _ = env.step(actions)
    rewards_by_step.append(sum(rewards).sum())

loss = -sum(rewards_by_step)
loss.backward()
