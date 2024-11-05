#from vmas.simulator.utils import Color
#from vmas.scenarios import transport
import torch
import vmas

#class Transport(transport.Scenario):
#
#    def reward(self, agent):
#        print("reward")
#        is_first = agent == self.world.agents[0]
#
#        if is_first:
#            self.rew = torch.zeros(
#                self.world.batch_dim,
#                device=self.world.device,
#                dtype=torch.float32,
#                requires_grad=True,
#            )
#
#            for package in self.packages:
#                package.dist_to_goal = torch.linalg.vector_norm(
#                    package.state.pos - package.goal.state.pos, dim=1
#                )
#                package.on_goal = self.world.is_overlapping(package, package.goal)
#                package.color = torch.tensor(
#                    Color.RED.value,
#                    device=self.world.device,
#                    dtype=torch.float32,
#                ).repeat(self.world.batch_dim, 1)
#                package.color[package.on_goal] = torch.tensor(
#                    Color.GREEN.value,
#                    device=self.world.device,
#                    dtype=torch.float32,
#                )
#
#                package_shaping = package.dist_to_goal * self.shaping_factor
#                self.rew = self.rew.clone() # <--- This line is the workaround
#                self.rew[~package.on_goal] += (
#                    package.global_shaping[~package.on_goal]
#                    - package_shaping[~package.on_goal]
#                )
#                package.global_shaping = package_shaping
#
#        return self.rew
#
#env = vmas.simulator.environment.Environment(
#    Transport()               ,
#    n_agents           = 3    ,
#    num_envs           = 32   ,
#    device             = "cpu",
#    grad_enabled       = True ,
#    continuous_actions = True ,
#    seed               = 42   ,
#)

env = vmas.make_env(
    scenario           = "transport" ,
    num_envs           = 32          ,
    device             = "cuda:0"       ,
    continuous_actions = True        ,
    grad_enabled       = True        ,
    seed               = 42          ,
)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(11, 2)
        self.act = torch.nn.Tanh()
    def forward(self, obs):
        return self.act(self.lin(obs))

policy = NN().to("cuda:0")
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
optimizer.zero_grad()

### TRAINIG ###
observations  = env.reset()
for agent_observ in observations: 
    agent_observ.retain_grad()
rewards_cache = []
action_cache  = []
observ_cache  = [observations]
for agent_observ in observations: agent_observ.retain_grad()
for step in range(16):
    actions = [policy(observation) for observation in observations]
    observations, rewards, dones, _ = env.step(actions)
    
    #for agent_reward in rewards: print(agent_reward.grad_fn, end=' ')
    #for agent_action in actions     : agent_action.retain_grad()
    #for agent_reward in rewards     : agent_reward.retain_grad()
    #for agent_observ in observations: agent_observ.retain_grad()

    action_cache .append(actions)
    rewards_cache.append(rewards)
    observ_cache .append(observations)

loss = sum(observations).sum()
loss.backward()

for actions in action_cache:
    for agent_action in actions: 
        print(agent_action.grad, end=' ')

print("="*100)

for rewards in rewards_cache:
    for agent_reward in rewards: 
        print(agent_reward.grad, end=' ')

print("="*100)

for observations in observ_cache:
    for agent_observ in observations: 
        print(agent_observ.grad, end=' ')


