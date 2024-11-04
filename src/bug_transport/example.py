import vmas
import torch
torch.autograd.set_detect_anomaly(True)

env = vmas.make_env(
    scenario           = "transport" ,
    n_agents           = 3           ,
    num_envs           = 32          ,
    max_steps          = 32          ,
    seed               = 42          ,
    device             = "cpu"       ,
    continuous_actions = True        ,
    grad_enabled       = True        ,
)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(11, 2)
        self.act = torch.nn.Tanh()
    def forward(self, obs):
        return self.act(self.lin(obs))

policy = NN()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)


### TRAINIG ###
#env.scenario.reset_world_at(None)
observations = env.reset()
for step in range(16):
    actions = [policy(observation) for observation in observations]
    observations, rewards, dones, _ = env.step(actions)
    observations = torch.stack(observations)

loss = -sum(rewards).sum()
loss.backward()
optimizer.step()

##### NEXT STEP ###
#del loss, step2reward, observations, rewards, actions, step, _, dones
#print(locals().keys())
#optimizer.zero_grad()
###env.world.zero_grad()
###observations = [observation.detach() for observation in observations]
#observations = env.reset()
#step2reward = []
#
#for step in range(32):
#    actions = [policy(observation) for observation in observations]
#    observations, rewards, dones, _ = env.step(actions)
#    print(step, rewards[0].grad_fn)
#    step2reward.append(torch.stack(rewards))
#
#loss = -torch.stack(step2reward).sum()
#loss.backward()
#optimizer.step()


