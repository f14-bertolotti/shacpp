import torch

@torch.no_grad
def compute_advantages(agent, rewards, next_obs, values, dones, next_done, gamma=.99, gaelambda=.95):
    steps = rewards.size(1)

    next_value = agent.get_value(next_obs)["values"]
    advantages = torch.zeros_like(rewards).to("cuda:0")
    lastgaelam = 0
    dones = dones.unsqueeze(-1).repeat(1,1,3)
    next_done = next_done.unsqueeze(-1).repeat(1,3)

    for t in reversed(range(steps)):
        if t == steps - 1:
            nextnonterminal = 1.0 - next_done.float()
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[:, t + 1].float()
            nextvalues = values[:, t + 1]
    
        delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:, t]

        advantages[:, t] = lastgaelam = delta + gamma * gaelambda * nextnonterminal * lastgaelam

    return advantages

        

