import torch

@torch.no_grad()
def compute_advantages(next_obs, next_done, values, rewards, dones, agent, gamma, gaelambda, steps):
    next_value = agent.get_value(next_obs)["values"].flatten()
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    for t in reversed(range(steps)):

        if t == steps - 1:
            nextnonterminal = 1.0 - next_done.float()
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].float()
            nextvalues = values[t + 1]
    
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gaelambda * nextnonterminal * lastgaelam
    return advantages
