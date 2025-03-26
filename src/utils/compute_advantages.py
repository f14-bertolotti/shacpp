import torch

@torch.no_grad
def compute_advantages(value_model, rewards, next_obs, values, dones, next_done, gamma=.99, gaelambda=.95):

    next_value = value_model(next_obs)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    for t in reversed(range(rewards.size(0))):
        if t == rewards.size(0) - 1:
            nextnonterminal = 1.0 - next_done.float()
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].float()
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gaelambda * nextnonterminal * lastgaelam

    return advantages


