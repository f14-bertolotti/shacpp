import torch

@torch.no_grad()
def compute_advantages(observation, values, rewards, agent, gamma, gaelambda):
    advantages = torch.zeros_like(rewards)
    steps      = rewards.size(0)

    lastgaelam = 0
    next_value = agent.get_value(observation)["values"].reshape(1, -1)
    for t in reversed(range(steps)):
        nextvalues = next_value if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalues - values[t]
        advantages[t] = lastgaelam = delta + gamma * gaelambda  * lastgaelam

    return advantages


