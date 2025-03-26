import torch

@torch.no_grad()
def random_dispatch(rewards, size, lamb=8):
    beta  = (1 / (rewards + 1/lamb) + 1e-5) * torch.ones_like(rewards)
    alpha = (lamb/beta + 1e-5) * torch.ones_like(rewards)
    result = (torch.distributions.Beta(alpha, beta).sample() * (size-1)).round().to(torch.long)
    return result


