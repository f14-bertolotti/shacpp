import torch

min_reward = float('+inf')
max_reward = float('-inf')
@torch.no_grad()
def bin_dispatch(rewards:torch.Tensor, bins:int, size:int) -> torch.Tensor:
    global min_reward, max_reward
    min_reward = min(min_reward, rewards.min().item())
    max_reward = max(max_reward, rewards.max().item())
    rewards = (rewards - min_reward) / (max_reward - min_reward + 1e-5)
    bin_idx = (rewards * (bins-1)).round()
    cache_idx = bin_idx * size + torch.randint(0, size, (rewards.size(0),), device=rewards.device)
    return cache_idx.to(torch.long)
 
