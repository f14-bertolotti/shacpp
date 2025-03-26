import torch

@torch.no_grad()
def compute_returns(advantages, values):
    return advantages + values


