import torch

@torch.no_grad()
def compute_returns(values, advantages):
    return advantages + values
