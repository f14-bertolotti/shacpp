import torch

@torch.no_grad()
def pert(low:torch.Tensor, peak:torch.Tensor, high:torch.Tensor, lamb:int=8):
    """ pert distribution   : https://en.wikipedia.org/wiki/PERT_distribution 
        implementation from : https://stackoverflow.com/questions/68476485/random-values-from-a-pert-distribution-in-python """
    r = high - low
    alpha = 1 + lamb * (peak - low) / r
    beta  = 1 + lamb * (high - peak) / r
    return low + torch.distributions.Beta(alpha, beta).sample() * r


