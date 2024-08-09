import random, numpy, torch

def layer_init(layer, std=1.141, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None: torch.nn.init.constant_(layer.bias, bias_const)
    return layer

@torch.no_grad()
def pert(low:torch.Tensor, peak:torch.Tensor, high:torch.Tensor, lamb:int=8):
    """ pert distribution   : https://en.wikipedia.org/wiki/PERT_distribution 
        implementation from : https://stackoverflow.com/questions/68476485/random-values-from-a-pert-distribution-in-python """
    r = high - low
    alpha = 1 + lamb * (peak - low) / r
    beta  = 1 + lamb * (high - peak) / r
    return low + torch.distributions.Beta(alpha, beta).sample() * r


def compute_values(steps, envirs, values, dones, agents, rewards, slam, gamma, device="cuda:0"):
    values, rewards = values.squeeze(-1), rewards.squeeze(-1)

    target_values = torch.zeros(steps, envirs, agents, dtype=torch.float32, device=device)
    Ai = torch.zeros(envirs, agents, dtype=torch.float32, device=device)
    Bi = torch.zeros(envirs, agents, dtype=torch.float32, device=device)
    lam = torch.ones(envirs, agents, dtype=torch.float32, device=device)

    for i in reversed(range(steps)):
        #lam = lam * slam
        #Ai = (slam * gamma * Ai + gamma * values[i] + (1. - lam) / (1. - slam) * rewards[i])
        #Bi = Bi + rewards[i]
        #target_values[i] = (1.0 - slam) * Ai + lam * Bi
        lam = slam * lam * (1. - dones[i]) + dones[i]
        Ai = (1.0 - dones[i]) * (slam * gamma * Ai + gamma * values[i] + (1. - lam) / (1. - slam) * rewards[i])
        Bi = gamma * (values[i] * dones[i] + Bi * (1.0 - dones[i])) + rewards[i]
        target_values[i] = (1.0 - slam) * Ai + lam * Bi

    target_values = target_values.unsqueeze(-1)

    return target_values


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): 
        return self.func(x)

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv
