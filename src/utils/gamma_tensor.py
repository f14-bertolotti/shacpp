import torch

def gamma_tensor(train_steps, train_envs, agents, gamma_factor):    
    """ returns a tensor of shape (train_steps, train_envs, agents) with the gamma factors """
    gammas = torch.ones(train_steps, dtype=torch.float)
    gammas[1:] = gamma_factor
    gammas = gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,train_envs,agents)
    return gammas

