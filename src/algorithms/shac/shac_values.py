import torch

def compute_shac_values(steps, envirs, values, agents, rewards, slam, gamma, device="cuda:0"):
    print(values.shape, rewards.shape)
    target_values = torch.zeros(steps, envirs, agents, dtype=torch.float32, device=device)
    Ai = torch.zeros(envirs, agents, dtype=torch.float32, device=device)
    Bi = torch.zeros(envirs, agents, dtype=torch.float32, device=device)
    lam = torch.ones(envirs, agents, dtype=torch.float32, device=device)

    for i in reversed(range(steps)):
        lam = lam * slam
        Ai = (slam * gamma * Ai + gamma * values[i] + (1. - lam) / (1. - slam) * rewards[i])
        Bi = Bi + rewards[i]
        target_values[i] = (1.0 - slam) * Ai + lam * Bi

    return target_values
