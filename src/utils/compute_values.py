import torch

@torch.no_grad()
def compute_values(values, rewards, dones, slam, gamma):
    steps, envirs, agents = values.size(0), values.size(1), values.size(2)

    target_values = torch.zeros(steps, envirs, agents, dtype=torch.float32, device=values.device)
    Ai = torch.zeros(envirs, agents, dtype=torch.float32, device=values.device)
    Bi = torch.zeros(envirs, agents, dtype=torch.float32, device=values.device)
    lam = torch.ones(envirs, agents, dtype=torch.float32, device=values.device)

    for i in reversed(range(steps)):
        lam = slam * lam * (1. - dones[i]) + dones[i]
        Ai = (1.0 - dones[i]) * (slam * gamma * Ai + gamma * values[i] + (1. - lam) / (1. - slam) * rewards[i])
        Bi = gamma * (values[i] * dones[i] + Bi * (1.0 - dones[i])) + rewards[i]
        target_values[i] = (1.0 - slam) * Ai + lam * Bi

    target_values = target_values

    return target_values


