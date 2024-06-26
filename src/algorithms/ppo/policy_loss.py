import torch

def policy_loss(advantages, ratio, clipcoef):
    loss1 = -advantages * ratio
    loss2 = -advantages * torch.clamp(ratio, 1 - clipcoef, 1 + clipcoef)
    loss = torch.max(loss1, loss2).mean()
    return loss


