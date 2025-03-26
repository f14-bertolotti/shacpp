import torch
def policy_loss(advantages, ratio, clipcoef):
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clipcoef, 1 + clipcoef)
    return torch.max(pg_loss1, pg_loss2).mean()


