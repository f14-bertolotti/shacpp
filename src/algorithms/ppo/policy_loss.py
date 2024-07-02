import torch

def policy_loss(advantages, ratio, clipcoef):
    #loss1 = -advantages.unsqueeze(-1) * ratio
    #loss2 = -advantages.unsqueeze(-1) * torch.clamp(ratio, 1 - clipcoef, 1 + clipcoef)

    #loss1 = loss1.mean(-1)
    #loss2 = loss2.mean(-1)

    #loss = torch.max(loss1, loss2).mean()
    #return loss

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clipcoef, 1 + clipcoef)
    return torch.max(pg_loss1, pg_loss2).mean()
