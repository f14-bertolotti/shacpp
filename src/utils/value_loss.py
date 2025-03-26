import torch

def value_loss(newvalue, oldvalues, returns, clipcoef=None):
    v_loss_unclipped = (newvalue - returns) ** 2
    if clipcoef is not None:
        v_clipped = oldvalues + torch.clamp(newvalue - oldvalues, -clipcoef, clipcoef)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        return 0.5 * v_loss_max.mean()
    return 0.5 * v_loss_unclipped.mean()


