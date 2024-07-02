import torch

def value_loss(newvalue, oldvalues, returns, clipcoef):
    return 0.5 * ((newvalue - returns) ** 2).mean()


#def value_loss(newvalue, oldvalues, returns, clipcoef):
#    v_loss_unclipped = (newvalue - returns) ** 2
#    v_clipped = oldvalues + torch.clamp(newvalue - oldvalues, -clipcoef, clipcoef)
#    v_loss_clipped = (v_clipped - returns) ** 2
#    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
#    v_loss = 0.5 * v_loss_max.mean()
#    return v_loss
