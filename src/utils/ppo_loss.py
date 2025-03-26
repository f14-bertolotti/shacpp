import torch
import utils

def ppo_loss(new_values, old_values, new_logprobs, old_logprobs, advantages, returns, entropy, vclip, clipcoef, vfcoef, entcoef, normadv = True):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) if normadv else advantages
    ratio = (new_logprobs - old_logprobs).exp()

    ploss = utils.policy_loss(advantages, ratio, clipcoef)
    vloss = utils.value_loss(new_values, old_values, returns, clipcoef if vclip else None)
    eloss = entropy.mean()

    return ploss + vloss * vfcoef - entcoef * eloss 


