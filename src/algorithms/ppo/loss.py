from algorithms.ppo import policy_loss, value_loss

def loss(new_values, old_values, new_logprobs, old_logprobs, advantages, returns, entropy, vclip, clipcoef, vfcoef, entcoef, normadv = False):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  if normadv else advantages
    ratio = (new_logprobs - old_logprobs).exp()

    ploss = policy_loss(advantages, ratio, clipcoef)
    vloss = value_loss(new_values, old_values, returns, clipcoef if vclip else None)
    eloss = entropy.mean()
    return ploss + vloss * vfcoef - entcoef * eloss 

