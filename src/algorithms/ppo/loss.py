from algorithms.ppo import policy_loss, value_loss

def loss(new, old, clipcoef, vfcoef, entcoef, normadv = False):
    advantages = (old["advantages"] - old["advantages"].mean()) / (old["advantages"].std() + 1e-8)  if normadv else old["advantages"]
    ratio = (new["logprobs"] - old["logprobs"]).exp()

    ploss = policy_loss(advantages, ratio, clipcoef)
    vloss = value_loss(new["values"], old["values"], old["returns"], clipcoef)
    eloss = new["entropy"].mean()
    return ploss + vloss * vfcoef - entcoef * eloss 

