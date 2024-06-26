from algorithms.ppo import policy_loss, value_loss

def loss(new, old, clipcoef, vfcoef, entcoef):
    advantages = (old["advantages"] - old["advantages"].mean()) / (old["advantages"].std() + 1e-8)
    ploss = policy_loss(advantages, (new["logprobs"] - old["logprobs"]).exp(), clipcoef)
    vloss = value_loss(new["values"], old["values"], old["returns"], clipcoef)
    eloss = old["entropy"].mean()
    return ploss + vloss * vfcoef - entcoef * eloss 

