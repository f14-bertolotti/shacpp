import torch

def is_early_stopping(eval_reward:torch.Tensor, max_reward:torch.Tensor, max_reward_fraction:float, max_envs_fraction:float) -> bool:
    """ Check if the early stopping condition is met """
    admissable_reward = max_reward * max_reward_fraction
    admissable_environments = (eval_reward.sum(0).sum(1) >= admissable_reward).sum().item()
    return admissable_environments >= max_envs_fraction * eval_reward.size(1)

