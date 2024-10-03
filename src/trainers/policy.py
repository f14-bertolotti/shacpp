import logging
import torch
import json

def train_policy(
        episode          : int                    ,
        policy_model     : torch.nn.Module        ,
        episode_data     : dict[str, torch.Tensor],
        optimizer        : torch.optim.Optimizer  ,
        gammas           : torch.Tensor           ,
        logger           : logging.Logger         ,
        clip_coefficient : float = .5             ,
    ):
    """ Train the policy model """

    steps, envs = episode_data["observations"].size(0), episode_data["observations"].size(1)
    
    # compute value cache mask
    dead_runs = episode_data["dones"][0,:,0]
    live_runs = dead_runs.logical_not()
    live_steps = episode_data["dones"][:,live_runs,0].logical_not().sum(0) - 1
    
    # compute loss
    optimizer.zero_grad()
    loss = -((episode_data["proxy_rewards"] * gammas * episode_data["dones"].logical_not()).sum() + ((gammas * episode_data["values"])[live_steps,live_runs]).sum()) / (steps * envs)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), clip_coefficient)
    optimizer.step()

    logger.info(json.dumps({
            "episode" : episode,
            "loss"    : loss.item(),
            "done"    : episode_data["dones"][-1,:,0].sum().int().item(),
            "reward"  : episode_data["rewards"].sum(0).mean(0).sum().item()
    }))


