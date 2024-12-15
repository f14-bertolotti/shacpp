import logging
import models
import torch
import json

def train_policy(
        episode          : int                    ,
        policy_model     : models.Model           ,
        episode_data     : dict[str, torch.Tensor],
        optimizer        : torch.optim.Optimizer  ,
        gammas           : torch.Tensor           ,
        logger           : logging.Logger         ,
        clip_coefficient : float|None = .5             ,
    ):
    """ Train the policy model """

    steps, envs = episode_data["observations"].size(0), episode_data["observations"].size(1)
    
    # compute value cache mask
    dones = episode_data["dones"][:-1]
    dead_runs = dones[0,:,0]
    live_runs = dead_runs.logical_not()
    live_steps = dones[:,live_runs,0].logical_not().sum(0) - 1

    # compute loss
    optimizer.zero_grad()
    loss = -(
        (episode_data["proxy_rewards"] * gammas * dones.logical_not()).sum() + 
        ((gammas * episode_data["values"])[live_steps,live_runs]
    ).sum()) / (steps * envs)

    # add outside action range loss
    loss += .1 * ((episode_data["logits"][episode_data["logits"].abs()>1]**2)-1).mean()

    # backward pass
    loss.backward()
    if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(policy_model.parameters(), clip_coefficient)
    optimizer.step()

    logger.info(json.dumps({
            "episode" : episode,
            "loss"    : loss.item(),
            "done"    : episode_data["dones"][-1,:,0].sum().int().item(),
            "reward"  : episode_data["rewards"].sum(0).mean(0).sum().item()
    }))


