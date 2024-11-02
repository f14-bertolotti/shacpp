import logging
import models
import torch
import utils
import json

def ppo_policy_value(
        policy_model        : models.Model           ,
        value_model         : models.Model           ,
        policy_optimizer    : torch.optim.Optimizer  ,
        value_optimizer     : torch.optim.Optimizer  ,
        episode_data        : dict[str, torch.Tensor],
        ppo_logger          : logging.Logger         ,
        epochs              : int   = 4              ,
        batch_size          : int   = 128            ,
        gamma               : float = .99            ,
        gaelm               : float = .95            ,
        value_clip          : bool  = True           ,
        clip_coefficient    : float = .2             ,
        value_coefficient   : float = .5             ,
        entropy_coefficient : float = .0             ,
        max_grad_norm       : float = .5             ,
    ):

    # comute values ###########################################################
    values = value_model(episode_data["observations"].flatten(0,1)).view(episode_data["rewards"].shape)

    # compute advantages ######################################################
    advantages = utils.compute_advantages(
        value_model = value_model                       ,
        rewards     = episode_data["rewards"]           ,
        next_obs    = episode_data["last_observations"] ,
        values      = values                            ,
        dones       = episode_data["dones"]             ,
        next_done   = episode_data["last_dones"]        ,
        gamma       = gamma                             ,
        gaelambda   = gaelm                             ,
    )

    # compute returns #########################################################
    returns = utils.compute_returns(
        advantages = advantages, 
        values     = values,
    )

    # create dataloader #######################################################
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            episode_data["observations"].detach().flatten(0,1),
            episode_data["logprobs"    ].detach().flatten(0,1),
            episode_data["actions"     ].detach().flatten(0,1),
            values                      .detach().flatten(0,1),
            advantages                  .detach().flatten(0,1),
            returns                     .detach().flatten(0,1),
        ),
        collate_fn = torch.utils.data.default_collate ,
        batch_size = batch_size                       ,
        drop_last  = False                            ,
        shuffle    = True                             ,
    )

    for epoch in range(epochs):
        for step, (observations, old_logprobs, actions, old_values, advantages, returns) in enumerate(dataloader):
            value_optimizer .zero_grad()
            policy_optimizer.zero_grad()

            policy_result = policy_model.eval_action(observations = observations, actions = actions)
            new_logprobs, entropy = policy_result["logprobs"], policy_result["entropy"]
            new_values = value_model(observations)

            loss = utils.ppo_loss(
                new_values   = new_values          ,
                old_values   = old_values          ,
                new_logprobs = new_logprobs        ,
                old_logprobs = old_logprobs        ,
                advantages   = advantages          ,
                returns      = returns             ,
                entropy      = entropy             ,
                vclip        = value_clip          ,
                clipcoef     = clip_coefficient    ,
                vfcoef       = value_coefficient   ,
                entcoef      = entropy_coefficient
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_( value_model.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            value_optimizer .step()
            policy_optimizer.step()

            ppo_logger.info(json.dumps({
                "epoch"     : epoch,
                "step"      : step,
                "loss"      : loss.item(),
                "entropy"   : entropy   .mean().item(),
                "advantages": advantages.mean().item(),
                "returns"   : returns   .mean().item(),
            }))

