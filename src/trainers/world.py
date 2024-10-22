import logging
import models
import torch
import utils
import json

def train_world(
        episode          : int                    ,
        model            : models.Model           ,
        episode_data     : dict[str,torch.Tensor] ,
        cached_data      : dict[str,torch.Tensor] ,
        batch_size       : int                    ,
        cache_size       : int                    ,
        bins             : int                    ,
        training_epochs  : int                    ,
        optimizer        : torch.optim.Optimizer  ,
        logger           : logging.Logger         ,
        slam             : float = .95            ,
        gamma            : float = .99            ,
        clip_coefficient : float = .5  
    ):

    target_values = utils.compute_values(
        values  = episode_data["values"]        ,
        rewards = episode_data["rewards"]       ,
        dones   = episode_data["dones"].float() ,
        slam    = slam                          ,
        gamma   = gamma
    )
 
    rewards = (episode_data["rewards"]).flatten(0,1).sum(1)
    indexes = utils.bin_dispatch(rewards, bins, cache_size // bins)

    alive   = episode_data["dones"][0,:,0].logical_not()
    indexes = indexes[alive]


    cached_data["mask"        ][indexes] = True
    cached_data["observations"][indexes] = episode_data["observations"     ].transpose(0,1)[alive].detach()
    cached_data["actions"     ][indexes] = episode_data["actions"          ].transpose(0,1)[alive].detach()
    cached_data["rewards"     ][indexes] = episode_data["rewards"          ].transpose(0,1)[alive].detach()
    cached_data["last_obs"    ][indexes] = episode_data["last_observations"][alive].detach() 
    cached_data["values"      ][indexes] = target_values.transpose(0,1)[alive].detach()

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            cached_data["observations"][cached_data["mask"]],
            cached_data["actions"     ][cached_data["mask"]],
            cached_data["rewards"     ][cached_data["mask"]],
            cached_data["values"      ][cached_data["mask"]],
            cached_data["last_obs"    ][cached_data["mask"]]
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = batch_size,
        drop_last  = False,
        shuffle    = True
    )

    for epoch in range(training_epochs):
        for step, (obs, act, tgt_rew, tgt_val, last) in enumerate(dataloader,1):
            optimizer.zero_grad()
            
            prd_rew, prd_val, prd_obs = model(obs[:,0].unsqueeze(1),act)

            # reward losses
            gtz = (tgt_rew  > 0)
            lr1 = ((prd_rew[gtz] - tgt_rew[gtz])**2).mean()
            lr2 = ((prd_rew[gtz.logical_not()] - tgt_rew[gtz.logical_not()])**2).mean()

            # observation loss
            lo = (((prd_obs[:,:-1] - obs)**2).sum(1) + (prd_obs[:,-1] - last)**2).mean() / prd_obs.size(1)

            # value loss
            lv = ((prd_val - tgt_val)**2).mean()

            loss = lr1 + lr2 + lo + lv

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
            optimizer.step()

            # log step data
            logger.info(json.dumps({
                "episode"         : episode,
                "epoch"           : epoch,
                "step"            : step,
                "reward_loss1"    : lr1.item(),
                "reward_loss2"    : lr2.item(),
                "observation_loss": lo.item(),
                "value_loss"      : lv.item(),
            }))


