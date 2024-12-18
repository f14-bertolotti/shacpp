import logging
import models
import torch
import utils
import json

def train_universe(
        episode          : int                                ,
        model            : models.Model                       ,
        episode_data     : dict[str,torch.Tensor]             ,
        batch_size       : int                                ,
        bins             : int|None                           ,
        training_epochs  : int                                ,
        optimizer        : torch.optim.Optimizer              ,
        logger           : logging.Logger                     ,
        cached_data      : dict[str,torch.Tensor]|None = None ,
        cache_size       : int|None = None                    ,
        slam             : float = .95                        ,
        gamma            : float = .99                        ,
        clip_coefficient : float|None = .5                    ,
        stop_threshold   : float|None = None                  ,
        obs_weight       : float = 0.1                        ,
        rew_weight       : float = 1                          ,
        val_weight       : float = 1                          ,
        tolerance        : float = .1                         ,
        ett              : int = 1                            ,
    ):
    """
        Training routine for the world model. It trains 'model' to match the observations of the environment.
        Training data are cached at each iteration, but the training is performed only every 'ett' episodes.
    """

    use_cache = cached_data is not None and cache_size is not None
    target_values = utils.compute_values(
        values  = episode_data["values"],
        rewards = episode_data["rewards"],
        dones   = episode_data["dones"].float(),
        slam    = slam,
        gamma   = gamma
    )
 
    alive   = episode_data["dones"][0,:,0].logical_not()
    if use_cache:
        rewards = (episode_data["rewards"]).sum(0).sum(1)
        indexes = utils.bin_dispatch(rewards, bins, cache_size // bins)
        indexes = indexes[alive]

        cached_data["mask"        ][indexes] = True
        cached_data["observations"][indexes] = episode_data["observations"].transpose(0,1)[alive].detach()
        cached_data["actions"     ][indexes] = episode_data["actions"     ].transpose(0,1)[alive].detach()
        cached_data["rewards"     ][indexes] = episode_data["rewards"     ].transpose(0,1)[alive].detach()
        cached_data["values"      ][indexes] = target_values               .transpose(0,1)[alive].detach()

    if episode % ett == 0: 

        # buid dataloader
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                cached_data["observations"][cached_data["mask"]] if use_cache else episode_data["observations"].detach().transpose(0,1)[alive],
                cached_data["actions"     ][cached_data["mask"]] if use_cache else episode_data["actions"     ].detach().transpose(0,1)[alive],
                cached_data["rewards"     ][cached_data["mask"]] if use_cache else episode_data["rewards"     ].detach().transpose(0,1)[alive],
                cached_data["values"      ][cached_data["mask"]] if use_cache else target_values               .detach().transpose(0,1)[alive],
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )

        for epoch in range(training_epochs):
            tpfn_obs, tot_obs = 0, 0
            tpfn_rew, tot_rew = 0, 0
            tpfn_val, tot_val = 0, 0
            for step, (obs, act, rew, val) in enumerate(dataloader,1):
                optimizer.zero_grad()
                
                # forward pass
                prd = model(obs,act)
                prd_obs = prd["observations"]
                prd_rew = prd["rewards"]
                prd_val = prd["values"]

                # observation loss
                obs_loss = torch.nn.functional.mse_loss(prd_obs, obs, reduction="mean")

                # reward loss
                rew_loss = torch.nn.functional.mse_loss(prd_rew[:,1:], rew, reduction="mean")

                # value loss
                val_loss = torch.nn.functional.mse_loss(prd_val[:,1:], val, reduction="mean")

                # loss
                loss = obs_loss * obs_weight + rew_loss * rew_weight + val_loss * val_weight

                # accuracy metric
                tot_obs  += obs.numel()
                tpfn_obs += obs.isclose(prd_obs,atol=tolerance).sum().item()
                tot_rew  += rew.numel()
                tpfn_rew += rew.isclose(prd_rew[:,1:],atol=tolerance).sum().item()
                tot_val  += val.numel()
                tpfn_val += val.isclose(prd_val[:,1:],atol=tolerance).sum().item()

                # backpropagation
                loss.backward()
                if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
                optimizer.step()

                # logging
                logger.info(json.dumps({
                    "episode"              : episode,
                    "epoch"                : epoch,
                    "step"                 : step,
                    "observation_shape"    : obs.shape,
                    "reward_shape"         : rew.shape,
                    "value_shape"          : val.shape,
                    "observation_accuracy" : tpfn_obs / (tot_obs + 1e-7),
                    "reward_accuracy"      : tpfn_rew / (tot_rew + 1e-7),
                    "value_accuracy"       : tpfn_val / (tot_val + 1e-7),
                    "observation_loss"     : loss.item(),
                }))

            if stop_threshold is not None and \
               tpfn_obs/(tot_obs+1e-7) > stop_threshold and \
               tpfn_rew/(tot_rew+1e-7) > stop_threshold and \
               tpfn_val/(tot_val+1e-7) > stop_threshold: break


