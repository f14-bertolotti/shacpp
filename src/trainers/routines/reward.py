import json
import utils
import torch
import models
import logging

def train_reward(
        episode         : int                                ,
        model           : models.Model                       ,
        episode_data    : dict[str,torch.Tensor]             ,
        batch_size      : int                                ,
        bins            : int|None                           ,
        training_epochs : int                                ,
        optimizer       : torch.optim.Optimizer              ,
        logger          : logging.Logger                     ,
        cached_data     : dict[str,torch.Tensor]|None = None ,
        cache_size      : int|None = None                    ,
        clip_coefficient: float|None = .5                    ,
        stop_threshold  : float|None = None                  ,
        tolerance       : float = .1                         ,
        ett             : int = 1                            ,
    ):

    use_cache = cached_data is not None and cache_size is not None

    if use_cache:
        rewards = (episode_data["rewards"]).flatten(0,1).sum(1)
        indexes = utils.bin_dispatch(rewards, bins, cache_size // bins)

        cached_data["mask"        ][indexes] = episode_data["dones"][:-1,:,0]   .flatten(0,1).detach().logical_not()
        cached_data["nextobs"     ][indexes] = episode_data["observations"][1:] .flatten(0,1).detach()
        cached_data["prevobs"     ][indexes] = episode_data["observations"][:-1].flatten(0,1).detach()
        cached_data["actions"     ][indexes] = episode_data["actions"]          .flatten(0,1).detach()
        cached_data["rewards"     ][indexes] = episode_data["rewards"]          .flatten(0,1).detach()

    if episode % ett == 0: 
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                cached_data["prevobs"][cached_data["mask"]] if use_cache else full_observations[:-1] .detach().flatten(0,1),
                cached_data["nextobs"][cached_data["mask"]] if use_cache else full_observations[+1:] .detach().flatten(0,1),
                cached_data["actions"][cached_data["mask"]] if use_cache else episode_data["actions"].detach().flatten(0,1),
                cached_data["rewards"][cached_data["mask"]] if use_cache else episode_data["rewards"].detach().flatten(0,1),
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )

        for epoch in range(training_epochs):
            tpfn,tot = 0,0
            for step, (prevobs, nextobs, act, tgt) in enumerate(dataloader,1):
                optimizer.zero_grad()
                prd = model(prevobs,act,nextobs)
                loss = ((prd - tgt)**2).mean()
                loss.backward()
                if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
                optimizer.step()

                tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=tolerance).float().sum().item(), tot + prd.numel() 
                logger.info(json.dumps({
                    "episode"         : episode,
                    "epoch"           : epoch,
                    "step"            : step,
                    "loss"            : loss.item(),
                    "accuracy"        : tpfn/(tot+1e-7),
                }))
            
            if stop_threshold is not None and tpfn/(tot+1e-7) > stop_threshold: break
