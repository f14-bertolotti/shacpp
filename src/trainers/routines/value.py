import logging
import models
import utils
import torch
import json

def train_value(
        episode          : int                                ,
        model            : models.Model                       ,
        optimizer        : torch.optim.Optimizer              ,
        episode_data     : dict[str,torch.Tensor]             ,
        training_epochs  : int                                ,
        batch_size       : int                                ,
        bins             : int|None                           ,
        logger           : logging.Logger                     ,
        cached_data      : dict[str,torch.Tensor]|None = None ,
        cache_size       : int|None = None                    ,
        slam             : float = .95                        ,
        gamma            : float = .99                        ,
        clip_coefficient : float|None = .5                    ,
        stop_threshold   : float|None = None                  ,
        tolerance        : float = .1                         ,
        ett              : int = 1                            ,
    ):
    """
        Training routine for the value model. It trains 'model' to predict the value of the observations.
        No caching is performed, the training is performed every 'ett' episodes.
    """

    use_cache = cached_data is not None and cache_size is not None

    target_values = utils.compute_values(
        values  = episode_data["values"] ,
        rewards = episode_data["rewards"],
        dones   = episode_data["dones"].float()  ,
        slam    = slam                   ,
        gamma   = gamma
    )

    if use_cache:
        values = (target_values).flatten(0,1).sum(1)
        indexes = utils.bin_dispatch(values, bins, cache_size // bins)
        cached_data["mask"        ][indexes] = episode_data["dones"][:-1,:,0]   .flatten(0,1).detach().logical_not()
        cached_data["observations"][indexes] = episode_data["observations"][:-1].flatten(0,1).detach()
        cached_data["targets"     ][indexes] = target_values                    .flatten(0,1).detach()

    if episode % ett == 0: 
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                cached_data["observations"][cached_data["mask"]] if use_cache else episode_data["observations"].detach().flatten(0,1),
                cached_data["targets"]     [cached_data["mask"]] if use_cache else target_values               .detach().flatten(0,1),
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )
        
        for epoch in range(1, training_epochs+1):
            tpfn,tot = 0,0
            for step, (obs, tgt) in enumerate(dataloader, 1):
                optimizer.zero_grad()

                # forward pass
                prd = model(obs)

                # compute loss
                loss = ((prd - tgt)**2).mean()

                # compute accuracy
                tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=tolerance).float().sum().item(), tot + tgt.numel()
        
                # backward pass
                loss.backward()
                if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
                optimizer.step()
        
                # loggging
                logger.info(json.dumps({
                    "episode"         : episode,
                    "epoch"           : epoch,
                    "accuracy"        : tpfn/(tot+1e-7),
                    "step"            : step,
                    "loss"            : loss.item(),
                }))

            if stop_threshold is not None and tpfn/(tot+1e-7) > stop_threshold: break


