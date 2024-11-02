import json
import utils
import numpy
import torch
import models
import logging

def train_reward(
        episode         : int                    ,
        model           : models.Model           ,
        episode_data    : dict[str,torch.Tensor] ,
        cached_data     : dict[str,torch.Tensor] ,
        batch_size      : int                    ,
        cache_size      : int                    ,
        bins            : int                    ,
        training_epochs : int                    ,
        optimizer       : torch.optim.Optimizer  ,
        logger          : logging.Logger         ,
        clip_coefficient: float|None = .5        ,
        ett             : int = 1                ,
    ):

    rewards = (episode_data["rewards"]).flatten(0,1).sum(1)
    indexes = utils.bin_dispatch(rewards, bins, cache_size // bins)

    cached_data["mask"        ][indexes] = episode_data["dones"][:,:,0].flatten(0,1).detach().logical_not()
    cached_data["observations"][indexes] = episode_data["observations"].flatten(0,1).detach()
    cached_data["actions"     ][indexes] = episode_data["actions"]     .flatten(0,1).detach()
    cached_data["rewards"     ][indexes] = episode_data["rewards"]     .flatten(0,1).detach()

    if episode % ett == 0: 
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                cached_data["observations"][cached_data["mask"]],
                cached_data["actions"     ][cached_data["mask"]],
                cached_data["rewards"     ][cached_data["mask"]],
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )

        for epoch in range(training_epochs):
            tpfn,tot = 0,0
            for step, (obs, act, tgt) in enumerate(dataloader,1):
                optimizer.zero_grad()
                prd = model(obs,act)
                loss = ((prd - tgt)**2).mean()
                loss.backward()
                if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
                optimizer.step()

                tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=.1).float().sum().item(), tot + numpy.prod(prd.shape).item() 
                logger.info(json.dumps({
                    "episode"         : episode,
                    "epoch"           : epoch,
                    "step"            : step,
                    "loss"            : loss.item(),
                    "accuracy"        : tpfn/(tot+1e-7),
                }))

