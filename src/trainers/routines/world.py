import logging
import models
import torch
import utils
import json

def train_world(
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
        clip_coefficient : float|None = .5                    ,
        stop_threshold   : float|None = None                  ,
        tolerance        : float = .1                         ,
        ett              : int = 1                            ,
    ):
    """
        Training routine for the world model. It trains 'model' to match the observations of the environment.
        Training data are cached at each iteration, but the training is performed only every 'ett' episodes.
    """

    use_cache = cached_data is not None and cache_size is not None
 
    alive   = episode_data["dones"][0,:,0].logical_not()
    if use_cache:
        rewards = (episode_data["rewards"]).sum(0).sum(1)
        indexes = utils.bin_dispatch(rewards, bins, cache_size // bins)

        # cache episode data
        indexes = indexes[alive]

        cached_data["mask"        ][indexes] = True
        cached_data["observations"][indexes] = episode_data["observations"].transpose(0,1)[alive].detach()
        cached_data["actions"     ][indexes] = episode_data["actions"     ].transpose(0,1)[alive].detach()

    if episode % ett == 0: 

        # buid dataloader
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                cached_data["observations"][cached_data["mask"]] if use_cache else episode_data["observations"].detach().transpose(0,1)[alive],
                cached_data["actions"     ][cached_data["mask"]] if use_cache else episode_data["actions"     ].detach().transpose(0,1)[alive],
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )

        for epoch in range(training_epochs):
            tpfn, tot = 0, 0
            for step, (obs, act) in enumerate(dataloader,1):
                optimizer.zero_grad()
                
                # forward pass
                prd = model(obs,act)["observations"]

                # observation loss
                loss = torch.nn.functional.mse_loss(prd, obs, reduction="mean")

                # accuracy metric
                tot  += prd.numel()
                tpfn += prd.isclose(obs,atol=tolerance).sum().item()

                # backpropagation
                loss.backward()
                if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
                optimizer.step()

                # logging
                logger.info(json.dumps({
                    "episode"         : episode,
                    "epoch"           : epoch,
                    "step"            : step,
                    "shape"           : prd.shape,
                    "accuracy"        : tpfn / (tot + 1e-7),
                    "observation_loss": loss.item(),
                }))

            if stop_threshold is not None and tpfn/(tot+1e-7) > stop_threshold: break


