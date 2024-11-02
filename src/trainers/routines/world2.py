import logging
import models
import torch
import utils
import json

def train_world2(
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
        clip_coefficient : float|None = .5        ,
        ett              : int = 1                ,
    ):
    """
        Training routine for the world model. It trains 'model' to match the observations of the environment.
        Training data are cached at each iteration, but the training is performed only every 'ett' episodes.
    """
 
    # cache episode data
    rewards = (episode_data["rewards"]).sum(0).sum(1)
    indexes = utils.bin_dispatch(rewards, bins, cache_size // bins)

    alive   = episode_data["dones"][0,:,0].logical_not()
    indexes = indexes[alive]

    cached_data["mask"        ][indexes] = True
    cached_data["observations"][indexes] = episode_data["observations"     ].transpose(0,1)[alive].detach()
    cached_data["actions"     ][indexes] = episode_data["actions"          ].transpose(0,1)[alive].detach()
    cached_data["last_obs"    ][indexes] = episode_data["last_observations"][alive].detach() 

    if episode % ett == 0: 

        # buid dataloader
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                cached_data["observations"][cached_data["mask"]],
                cached_data["actions"     ][cached_data["mask"]],
                cached_data["last_obs"    ][cached_data["mask"]]
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )

        for epoch in range(training_epochs):
            tpfn, tot = 0, 0
            for step, (obs, act, last) in enumerate(dataloader,1):
                optimizer.zero_grad()
                
                # forward pass
                _, _, prd_obs = model(obs[:,0].unsqueeze(1),act)

                # observation loss
                loss = (((prd_obs[:,:-1] - obs)**2).sum(1) + (prd_obs[:,-1] - last)**2).mean() / prd_obs.size(1)

                # accuracy metric
                tot  += prd_obs.numel()
                tpfn += prd_obs[:,:-1].isclose(obs,atol=.1).sum().item() + prd_obs[:,-1].isclose(last,atol=.1).sum().item()

                # backpropagation
                loss.backward()
                if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
                optimizer.step()

                # logging
                logger.info(json.dumps({
                    "episode"         : episode,
                    "epoch"           : epoch,
                    "step"            : step,
                    "accuracy"        : tpfn / tot,
                    "observation_loss": loss.item(),
                }))


