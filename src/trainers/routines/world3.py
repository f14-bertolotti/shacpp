import logging
import models
import torch
import utils
import json

def train_world3(
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
        clip_coefficient : float|None = .5        ,
        ett              : int = 1                ,
    ):
    """
        Training routine for the world model. It trains 'model' to match the observations/reward/value of the environment.
        Training data are cached at each iteration, but the training is performed only every 'ett' episodes.
    """

    rewards = (episode_data["rewards"]).sum(0).sum(1)
    indexes = utils.bin_dispatch(rewards, bins, cache_size // bins)

    alive   = episode_data["dones"][0,:,0].logical_not()
    indexes = indexes[alive]

    cached_data["mask"        ][indexes] = True
    cached_data["observations"][indexes] = episode_data["observations"     ].transpose(0,1)[alive].detach()
    cached_data["actions"     ][indexes] = episode_data["actions"          ].transpose(0,1)[alive].detach()
    cached_data["rewards"     ][indexes] = episode_data["rewards"          ].transpose(0,1)[alive].detach()

    if episode % ett == 0:

        # buid dataloader
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
            for step, (obs, act, tgt_rew) in enumerate(dataloader,1):
                optimizer.zero_grad()
                
                # forward pass
                prediction = model(obs[:,0].unsqueeze(1),act)

                # reward losses
                gtz = (tgt_rew  > 0)
                lr1 = ((prediction["rewards"][gtz] - tgt_rew[gtz])**2).mean()
                lr2 = ((prediction["rewards"][gtz.logical_not()] - tgt_rew[gtz.logical_not()])**2).mean()
                lr  = lr1 + lr2

                # observation loss
                lo = ((prediction["observations"] - obs)**2).mean()

                loss = lr + lo

                # backpropagation
                loss.backward()
                if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
                optimizer.step()

                # log step data
                logger.info(json.dumps({
                    "episode"         : episode,
                    "epoch"           : epoch,
                    "step"            : step,
                    "reward_loss1"    : lr1.item(),
                    "reward_loss2"    : lr2.item(),
                    "observation_loss": lo.item(),
                }))


