import logging
import models
import utils
import torch
import json

def train_value(
        episode          : int                    ,
        model            : models.Model           ,
        optimizer        : torch.optim.Optimizer  ,
        episode_data     : dict[str,torch.Tensor] ,
        training_epochs  : int                    ,
        batch_size       : int                    ,
        logger           : logging.Logger         ,
        slam             : float = .95            ,
        gamma            : float = .99            ,
        clip_coefficient : float|None = .5        ,
        ett              : int = 1                ,
    ):
    """
        Training routine for the value model. It trains 'model' to predict the value of the observations.
        No caching is performed, the training is performed every 'ett' episodes.
    """


    
    if episode % ett == 0: 
        target_values = utils.compute_values(
            values  = episode_data["values"] ,
            rewards = episode_data["rewards"],
            dones   = episode_data["dones"].float()  ,
            slam    = slam                   ,
            gamma   = gamma
        )

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                episode_data["observations"].flatten(0,1).detach(),
                target_values               .flatten(0,1).detach(),
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )
        
        for epoch in range(1, training_epochs+1):
            for step, (obs, tgt) in enumerate(dataloader, 1):
                optimizer.zero_grad()

                # forward pass
                prd = model(obs)

                # compute loss
                loss = ((prd - tgt)**2).mean()
        
                # backward pass
                loss.backward()
                if clip_coefficient is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
                optimizer.step()
        
                # loggging
                logger.info(json.dumps({
                    "episode"         : episode,
                    "epoch"           : epoch,
                    "step"            : step,
                    "loss"            : loss.item(),
                }))


