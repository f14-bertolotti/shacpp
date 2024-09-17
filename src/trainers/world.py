import utils, json, torch

def train_world(
        episode         ,
        model           ,
        episode_data    ,
        cached_data     ,
        batch_size      ,
        dataset_size    ,
        training_epochs ,
        optimizer       ,
        logger          ,
        clip_coefficient = .5,
        slam = .95,
        gamma = .99,
    ):


    target_values = utils.compute_values(
        values  = episode_data["values"] ,
        rewards = episode_data["rewards"],
        dones   = episode_data["dones"].float()  ,
        slam    = slam                   ,
        gamma   = gamma
    )
    
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            episode_data["observations"].transpose(0,1).detach(),
            episode_data["actions"     ].transpose(0,1).detach(),
            episode_data["rewards"     ].transpose(0,1).detach(),
            episode_data["values"      ].transpose(0,1).detach(),
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = batch_size,
        drop_last  = False,
        shuffle    = True
    )
 
    for epoch in range(training_epochs):
        for step,(obs,act,rew,val) in enumerate(dataloader):
            optimizer.zero_grad()
            prd_rew, prd_val = model(obs[:,0].unsqueeze(1), act)
            rew_loss = ((prd_rew[rew == 0] - rew[rew == 0])**2).mean()
            if (rew>0).any(): rew_loss += ((prd_rew[rew >  0] - rew[rew >  0])**2).mean()
            val_loss = ((prd_val - val)**2).mean()
            loss = ( rew_loss + val_loss ) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
            optimizer.step()

            logger.info(json.dumps({
                "episode" : episode,
                "epoch"   : epoch,
                "step"    : step,
                "loss"    : loss.item(),
                "reward_accuracy"    : prd_rew.isclose(rew, atol=.1).float().mean().item(),
                "reward_accuracy_nz" : prd_rew[rew>0].isclose(rew[rew>0], atol=.1).float().mean().item(),
            }))

