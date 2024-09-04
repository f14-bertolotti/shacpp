import utils, json, torch
def train_value(
        episode,
        model, 
        optimizer, 
        episode_data,
        training_epochs, 
        batch_size, 
        slam, 
        gamma, 
        logger,
        clip_coefficient = .5
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
            prd = model(obs)
            loss = ((prd - tgt)**2).mean()
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
            optimizer.step()
    
            logger.info(json.dumps({
                "episode"         : episode,
                "epoch"           : epoch,
                "step"            : step,
                "loss"            : loss.item(),
            }))


