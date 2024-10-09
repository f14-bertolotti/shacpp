import json, utils, numpy, torch

def train_reward(
        episode         ,
        model           ,
        episode_data    ,
        cached_data     ,
        batch_size      ,
        cache_size      ,
        bins            ,
        training_epochs ,
        optimizer       ,
        logger          ,
    ):

    rewards = (episode_data["rewards"]).flatten(0,1).sum(1)
    indexes = utils.bin_dispatch(rewards, bins, cache_size // bins)

    cached_data["mask"        ][indexes] = episode_data["dones"][:,:,0].flatten(0,1).detach().logical_not()
    cached_data["observations"][indexes] = episode_data["observations"].flatten(0,1).detach()
    cached_data["actions"     ][indexes] = episode_data["actions"]     .flatten(0,1).detach()
    cached_data["rewards"     ][indexes] = episode_data["rewards"]     .flatten(0,1).detach()

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
            loss = ((prd[tgt==0] - tgt[tgt==0])**2).mean() + ((prd[tgt>0] - tgt[tgt>0])**2).mean()
            loss.backward()
            optimizer.step()

            tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=.1).float().sum().item(), tot + numpy.prod(prd.shape).item() 
            logger.info(json.dumps({
                "episode"         : episode,
                "epoch"           : epoch,
                "step"            : step,
                "loss"            : loss.item(),
                "filled"          : cached_data["mask"].sum().item(),
                "filled_gt0"      : cached_data["mask"].logical_and(cached_data["rewards"].sum(-1)>0).sum().item(),
                "accuracy"        : tpfn/(tot+1e-7),
            }))

