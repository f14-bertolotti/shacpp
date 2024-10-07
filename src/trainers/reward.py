import json, utils, numpy, torch

max_reward = float("-inf")
min_reward = float("+inf")

def train_reward(
        episode         ,
        model           ,
        episode_data    ,
        cached_data     ,
        batch_size      ,
        cache_size      ,
        training_epochs ,
        optimizer       ,
        logger          ,
    ):

    global max_reward, min_reward

    rewards = (episode_data["rewards"]).flatten(0,1).sum(1)
    #rewards[rewards > 0] += 10
    max_reward = max(max_reward, rewards.max().item())
    min_reward = min(min_reward, rewards.min().item())

    normalized_reward = (rewards - min_reward) / (max_reward - min_reward + 1e-5)
    indexes = utils.random_dispatch(normalized_reward, cache_size, lamb=64)
    #print("-"*100)
    #print(indexes[rewards <= 0][:33])
    #print(normalized_reward[rewards == 0][:33])
    #print(normalized_reward[rewards > 0][:33])
    #print(indexes[rewards > 0][:33])

    #print("="*100)

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

