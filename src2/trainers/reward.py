import json, utils, numpy, torch

def train_reward(
        episode, 
        model, 
        episode_data, 
        cached_data,
        batch_size, 
        dataset_size, 
        training_epochs,
        optimizer,
        logger,
    ):

    peak_cache = episode_data["rewards"].flatten(0,1).sum(1)
    indexes = utils.pert(
        low  = cached_data["pert_low"],
        high = cached_data["pert_high"],
        peak = (peak_cache - peak_cache.min()) / (peak_cache.max() - peak_cache.min() + 1e-5) * (dataset_size-1),
    ).round().to(torch.long)

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
        drop_last  = True,
        shuffle    = True
    )

    for epoch in range(training_epochs):
        tpfn,tot = 0,0
        tpfn_nonzero, tot_nonzero = 0,0
        for step, (obs, act, tgt) in enumerate(dataloader,1):
            optimizer.zero_grad()
            prd = model(obs,act)
            loss = ((prd[tgt==0] - tgt[tgt==0])**2).mean() + ((prd[tgt>0] - tgt[tgt>0])**2).mean()
            loss.backward()
            optimizer.step()

            tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=.1).float().sum().item(), tot + numpy.prod(prd.shape).item() 
            tpfn_nonzero, tot_nonzero = tpfn_nonzero + torch.isclose(prd[tgt>0], tgt[tgt>0], atol=.1).float().sum().item(), tot_nonzero + numpy.prod(prd[tgt>0].shape).item()
            logger.info(json.dumps({
                "episode"         : episode,
                "epoch"           : epoch,
                "step"            : step,
                "loss"            : loss.item(),
                "accuracy"        : tpfn/(tot+1e-7),
                "accuracy_nz"     : tpfn_nonzero/(tot_nonzero+1e-7),
                "dataset_size"    : cached_data["mask"].sum().item(),
                "dataset_size_nz" : (cached_data["rewards"][cached_data["mask"]] > 0).sum().item(),
            }))

        if tpfn/tot > .8: break
