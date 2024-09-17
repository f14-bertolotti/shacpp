import json, utils, numpy, torch

def train_world(
        episode, 
        model, 
        episode_data, 
        cached_data,
        batch_size, 
        training_epochs,
        optimizer,
        logger,
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
 
    peak_cache = episode_data["rewards"].transpose(0,1).sum(-1).sum(-1)
    indexes = utils.pert(
        low  = cached_data["pert_low"],
        high = cached_data["pert_high"],
        peak = (peak_cache - peak_cache.min()) / (peak_cache.max() - peak_cache.min() + 1e-5) * (cached_data["observations"].size(0)-1),
    ).round().to(torch.long)

    cached_data["mask"        ][indexes] = episode_data["dones"][0,:,0].detach().logical_not()
    cached_data["observations"][indexes] = episode_data["observations"][0].detach().clone()
    cached_data["actions"     ][indexes] = episode_data["actions"]        .transpose(0,1).detach().clone()
    cached_data["rewards"     ][indexes] = episode_data["rewards"]        .transpose(0,1).detach().clone()
    cached_data["values"      ][indexes] = target_values                  .transpose(0,1).detach().clone()


    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            cached_data["observations"][cached_data["mask"]],
            cached_data["actions"     ][cached_data["mask"]],
            cached_data["rewards"     ][cached_data["mask"]],
            cached_data["values"      ][cached_data["mask"]]
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = batch_size,
        drop_last  = False,
        shuffle    = True
    )

    for epoch in range(4):
        tpfn_rew,tot_rew = 0,0
        tpfn_val,tot_val = 0,0
        tpfn_nonzero_rew, tot_nonzero_rew = 0,0
        tpfn_nonzero_val, tot_nonzero_val = 0,0
        for step, (obs, act, tgt_rew, tgt_val) in enumerate(dataloader,1):
            optimizer.zero_grad()
            prd_rew,prd_val = model(obs.unsqueeze(1),act)

            gtz = (tgt_rew  > 0)
            l1 = ((prd_rew[gtz] - tgt_rew[gtz])**2).mean()
            l2 = ((prd_rew[gtz.logical_not()] - tgt_rew[gtz.logical_not()])**2).mean()


            loss = l1 + l2 + ((prd_val - tgt_val)**2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            tpfn_rew, tot_rew = tpfn_rew + torch.isclose(prd_rew, tgt_rew, atol=.1).sum().item(), tot_rew + numpy.prod(prd_rew.shape).item() 
            tpfn_val, tot_val = tpfn_val + torch.isclose(prd_val, tgt_val, atol=.1).sum().item(), tot_val + numpy.prod(prd_val.shape).item() 
            tpfn_nonzero_rew, tot_nonzero_rew = tpfn_nonzero_rew + torch.isclose(prd_rew[tgt_rew>0], tgt_rew[tgt_rew>0], atol=.1).float().sum().item(), tot_nonzero_rew + numpy.prod(prd_rew[tgt_rew>0].shape).item()
            tpfn_nonzero_val, tot_nonzero_val = tpfn_nonzero_val + torch.isclose(prd_val[tgt_val>0], tgt_val[tgt_val>0], atol=.1).float().sum().item(), tot_nonzero_val + numpy.prod(prd_val[tgt_val>0].shape).item()
            logger.info(json.dumps({
                "episode"         : episode,
                "epoch"           : epoch,
                "step"            : step,
                "loss"            : loss.item(),
                "accuracy_rew"    : tpfn_rew/(tot_rew+1e-7),
                "accuracy_rew_nz" : tpfn_nonzero_rew/(tot_nonzero_rew+1e-7),
                "accuracy_val"    : tpfn_val/(tot_val+1e-7),
                "accuracy_val_nz" : tpfn_nonzero_val/(tot_nonzero_val+1e-7),
                "full" : cached_data["mask"].sum().item()
            }))


