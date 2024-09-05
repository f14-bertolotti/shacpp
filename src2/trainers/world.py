
import json, torch
def train_world(
        episode         ,
        model           ,
        episode_data    ,
        optimizer       ,
        logger          ,
        clip_coefficient = .5
    ):
    
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            episode_data["observations"].transpose(0,1).detach(),
            episode_data["actions"].transpose(0,1).detach()
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = 64,
        drop_last  = False,
        shuffle    = True
    )
 
    for epoch in range(4):
        for step,(obs,act) in enumerate(dataloader):
            optimizer.zero_grad()
            prd = model(obs[:,0], act.transpose(0,1)).transpose(0,1)
            loss = ((obs - prd)**2).mean()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_coefficient)
            optimizer.step()

            m = obs.size(0)//2
            acc0 = obs[+0].isclose(prd[+0], atol=.1).float().mean().item()
            accm = obs[+m].isclose(prd[+m], atol=.1).float().mean().item()
            accf = obs[-1].isclose(prd[-1], atol=.1).float().mean().item()
            
            logger.info(json.dumps({
                "episode" : episode,
                "epoch"   : epoch,
                "step"    : step,
                "loss"    : loss.item(),
                "early_accuracy" : acc0,
                "mid_accuracy"  : accm,
                "final_accuracy" : accf,
            }))


