import torch, json
import utils
import random

def train_policy(
        episode,
        policy_model,
        value_model,
        episode_data,
        optimizer,
        gammas,
        logger,
        clip_coefficient = .5
    ):
    steps, envs = episode_data["observations"].size(0), episode_data["observations"].size(1)
   
    ## compute value cache mask
    #dead_runs = episode_data["dones"][0,:,0]
    #live_runs = dead_runs.logical_not()
    #live_steps = episode_data["dones"][:,live_runs,0].logical_not().sum(0) - 1

    last_valid_step = episode_data["dones"].logical_not().sum(0)[:,0]-1
    
    # compute loss
    optimizer.zero_grad()
    
    #advantages = utils.compute_advantages(
    #    value_model, 
    #    episode_data["rewards"], 
    #    episode_data["last_observations"], 
    #    episode_data["values"], 
    #    episode_data["dones"], 
    #    episode_data["last_dones"]
    #)

    #logprobs = policy_model.eval_action(episode_data["observations"].flatten(0,1), episode_data["actions"].flatten(0,1))["logprobs"].view(episode_data["rewards"].shape)
    #choice = random.choice(list(range(episode_data["actions"].size(2)))) 

    
    #print(torch.gather(episode_data["values"], 0, last_valid_step).shape)
    values = episode_data["values"][last_valid_step,torch.arange(last_valid_step.size(0))]
    vgamma = gammas[last_valid_step, torch.arange(last_valid_step.size(0))]

    loss = -(
                (episode_data["proxy_rewards"] * gammas * episode_data["dones"].logical_not()).sum() #+ \
                #((vgamma * values)).sum() # + \
                #(-advantages.detach() * logprobs.exp()).sum()
            ) / (steps * envs)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), clip_coefficient)
    optimizer.step()
    logger.info(json.dumps({
            "episode" : episode,
            "loss"    : loss.item(),
            "done"    : episode_data["dones"][-1,:,0].sum().int().item(),
            "reward"  : episode_data["rewards"].sum(0).mean(0).sum().item()
    }))


