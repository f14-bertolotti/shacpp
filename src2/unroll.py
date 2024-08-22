import torch

def unroll(
        policy_model,
        value_model,
        reward_model,
        world,
        observations = None,
        actions      = None,
        unroll_steps = 64
    ):

    world.world.zero_grad()
    if observations is None: 
        observations = torch.stack(world.reset()).transpose(0,1)
    if actions is None:
        actions = torch.zeros(observations.size(0), observations.size(1), 2, device=observations.device)

    observation_cache = []
    action_cache      = []
    reward_cache      = []
    done_cache        = []

    for step in range(1, unroll_steps+1):
        observation_cache.append(observations)
        
        actions = policy_model(observations, actions)
        observations, rewards, dones, _ = world.step(actions.transpose(0,1))
        observations = torch.stack(observations).transpose(0,1)
        rewards      = torch.stack(rewards     ).transpose(0,1).unsqueeze(-1)

        action_cache.append(actions)    
        reward_cache.append(rewards)     
        done_cache  .append(dones)

    observation_cache = torch.stack(observation_cache)
    action_cache      = torch.stack(action_cache)
    reward_cache      = torch.stack(reward_cache)
    done_cache        = torch.stack(done_cache)

    value_model.eval(); reward_model.eval()
    value_cache         = value_model(observation_cache.flatten(0,1)).view(observation_cache.size(0), observation_cache.size(1), observation_cache.size(2), 1)
    proxy_rewards_cache = reward_model(observation_cache.flatten(0,1), action_cache.flatten(0,1)).view(reward_cache.shape)
    reward_model.train(); value_model.train()
 

    return { 
            "observations"  : observation_cache, 
            "actions"       : action_cache, 
            "rewards"       : reward_cache, 
            "proxy_rewards" : proxy_rewards_cache,
            "values"        : value_cache,
            "dones"         : done_cache.unsqueeze(-1).repeat(1,1,observations.size(1)).float()
    }


