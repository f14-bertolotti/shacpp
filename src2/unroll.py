import torch

def unroll(
        policy_model,
        world,
        observations = None,
        dones        = None,
        unroll_steps = 64
    ):

    world.world.zero_grad()
    if observations is None: observations = torch.stack(world.reset()).transpose(0,1)
    if dones        is None: dones        = torch.zeros(observations.size(0), observations.size(1), device=observations.device).bool()

    observation_cache = []
    action_cache      = []
    reward_cache      = []
    done_cache        = []
    logprobs_cache    = []
    entropy_cache     = []

    for step in range(1, unroll_steps+1):
        observation_cache.append(observations)
        done_cache       .append(dones)
        
        policy_result = policy_model(observations)
        actions  = policy_result["actions"]
        logprobs = policy_result.get("logprobs",torch.empty(0))
        entropy  = policy_result.get("entropy" ,torch.empty(0))

        observations, rewards, dones, _ = world.step(actions.transpose(0,1))
        observations = torch.stack(observations).transpose(0,1)
        rewards      = torch.stack(rewards     ).transpose(0,1)
        dones        = dones.unsqueeze(-1).repeat(1,observations.size(1))

        action_cache   .append(actions)
        reward_cache   .append(rewards)
        logprobs_cache .append(logprobs)
        entropy_cache  .append(entropy)

    observation_cache = torch.stack(observation_cache)
    action_cache      = torch.stack(action_cache)
    reward_cache      = torch.stack(reward_cache)
    done_cache        = torch.stack(done_cache)
    logprobs_cache    = torch.stack(logprobs_cache)
    entropy_cache     = torch.stack(entropy_cache)

    return { 
            "logprobs"      : logprobs_cache,
            "entropy"       : entropy_cache,
            "actions"       : action_cache,
            "observations"  : observation_cache, 
            "rewards"       : reward_cache, 
            "dones"         : done_cache,

            "last_dones"        : dones,
            "last_observations" : observations
    }


