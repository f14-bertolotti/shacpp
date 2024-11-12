import torch

def unroll(
        policy_model           ,
        world                  ,
        observations   = None  ,
        dones          = None  ,
        unroll_steps   = 64    ,
        reset_dones    = False ,
        use_diffreward = False ,
    ):

    world.world.zero_grad()

    if dones is not None and observations is not None and reset_dones:
        for i in dones[:,0].nonzero():
            observations[i] = torch.stack(world.reset_at(i)).transpose(0,1)[i]

    if observations is None and dones is None: 
        observations = torch.stack(world.reset()).transpose(0,1)
        dones        = torch.zeros(observations.size(0), observations.size(1), device=observations.device).bool()

    max_reward   = world.scenario.max_rewards()

    observations = observations.detach()

    observation_cache = []
    action_cache      = []
    reward_cache      = []
    done_cache        = []
    logprobs_cache    = []
    entropy_cache     = []
    logits_cache      = []

    for step in range(1, unroll_steps+1):
        observation_cache.append(observations)
        done_cache       .append(dones)
        policy_result = policy_model(observations)
        actions  = policy_result["actions"]
        logprobs = policy_result.get("logprobs",torch.empty(0))
        entropy  = policy_result.get("entropy" ,torch.empty(0))
        logits   = policy_result.get("logits"  ,torch.empty(0))
        prev = observations 
        observations, rewards, dones, _ = world.step(actions.transpose(0,1))
        if use_diffreward: rewards = world.scenario.diffreward(prev.transpose(0,1), actions, observations)
        observations = torch.stack(observations).transpose(0,1)
        rewards      = torch.stack(rewards     ).transpose(0,1)
        dones        = dones.unsqueeze(-1).repeat(1,observations.size(1))

        action_cache   .append(actions)
        reward_cache   .append(rewards)
        logprobs_cache .append(logprobs)
        entropy_cache  .append(entropy)
        logits_cache   .append(logits)

    observation_cache = torch.stack(observation_cache + [observations])
    done_cache        = torch.stack(done_cache + [dones])
    action_cache      = torch.stack(action_cache)
    reward_cache      = torch.stack(reward_cache)
    logprobs_cache    = torch.stack(logprobs_cache)
    entropy_cache     = torch.stack(entropy_cache)
    logits_cache      = torch.stack(logits_cache)


    return { 
            "logprobs"      : logprobs_cache    ,
            "entropy"       : entropy_cache     ,
            "actions"       : action_cache      ,
            "observations"  : observation_cache ,
            "rewards"       : reward_cache      ,
            "logits"        : logits_cache      ,
            "dones"         : done_cache        ,
            "max_reward"    : max_reward        ,
    }


