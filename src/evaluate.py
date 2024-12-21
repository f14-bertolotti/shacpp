import json, torch
from unroll import unroll

@torch.no_grad
def evaluate(
        episode,
        policy_model,
        world,
        steps,
        envs,
        logger,
        reward_model = None,
        world_model  = None,
        reward_tolerance = 0.1,
        world_tolerance  = 0.1
    ):
    policy_model.eval()
    if reward_model: reward_model.eval()
    if world_model :  world_model.eval()

    eval_episode = unroll(
        observations = None, 
        world        = world,
        unroll_steps = steps,
        policy_model = policy_model.act,
    )
   
    ground_proxy_rewards = None
    proxy_observations   = None
    proxy_rewards        = None
    rewards = eval_episode["rewards"]
    indices = torch.randint(0, rewards.size(0)*rewards.size(1), (1000,), device=rewards.device)

    if reward_model is not None: 
        ground_proxy_rewards = reward_model(
            eval_episode["observations"][:-1].flatten(0,1)[indices], 
            eval_episode["actions"     ]     .flatten(0,1)[indices], 
            eval_episode["observations"][+1:].flatten(0,1)[indices]
        )

    if  world_model is not None:
        proxy_observations = world_model(
            eval_episode["observations"]                    .transpose(0,1), 
            eval_episode["actions"     ][:world_model.steps].transpose(0,1)
        )["observations"].transpose(0,1)

    if world_model is not None and reward_model is not None:
        proxy_rewards = reward_model(
            proxy_observations     [:-1               ].flatten(0,1), 
            eval_episode["actions"][:world_model.steps].flatten(0,1), 
            proxy_observations     [+1:               ].flatten(0,1)
        ).view(world_model.steps, rewards.size(1), rewards.size(2))

    policy_model.train()
    if reward_model: reward_model.train()
    if world_model :  world_model.train()
    
    logger.info(json.dumps({
        "episode"              : episode,
        "done"                 : eval_episode["dones"][-1,:,0].sum().int().item(),
        "reward"               : rewards.sum().item() / envs,
    } | ({} if ground_proxy_rewards is None else {
        "ground_reward_accuracy" : ground_proxy_rewards.isclose(rewards.flatten(0,1)[indices], atol=reward_tolerance).float().mean().item(),
    }) | ({} if proxy_observations  is None else {
        "observation_accuracy"   : proxy_observations.isclose(eval_episode["observations"][:world_model.steps+1], atol=world_tolerance).float().mean().item(),
    }) | ({} if proxy_rewards       is None else {
        "proxy_reward_accuracy"  : proxy_rewards.isclose(eval_episode["rewards"][:world_model.steps], atol=reward_tolerance).float().mean().item(),
    })))

    return {
        "rewards" : rewards,
        "max_reward" : eval_episode["max_reward"]
    }


