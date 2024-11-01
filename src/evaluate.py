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
        world_model = None
    ):
    policy_model.eval()
    if reward_model: reward_model.eval()
    if world_model :  world_model.eval()

    eval_episode = unroll(
        observations = None, 
        world        = world,
        unroll_steps = steps,
        policy_model = policy_model,
    )
   
    rewards = eval_episode["rewards"]
    proxy = None
    if reward_model is not None: 
        proxy = reward_model(eval_episode["observations"].flatten(0,1), eval_episode["actions"].flatten(0,1)).view(eval_episode["rewards"].shape)
    if  world_model is not None:
        proxy = world_model(eval_episode["observations"][0].unsqueeze(1), eval_episode["actions"][:world_model.steps].transpose(0,1))[0].transpose(0,1)
        rewards = rewards[:world_model.steps]

    policy_model.train()
    if reward_model: reward_model.train()
    if world_model :  world_model.train()

    logger.info(json.dumps({
        "episode"            : episode,
        "done"               : eval_episode["dones"][-1,:,0].sum().int().item(),
        "reward"             : rewards.sum().item() / envs,
    } | ({} if proxy is None else {
        "reward_acc"         : proxy.isclose(rewards, atol=.1).float().mean().item(),
    })))

    return {
        "rewards" : rewards.sum().item() / envs,
        "max_reward" : eval_episode["max_reward"]
    }


