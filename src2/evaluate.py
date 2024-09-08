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
        reward_model = None
    ):
    eval_episode = unroll(
        observations = None, 
        world        = world,
        unroll_steps = steps,
        policy_model = policy_model,
    )
   
    proxy = None
    if reward_model is not None: 
        proxy = reward_model(eval_episode["observations"], eval_episode["actions"])
    rewards = eval_episode["rewards"]

    logger.info(json.dumps({
        "episode"            : episode,
        "done"               : eval_episode["dones"][-1,:,0].sum().int().item(),
        "reward"             : rewards.sum().item() / envs,
    } | ({} if proxy is None else {
        "reward_acc" : proxy.isclose(rewards, atol=.1).float().mean().item(),
        "reward_acc_nz" : proxy[rewards > 0].isclose(rewards[rewards > 0], atol=.1).float().mean().item(),
    })))

    return {
        "rewards" : rewards.sum().item() / envs
    }


