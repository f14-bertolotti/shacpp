import json, torch
from unroll import unroll

@torch.no_grad
def evaluate(
        episode,
        policy_model,
        reward_model, 
        value_model,
        world,
        steps,
        envs,
        logger
    ):
    policy_model.eval()
    eval_episode = unroll(
        observations = None, 
        world        = world,
        unroll_steps = steps,
        policy_model = policy_model,
        reward_model = reward_model,
        value_model  = value_model
    )
    policy_model.train()
   
    logger.info(json.dumps({
        "episode"            : episode,
        "done"               : eval_episode["dones"][-1,:,0].sum().int().item(),
        "reward"             : eval_episode["rewards"].sum().item() / envs,
        "reward_loss"        : ((eval_episode["rewards"] - eval_episode["proxy_rewards"])**2).mean().item(),
        "reward_accuracy"    : torch.isclose(eval_episode["proxy_rewards"], eval_episode["rewards"], atol=.1).float().mean().item(),
        "reward_accuracy_nz" : torch.isclose(eval_episode["proxy_rewards"][eval_episode["rewards"]>0], eval_episode["rewards"][eval_episode["rewards"]>0], atol=.1).float().mean().item()
    }))

    return {
        "rewards" : eval_episode["rewards"].sum().item() / envs
    }


