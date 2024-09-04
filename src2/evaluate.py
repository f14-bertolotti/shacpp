import json, torch
from unroll import unroll

@torch.no_grad
def evaluate(
        episode,
        policy_model,
        world,
        steps,
        envs,
        logger
    ):
    eval_episode = unroll(
        observations = None, 
        world        = world,
        unroll_steps = steps,
        policy_model = policy_model,
    )
   
    logger.info(json.dumps({
        "episode"            : episode,
        "done"               : eval_episode["dones"][-1,:,0].sum().int().item(),
        "reward"             : eval_episode["rewards"].sum().item() / envs,
    }))

    return {
        "rewards" : eval_episode["rewards"].sum().item() / envs
    }


