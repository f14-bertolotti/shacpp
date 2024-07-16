from callbacks import callback
import loggers, click, torch, tqdm, math

@callback.group(invoke_without_command=True)
@click.option("--steps" , "steps" , type=int          , default=64  , help="number of steps of unroll for the evaluation")
@click.option("--size"  , "size"  , type=int          , default=512 , help="number of envs for the evaluation")
@click.option("--path"  , "path"  , type=click.Path() , default=512 , help="path for logging")
@click.option("--ete"   , "ete"   , type=int          , default=100 , help="epochs before an evaluation")
@click.pass_obj
def evaluate_sr(trainer, steps, size, path, ete):

    file_logger = loggers.File(path)

    def wrapped(episode, **kwargs):
        if episode % ete != 0:  return {}

        epochs  = math.ceil(size / trainer.environment.envirs)
        envirs  = epochs * trainer.environment.envirs
        real_rewards      = torch.zeros(envirs , steps, trainer.environment.agents)
        surrogate_rewards = torch.zeros(envirs , steps, trainer.environment.agents)
        
        for epoch in range(epochs):
            current_observation = trainer.environment.reset()
            for step in range(steps):
                agent_result = trainer.agent.get_action(trainer.environment.normalize(current_observation))
                envir_result = trainer.environment.step(oldobs=current_observation, action=agent_result["logits"])
                surrogate_rewards[trainer.environment.envirs * epoch:trainer.environment.envirs * (epoch+1), step] = envir_result["surrogate_reward"]
                real_rewards     [trainer.environment.envirs * epoch:trainer.environment.envirs * (epoch+1), step] = envir_result["real_reward"]
                current_observation = envir_result["observation"]

        avg_real_reward      = real_rewards     .sum(1).sum(1).mean().item()
        avg_surrogate_reward = surrogate_rewards.sum(1).sum(1).mean().item()
        file_logger.log({"episode" : episode, "reward" : avg_real_reward, "surrogate_reward": avg_surrogate_reward})

        return {
            "eval_reward" : avg_real_reward,
        }

    trainer.add_callback(wrapped)

