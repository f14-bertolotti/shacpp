import torch, click

def make_command(where, root):
    @root.group(invoke_without_command=True)
    @click.option("--path", "path", type=click.Path(), default="agent.pkl")
    @click.pass_obj
    def save_best(trainer, path):
        def wrapper(episode, prev_result, **kwargs):
            reward = prev_result.get("eval_reward", float("-inf"))
            if reward >= wrapper.best_reward:
                wrapper.best_reward = reward
                torch.save({
                    "agentsd"  : trainer.agent.state_dict(),
                    "rewardsd" : trainer.environment.rewardnn.state_dict(),
                    "rms"      : trainer.environment.state_dict(),
                    "episode"  : episode
                }, path)
            return {}
        wrapper.best_reward = float("-inf")
    
        where.add_callback(wrapper)
    
