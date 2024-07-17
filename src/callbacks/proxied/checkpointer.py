import torch, click


def make_command(where, root):
    @root.group(invoke_without_command=True)
    @click.option("--path", "path", type=click.Path(), default="agent.pkl")
    @click.option("--ete" , "ete" , type=int         , default=10)
    @click.pass_obj
    def checkpointer(trainer, path, ete):
        def wrapper(episode, **kwargs):
            if episode % ete == 0: 
                torch.save({
                    "agentsd"  : trainer.agent.state_dict(),
                    "rewardsd" : trainer.environment.rewardnn.state_dict(),
                    "rms"      : trainer.environment.state_dict(),
                    "episode"  : episode
                }, path)
            return {}
    
        where.add_callback(wrapper)
    
