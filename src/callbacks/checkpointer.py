from callbacks import callback
import torch, click


@callback.group(invoke_without_command=True)
@click.option("--path", "path", type=click.Path(), default="agent.pkl")
@click.option("--ete" , "ete" , type=int         , default=10)
@click.pass_obj
def checkpointer(trainer, path, ete):
    def wrapper(episode, **kwargs):
        if episode % ete == 0: 
            torch.save({
                "agentsd"  : trainer.agent.state_dict(),
                "rms"      : trainer.environment.state_dict(),
                "episode"  : episode
            }, path)
        return {}

    trainer.add_callback(wrapper)

