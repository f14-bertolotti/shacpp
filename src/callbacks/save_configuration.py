from callbacks import callback
import jsonpickle, click


@callback.group(invoke_without_command=True)
@click.option("--path", "path", type=click.Path(), default="agent.pkl")
@click.pass_obj
def save_configuration(trainer, path):
    def wrapper(episode, **kwargs):
        if episode == 1: 
            with open(path, "w") as file: 
                file.write(str(jsonpickle.encode(trainer,indent=4)))

        return {}
    trainer.add_callback(wrapper)

