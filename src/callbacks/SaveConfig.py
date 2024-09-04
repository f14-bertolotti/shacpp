from callbacks import callback, Callback
import jsonpickle, click

class SaveConfig(Callback):
    def __init__(self, trainer, path="agent.pkl"):
        self.trainer = trainer
        self.path    =    path

    def start(self, data):
        pass
        #with open(self.path, "w") as file: 
        #    file.write(str(jsonpickle.encode(self.trainer,indent=4)))
 

@callback.group
@click.option("--path", "path", type=click.Path(), default="agent.pkl")
@click.pass_obj
def saveconfig(trainer, path):
    trainer.add_callback(
        SaveConfig(
            trainer = trainer,
            path    =    path,
        )
    )

