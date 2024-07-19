from callbacks import callback, Callback
import click

class Greeting(Callback):

    def start(self, data): 
        print("="*10 + " start of training " + "="*10)


@callback.group()
@click.pass_obj
def greeting(trainer):
    trainer.add_callback(
        Greeting(
        )
    )

