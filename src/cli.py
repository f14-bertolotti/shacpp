from Trainer import Trainer
from Vizualize import viz
from environments import environment
from algorithms   import   algorithm
from callbacks    import    callback
from agents       import       agent
import jsonpickle
import click


@click.group(invoke_without_command=True, context_settings={'show_default': True})
@click.pass_context
def cli(context):
    if not context.obj: context.obj = Trainer()

@staticmethod
@cli.group(invoke_without_command=True)
@click.option("--path", "path", type=click.Path(), default="")
@click.pass_obj
def save_configuration(trainer, path):
    with open(path, "w") as file: file.write(str(jsonpickle.encode(trainer,indent=4)))

# node commands, can be called in chain by navigating upwards
# in the command tree
cli.add_command(environment)
cli.add_command(algorithm)
cli.add_command(callback)
cli.add_command(agent)

# put the cli group command as last command in the command tree
# so that commands can be chained 
def visit(command):
    if isinstance(command, click.core.Group) and not command.commands: return [command]
    elif isinstance(command, click.core.Group) and command.commands: return [command] + [c for cmd in command.commands.values() for c in visit(cmd)]
    else: return []
for grp in visit(cli): grp.add_command(cli)

cli.add_command(cli)

# leaf commands, these are final
cli.add_command(Trainer.train)
cli.add_command(viz)

if __name__ == "__main__":
    cli()

