
import experiments
import click

@click.group()
def cli(): pass

@cli.command()
def shacwm_a3(): experiments.shacwm_a3()

@cli.command()
def shacrm_a3(): experiments.shacrm_a3()

@cli.command()
def ppo_a3(): experiments.ppo_a3()

if __name__ == "__main__":
    cli()


