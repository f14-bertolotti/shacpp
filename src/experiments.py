
import experiments
import click

@click.group()
def cli(): pass

@cli.command()
def shacwm_dispersion_a7(): experiments.shacwm_dispersion_a7()

@cli.command()
def shacwm_dispersion_a3(): experiments.shacwm_dispersion_a3()

@cli.command()
def shacrm_dispersion_a3(): experiments.shacrm_dispersion_a3()

@cli.command()
def shacrm_transport_a3(): experiments.shacrm_transport_a3()

@cli.command()
def shacwm_transport_a3(): experiments.shacwm_transport_a3()

@cli.command()
def ppo_a3(): experiments.ppo_a3()

if __name__ == "__main__":
    cli()


