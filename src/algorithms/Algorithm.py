import click

@click.group
def algorithm(): pass

class Algorithm:
    def __init__(self): pass
    def evaluate(self): pass
    def start(self): pass
    def step(self, episode): pass
    def end(self): pass
