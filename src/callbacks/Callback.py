import click

@click.group
def callback(): pass

class Callback:

    def __getattr__(self, name):
        return self.__dict__.get(name, lambda _:None)

