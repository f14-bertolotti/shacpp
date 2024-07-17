import click
from callbacks.proxied import make_proxied
from callbacks import make_trainlog
from callbacks import make_validate
from callbacks import make_checkpointer
from callbacks import make_save_configuration
from callbacks import make_save_best
from callbacks import make_bar

def make_command(where, root):

    @root.group()
    def callback(): pass
    
    make_trainlog           ( where = where, root = callback)
    make_validate           ( where = where, root = callback)
    make_checkpointer       ( where = where, root = callback)
    make_save_configuration ( where = where, root = callback)
    make_save_best          ( where = where, root = callback)
    make_bar                ( where = where, root = callback)
    make_proxied            ( where = where, root = callback)


