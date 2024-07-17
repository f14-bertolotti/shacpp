from callbacks.proxied import make_checkpointer
from callbacks.proxied import make_save_best
from callbacks.proxied import make_validate

def make_command(where, root):

    @root.group()
    def proxied(): pass

    make_checkpointer(where=where, root=proxied)
    make_save_best   (where=where, root=proxied)
    make_validate    (where=where, root=proxied)

    
