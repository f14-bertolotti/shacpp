from callbacks.trainlog           import make_command as make_trainlog
from callbacks.validate           import make_command as make_validate
from callbacks.checkpointer       import make_command as make_checkpointer
from callbacks.save_configuration import make_command as make_save_configuration
from callbacks.savebest           import make_command as make_save_best
from callbacks.bar                import make_command as make_bar
from callbacks.callback           import make_command as make_callback


from callbacks.proxied import make_proxied 
