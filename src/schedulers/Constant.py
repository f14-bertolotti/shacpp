import click 

class ConstantScheduler:
    def step(self, *args, **kwargs): pass
    def get_last_lr(self): return None

def add_constant_command(base, srcnav=lambda x:x, tgtnav=lambda x:x):
    @base.group(invoke_without_command=True)
    @click.pass_obj
    def constant(trainer):
        srcnav(trainer).set_scheduler(ConstantScheduler())
        
    
    
