import torch, click

def add_cosine_command(base, srcnav=lambda x:x, tgtnav=lambda x:x, attrname="set_scheduler"):
    @base.group(invoke_without_command=True)
    @click.option("--tmax"  , "tmax"  , type=float , default=100)
    @click.option("--lrmin" , "lrmin" , type=float , default=1e-5)
    @click.pass_obj
    def cosine(trainer, tmax, lrmin):
        getattr(srcnav(trainer), attrname)(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                tgtnav(trainer).optimizer, 
                T_max   = tmax,
                eta_min = lrmin
            )
        )
        
    
    
