import click, torch

def add_adam_command(base, srcnav=lambda x:x, tgtnav=lambda x:x.agent):
    @base.group(invoke_without_command=True)
    @click.option("--learning-rate", "lr"          , type=float          , default=1e-4)
    @click.option("--betas"        , "betas"       , type=(float, float) , default=(.9, .99))
    @click.option("--weight-decay" , "weight_decay", type=float          , default=0)
    @click.pass_obj
    def adam(trainer, lr, betas, weight_decay):
        srcnav(trainer).set_optimizer(torch.optim.Adam(tgtnav(trainer).parameters(), lr=lr, betas=betas, weight_decay=weight_decay))


