import click, torch

def add_sgd_command(base, srcnav=lambda x:x, tgtnav=lambda x:x.agent, attrname = "set_optimizer"):
    @base.group(invoke_without_command=True)
    @click.option("--learning-rate", "lr"          , type=float          , default=1e-4)
    @click.pass_obj
    def sgd(trainer, lr):
        getattr(srcnav(trainer), attrname)(torch.optim.SGD(tgtnav(trainer).parameters(), lr=lr))


