from optimizers import optimizer
import click, torch

@optimizer.group(invoke_without_command=True)
@click.option("--lr"    , "lr"    , type=float          , default=1e-4)
@click.option("--betas" , "betas" , type=(float, float) , default=(.9, .99))
@click.pass_obj
def adam(trainer, lr, betas):
    trainer.set_optimizer(
        torch.optim.Adam(trainer.agent.parameters(), lr=lr, betas=betas)
    )

