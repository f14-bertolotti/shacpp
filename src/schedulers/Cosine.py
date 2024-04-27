from schedulers import scheduler
import torch, click

@scheduler.group(invoke_without_command=True)
@click.option("--tmax"  , "tmax"  , type=float , default=400)
@click.option("--lrmin" , "lrmin" , type=float , default=1e-5)
@click.pass_obj
def cosine(trainer, tmax, lrmin):
    trainer.set_scheduler(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, 
            T_max   = tmax,
            eta_min = lrmin
        )
    )
    


