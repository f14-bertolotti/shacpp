from callbacks import callback, Callback
import click, tqdm

class Description:
    def __init__(self):
        self.loss             = float("NaN")
        self.episode          = float("NaN")
        self.epoch            = float("NaN")
        self.step             = float("NaN")
        self.eval_reward      = float("NaN")
        self.best_eval_reward = float("NaN")

    def __str__(self):
        return f"{self.episode}/{self.epoch}/{self.step}, lss:{self.loss:2.4f}, rew:{self.eval_reward:2.4f}, best:{self.best_eval_reward:2.4f}"

    def update(self, loss = None, episode = None, epoch = None, step = None, eval_reward = None, best_eval_reward = None):
        if loss    : self.loss                     = loss
        if episode : self.episode                  = episode
        if epoch   : self.epoch                    = epoch
        if step    : self.step                     = step
        if eval_reward: self.eval_reward           = eval_reward
        if best_eval_reward: self.best_eval_reward = best_eval_reward


class Barline(Callback):
    def __init__(self, trainer):
        self.description = Description()
        self.trainer = trainer
        self.trainer.bar_description = self.description

    def start(self, data): 
        self.bar = tqdm.tqdm(total=data["episodes"])

    def end_episode(self, data):
        self.bar.update(1)

    def end_step(self, data):
        self.description.update(episode = data["episode"], epoch = data["epoch"], step = data["step"])
        self.bar.set_description(str(self.description))
    
    def end_epoch(self, data):
        self.description.update(loss = data["lossval"].item())

@callback.group()
@click.pass_obj
def barline(trainer):
    trainer.add_callback(
        Barline(
            trainer = trainer
        )
    )

