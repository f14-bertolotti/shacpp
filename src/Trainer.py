import utils, tqdm, torch, click


class Trainer:
    def __init__(self): self.callbacks = []

    def set_loss        (self, value): self.loss        = value
    def set_agent       (self, value): self.agent       = value
    def set_environment (self, value): self.environment = value
    def set_storage     (self, value): self.storage     = value
    def set_algorithm   (self, value): self.algorithm   = value
    def set_evaluator   (self, value): self.evaluator   = value
    def add_callback    (self, value): self.callbacks.append(value)

    @click.command()
    @click.option("--detect-anomaly" , "detect_anamaly" , type=bool , default=False)
    @click.option("--seed"           , "seed"           , type=int  , default=42)
    @click.option("--episodes"       , "episodes"       , type=int  , default=1000)
    @click.pass_obj
    @staticmethod
    def train(trainer, seed, episodes, detect_anamaly):

        torch.autograd.set_detect_anomaly(detect_anamaly)
        utils.seed_everything(seed)

        trainer.algorithm.start()

        for episode in (tbar:=tqdm.tqdm(range(1, episodes + 1))):
            train_result = trainer.algorithm.step(episode)
            for callback in trainer.callbacks: callback(**locals())

        trainer.algorithm.end()


