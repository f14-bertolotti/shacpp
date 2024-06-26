import utils, tqdm, torch, click


class Trainer:
    def __init__(self): self.callbacks = []

    def set_loss       (self, value): self.loss        = value
    def set_agent      (self, value): self.agent       = value
    def set_environment(self, value): self.environment = value
    def set_storage    (self, value): self.storage     = value
    def set_logger     (self, value): self.logger      = value
    def add_callback   (self, value): self.callbacks.append(value)
    def set_algorithm  (self, value): self.algorithm    = value


    @click.command()
    @click.option("--detect-anomaly" , "detect_anamaly" , type=bool , default=False)
    @click.option("--seed"           , "seed"           , type=int  , default=42)
    @click.option("--epochs"         , "epochs"         , type=int  , default=1000)
    @click.pass_obj
    @staticmethod
    def train(trainer, seed, epochs, detect_anamaly):

        torch.autograd.set_detect_anomaly(detect_anamaly)
        utils.seed_everything(seed)

        trainer.algorithm.start()

        for update in (tbar:=tqdm.tqdm(range(1, epochs + 1))):

            trainer.algorithm.step(update, tbar)

        trainer.algorithm.end()


