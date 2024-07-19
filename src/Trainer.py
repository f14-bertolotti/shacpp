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
    @click.option("--detect-anomaly" , "detect_anamaly" , type=bool , default=False , help="True if torch detect anamaly should be enabled" )
    @click.option("--seed"           , "seed"           , type=int  , default=42    , help="seed to the run"                                )
    @click.option("--episodes"       , "episodes"       , type=int  , default=1000  , help="max number of training episodes"                )
    @click.option("--etc"            , "etc"            , type=int  , default=10    , help="epochs to validation"                           )
    @click.pass_obj
    @staticmethod
    def train(trainer, seed, episodes, detect_anamaly, etc):

        torch.autograd.set_detect_anomaly(detect_anamaly)
        utils.seed_everything(seed)

        trainer.algorithm.start()
        for callback in trainer.callbacks: callback.start(locals())

        for episode in range(1, episodes + 1):
            trainer.algorithm.step(episode)

            if episode % etc == 0: trainer.algorithm.evaluate()
            
        for callback in trainer.callbacks: callback.end(locals())
        trainer.algorithm.end()


