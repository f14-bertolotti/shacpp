import utils, tqdm, torch, click


class Trainer:
    def __init__(self): self.callbacks = []

    def set_trajectory (self, value): self.trajectory  = value
    def set_loss       (self, value): self.loss        = value
    def set_agent      (self, value): self.agent       = value
    def set_optimizer  (self, value): self.optimizer   = value
    def set_scheduler  (self, value): self.scheduler   = value
    def set_environment(self, value): self.environment = value
    def set_storage    (self, value): self.storage     = value
    def set_logger     (self, value): self.logger      = value
    def add_callback   (self, value): self.callbacks.append(value)


    @click.command()
    @click.option("--detect-anomaly"         , "detect_anamaly" , type=bool         , default=False)
    @click.option("--batch-size"             , "batch_size"     , type=int          , default=256)
    @click.option("--max-grad-norm"          , "max_grad_norm"  , type=float        , default=1)
    @click.option("--updates"                , "updates"        , type=int          , default=1000)
    @click.option("--epochs"                 , "epochs"         , type=int          , default=4)
    @click.option("--seed"                   , "seed"           , type=int          , default=42)
    @click.pass_obj
    @staticmethod
    def train(trainer, seed, updates, epochs, batch_size, max_grad_norm, detect_anamaly):

        torch.autograd.set_detect_anomaly(detect_anamaly)
        utils.seed_everything(seed)

        storage = None
        for update in (tbar:=tqdm.tqdm(range(1, updates + 1))):

            trajectory_result = trainer.trajectory(
                agent       = trainer.agent,
                environment = trainer.environment,
                storage     = trainer.storage,
                prev_trajectories = storage
            )
            storage = trajectory_result["storage"]

            dataloader = torch.utils.data.DataLoader(
                storage,
                collate_fn = storage.collate_fn,
                batch_size = batch_size,
                shuffle    = True
            )

            for epoch in range(epochs):
                for step, batch in enumerate(dataloader):
                    trainer.optimizer.zero_grad()

                    result = trainer.agent.get_action_and_value(observation = batch["observations"], action = batch["actions"])
                    losses = trainer.loss(new = result, old = batch)

                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(trainer.agent.critic.parameters(), max_grad_norm)
                    trainer.optimizer.step()

                    for callback in trainer.callbacks: callback(**locals())

                trainer.scheduler.step()


