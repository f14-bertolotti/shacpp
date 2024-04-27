import utils, tqdm, torch, click


class Trainer:

    def set_trajectory (self, value): self.trajectory  = value
    def set_loss       (self, value): self.loss        = value
    def set_agent      (self, value): self.agent       = value
    def set_optimizer  (self, value): self.optimizer   = value
    def set_scheduler  (self, value): self.scheduler   = value
    def set_environment(self, value): self.environment = value
    def set_storage    (self, value): self.storage     = value
    def set_logger     (self, value): self.logger      = value

    @click.command()
    @click.option("--detect-anomaly"         , "detect_anamaly" , type=bool         , default=False)
    @click.option("--batch-size"             , "batch_size"     , type=int          , default=256)
    @click.option("--max-grad-norm"          , "max_grad_norm"  , type=float        , default=.5)
    @click.option("--updates-to-checkpoints" , "utc"            , type=int          , default=10)
    @click.option("--updates"                , "updates"        , type=int          , default=1000)
    @click.option("--epochs"                 , "epochs"         , type=int          , default=4)
    @click.option("--seed"                   , "seed"           , type=int          , default=42)
    @click.option("--checkpoint-path"        , "checkpoint_path", type=click.Path() , default= "./agent.pkl")
    @click.pass_obj
    @staticmethod
    def train(trainer, seed, updates, epochs, batch_size, max_grad_norm, utc, checkpoint_path, detect_anamaly):
        utils.seed_everything(seed)
        torch.autograd.set_detect_anomaly(detect_anamaly)

        for update in (tbar:=tqdm.tqdm(range(1, updates + 1))):

            storage = trainer.trajectory(
                agent       = trainer.agent,
                environment = trainer.environment,
                storage     = trainer.storage
            )

            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    storage.observations.flatten(0,1),
                    storage.actions     .flatten(0,1),
                    storage.logprobs    .flatten(),
                    storage.rewards     .flatten(),
                    storage.values      .flatten(),
                    storage.returns     .flatten(),
                    storage.advantages  .flatten()
                ), 
                batch_size = batch_size,
                shuffle    = True
            )

            for epoch in range(epochs):
                for i, (observations, actions, oldlogprobs, rewards, oldvalues, returns, advantages) in enumerate(dataloader):

                    result = trainer.agent.get_action_and_value(observations, action=actions)

                    losses = trainer.loss(**(result | {
                        "observations" : observations,
                        "actions"      : actions,
                        "oldlogprobs"  : oldlogprobs,
                        "rewards"      : rewards,
                        "oldvalues"    : oldvalues,
                        "returns"      : returns,
                        "advantages"   : advantages
                    }))

                    trainer.optimizer.zero_grad()
                    (loss := losses["loss"]).backward()
                    torch.nn.utils.clip_grad_norm_(trainer.agent.parameters(), max_grad_norm)
                    trainer.optimizer.step()

                    tbar.set_description(f"{update}-{epoch}-{i}, lr:{trainer.scheduler.get_last_lr()[0]:7.6f}, l:{loss.item():7.4f}, r:{rewards.mean():7.4f}")
                    trainer.logger.log({
                        "loss"   : loss.item(),
                        "reward" : rewards.mean(),
                        "update" : update,
                        "epoch"  : epoch,
                        "step"   : i,
                    })

                trainer.scheduler.step()

            if update % utc == 0: torch.save({"agentsd" : trainer.agent.state_dict()}, checkpoint_path)

