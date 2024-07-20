from callbacks.proxied import proxied
import os, math, torch, click, loggers, callbacks



class Validate(callbacks.Callback):
    def __init__(self, trainer, logpath="valid.log", sdpath="best.pkl", steps=64, size=512, atol=.1, etc = 100):
        self.trainer = trainer
        self.steps   =   steps
        self.logpath = logpath
        self.sdpath  =  sdpath
        self.size    =    size
        self.atol    =    atol
        self.etc     =     etc
        self.logger  = loggers.File(logpath)
        self.best    = float("-inf")

    def end_episode(self, data):
        if data["episode"] % self.etc != 0:  return
    
        epochs  = math.ceil(self.size / self.trainer.environment.envirs)
        envirs  = epochs * self.trainer.environment.envirs
        real_reward  = torch.zeros(envirs , self.steps, self.trainer.environment.agents)
        proxy_reward = torch.zeros(envirs , self.steps, self.trainer.environment.agents)
        
        # evaluate
        self.trainer.agent.eval()
        for epoch in range(epochs):
            current_observation = self.trainer.environment.reset()
            for step in range(self.steps):
                agent_result = self.trainer.agent.get_action(self.trainer.environment.normalize(current_observation))
                envir_result = self.trainer.environment.step(agent_result["logits"])
                real_reward [self.trainer.environment.envirs * epoch:self.trainer.environment.envirs * (epoch+1), step] = envir_result["real_reward"]
                proxy_reward[self.trainer.environment.envirs * epoch:self.trainer.environment.envirs * (epoch+1), step] = envir_result["proxy_reward"]
                current_observation = envir_result["observation"]
        self.trainer.agent.train()
    
        # log metrics
        avg_real_reward  = real_reward .sum(1).sum(1).mean().item()
        avg_proxy_reward = proxy_reward.sum(1).sum(1).mean().item()
        avg_proxy_loss   = ((real_reward - proxy_reward)**2).mean().item()
        avg_proxy_acc    = real_reward.isclose(proxy_reward, atol=self.atol).float().mean().item()

        self.logger.log({
            "episode"      : data["episode"],
            "reward"       : avg_real_reward,
            "proxy_reward" : avg_proxy_reward,
            "proxy_loss"   : avg_proxy_loss,
            "proxy_acc"    : avg_proxy_acc,
        })

        # save best model if one is found
        if avg_real_reward > self.best:
            self.best = avg_real_reward

            torch.save({
                "agentsd"  : self.trainer.agent.state_dict(),
                "envsd"    : self.trainer.environment.state_dict(),
                "episode"  : data["episode"],
                "reward"   : self.best,
            }, self.sdpath)

        # set trainer bar description
        if hasattr(self.trainer, "bar_description"): self.trainer.bar_description.update(eval_reward = avg_real_reward, best_eval_reward = self.best)

@proxied.group()
@click.option("--steps"   , "steps"   , type=int          , default=64          , help="number of steps of unroll for the evaluation"  )
@click.option("--size"    , "size"    , type=int          , default=512         , help="number of envs for the evaluation"             )
@click.option("--logpath" , "logpath" , type=click.Path() , default="valid.log" , help="path for logging"                              )
@click.option("--sdpath"  , "sdpath"  , type=click.Path() , default="best.pkl"  , help="path for saving the best model"                )
@click.option("--etc"     , "etc"     , type=int          , default=100         , help="epochs before an evaluation"                   )
@click.option("--atol"    , "atol"    , type=float        , default=.1          , help="tolerance in proxy reward accuracy computation")
@click.pass_obj
def validate(trainer, steps, size, logpath, sdpath, atol, etc):
    trainer.add_callback(
        val := Validate(
            trainer = trainer,
            logpath = logpath,
            sdpath  =  sdpath,
            steps   =   steps,
            size    =    size,
            atol    =    atol,
            etc     =     etc,
        )
    )

    if os.path.isfile(sdpath):
        val.best = torch.load(sdpath)["reward"]
