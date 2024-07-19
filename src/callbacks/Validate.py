from callbacks import callback, Callback
from loggers   import File
import os, math, torch, click

class Validate(Callback):
    def __init__(self, trainer, logpath="valid.log", sdpath="best.pkl", steps=64, size=512, etc = 100):
        self.trainer = trainer
        self.steps   =   steps
        self.logpath = logpath
        self.sdpath  =  sdpath
        self.size    =    size
        self.etc     =     etc
        self.logger  = File(logpath)
        self.best    = float("-inf")

    def end_episode(self, data):
        if data["episode"] % self.etc != 0:  return
    
        epochs  = math.ceil(self.size / self.trainer.environment.envirs)
        envirs  = epochs * self.trainer.environment.envirs
        rewards = torch.zeros(envirs , self.steps, self.trainer.environment.agents)
        
        # evaluate
        self.trainer.agent.eval()
        for epoch in range(epochs):
            current_observation = self.trainer.environment.reset()
            for step in range(self.steps):
                agent_result = self.trainer.agent.get_action(self.trainer.environment.normalize(current_observation))
                envir_result = self.trainer.environment.step(agent_result["logits"])
                rewards[self.trainer.environment.envirs * epoch:self.trainer.environment.envirs * (epoch+1), step] = envir_result["reward"]
                current_observation = envir_result["observation"]
        self.trainer.agent.train()
    
        # log reward
        reward = rewards.sum(1).sum(1).mean().item()
        self.logger.log({"episode" : data["episode"], "reward" : reward})

        # save best model if one is found
        if reward > self.best:
            self.best = reward

            torch.save({
                "agentsd"  : self.trainer.agent.state_dict(),
                "envsd"    : self.trainer.environment.state_dict(),
                "episode"  : data["episode"],
                "reward"   : self.best,
            }, self.sdpath)


        # set trainer bar description
        if hasattr(self.trainer, "bar_description"): self.trainer.bar_description.update(eval_reward = reward, best_eval_reward = self.best)

@callback.group()
@click.option("--steps"   , "steps"   , type=int          , default=64          , help="number of steps of unroll for the evaluation" )
@click.option("--size"    , "size"    , type=int          , default=512         , help="number of envs for the evaluation"            )
@click.option("--logpath" , "logpath" , type=click.Path() , default="valid.log" , help="path for logging"                             )
@click.option("--sdpath"  , "sdpath"  , type=click.Path() , default="best.pkl"  , help="path for saving the best model"               )
@click.option("--etc"     , "etc"     , type=int          , default=100         , help="epochs before an evaluation"                  )
@click.pass_obj
def validate(trainer, steps, size, logpath, sdpath, etc):
    trainer.add_callback(
        val := Validate(
            trainer = trainer,
            logpath = logpath,
            sdpath  =  sdpath,
            steps   =   steps,
            size    =    size,
            etc     =     etc,
        )
    )

    if os.path.isfile(sdpath):
        val.best = torch.load(sdpath)["reward"]
