from algorithms.ppo import compute_advantages, compute_returns, ppo
from algorithms import Options
import click, torch 

class Trajectory:
    def __init__(self, trainer, gamma=.99, gaelm=.95, steps=64, utr=1):
    
        self.trainer = trainer
        self.gamma = gamma
        self.gaelm = gaelm
        self.steps = steps
        self.utr   =   utr

        self.device = trainer.environment.device
        self.envirs = trainer.environment.envirs
        self.agents = trainer.environment.agents

        self.obs      = torch.zeros((self.envirs , self.steps, self.agents, trainer.environment.get_observation_size()), device=self.device)
        self.actions  = torch.zeros((self.envirs , self.steps, self.agents, trainer.environment.get_action_size()     ), device=self.device)
        self.logprobs = torch.zeros((self.envirs , self.steps, self.agents                                            ), device=self.device)
        self.rewards  = torch.zeros((self.envirs , self.steps, self.agents                                            ), device=self.device)
        self.dones    = torch.ones ((self.envirs , self.steps                                                         ), device=self.device, dtype=torch.bool)
        self.values   = torch.zeros((self.envirs , self.steps, self.agents                                            ), device=self.device)

        self.rollouts = 0

    def reset_storage(self):
        """ reset storage into a full zero state """
        self.obs     .zero_()
        self.actions .zero_()
        self.logprobs.zero_()
        self.rewards .zero_()
        self.dones   .zero_()
        self.values  .zero_()

    def __call__(self, environment, agent):
 
        next_obs  = environment.reset()                          if self.rollouts % self.utr == 0 else self.obs  [:,-1].clone()
        next_done = torch.zeros(self.envirs, device=self.device) if self.rollouts % self.utr == 0 else self.dones[:,-1].clone()
        self.rollouts += 1
        self.reset_storage()

        for callback in self.trainer.callbacks: callback.start_trajectory(locals())

        # unroll trajectories
        for step in range(0, self.steps):
            for callback in self.trainer.callbacks: callback.start_trajectory_step(locals())

            self.obs  [:, step] = next_obs
            self.dones[:, step] = next_done
            
            with torch.no_grad(): agent_result = agent.get_action_and_value(next_obs)

            self.values  [:, step] = agent_result["values"  ]
            self.actions [:, step] = agent_result["actions" ]
            self.logprobs[:, step] = agent_result["logprobs"]
            envir_result = environment.step(oldobs=self.obs[:,step], action=agent_result["actions"])
            self.rewards [:, step] = envir_result["reward"  ]

            next_obs, next_done = envir_result["observation"], envir_result["done"]

            for callback in self.trainer.callbacks: callback.end_trajectory_step(locals())

        # compute trajectories statistics and normalize
        environment.update_statistics(self.obs)
        self.obs = environment.normalize(self.obs)

        # compute advantages and returns
        advantages = compute_advantages(
            agent     = agent          ,
            rewards   = self.rewards   ,
            next_obs  = next_obs       ,
            values    = self.values    ,
            dones     = self.dones     ,
            next_done = next_done      ,
            gamma     = self.gamma     ,
            gaelambda = self.gaelm     ,
        )
        returns = compute_returns   (advantages, self.values)


        for callback in self.trainer.callbacks: callback.end_trajectory(locals())
       
        return {
            "observations" : self.obs      ,
            "rewards"      : self.rewards  ,
            "logprobs"     : self.logprobs ,
            "actions"      : self.actions  ,
            "values"       : self.values   ,
            "advantages"   : advantages    ,
            "returns"      : returns       ,
            "dones"        : self.dones    ,
        }


@ppo.group()
def trajectory(): pass

@trajectory.group(invoke_without_command=True)
@Options.trajectory
@click.pass_obj
def default(trainer, gamma, gaelambda, steps, utr):
    trainer.algorithm.set_trajectory(
        Trajectory( 
            trainer = trainer   ,
            steps   = steps     ,
            gamma   = gamma     ,
            gaelm   = gaelambda ,
            utr     = utr       ,
        )
    )
