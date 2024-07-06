from algorithms.ppo import compute_advantages, compute_returns, ppo
import click, utils, torch 

class Trajectory:
    def __init__(self, trainer, gamma=.99, gaelambda=.95, steps=64, reset_prb=None, feedback=False):
        self.gamma, self.gaelambda, self.steps, self.feedback = gamma, gaelambda, steps, feedback
        self.device = trainer.environment.device
        self.envs   = trainer.environment.envs
        self.agents = trainer.environment.agents
        self.reset_prb = reset_prb

        self.obs      = torch.zeros((self.envs , self.steps , self.agents, trainer.environment.get_observation_size()), device=self.device)
        self.actions  = torch.zeros((self.envs , self.steps , self.agents, trainer.environment.get_action_size()     ), device=self.device)
        self.logprobs = torch.zeros((self.envs , self.steps , self.agents                                            ), device=self.device)
        self.rewards  = torch.zeros((self.envs , self.steps , self.agents                                            ), device=self.device)
        self.dones    = torch.ones ((self.envs , self.steps                                                          ), device=self.device, dtype=torch.bool)
        self.values   = torch.zeros((self.envs , self.steps , self.agents                                            ), device=self.device)

    def reset_storage(self):
        """ reset storage into a full zero state """
        self.obs     .zero_()
        self.actions .zero_()
        self.logprobs.zero_()
        self.rewards .zero_()
        self.dones   .zero_()
        self.values  .zero_()

    def __call__(self, environment, agent):
        self.reset_storage()
 
        next_obs  = environment.reset(prev=self.obs[:,-1], dones=self.dones[:,-1], reset_prb=self.reset_prb) if self.feedback else environment.reset()
        next_done = torch.zeros(self.envs, device=self.device)

        # unroll trajectories
        for step in range(0, self.steps):
            self.obs  [:, step] = next_obs
            self.dones[:, step] = next_done

            with torch.no_grad(): agent_result = agent.get_action_and_value(next_obs)

            self.values  [:, step] = agent_result["values"  ]
            self.actions [:, step] = agent_result["actions" ]
            self.logprobs[:, step] = agent_result["logprobs"]
            envir_result = environment.step(agent_result["actions"])
            self.rewards [:, step] = envir_result["reward"  ]

            next_obs, next_done = envir_result["observation"], envir_result["done"]

        # compute trajectories statistics and normalize
        environment.update_statistics(self.obs)
        self.obs = environment.normalize(self.obs)

        # compute advantages and returns
        advantages = compute_advantages(
            agent     = agent,
            rewards   = self.rewards,
            next_obs  = next_obs,
            values    = self.values,
            dones     = self.dones,
            next_done = next_done,
            gamma     = self.gamma,
            gaelambda = self.gaelambda
        )
        returns = compute_returns   (advantages, self.values)
       
        return {
            "observations" : self.obs,
            "rewards"      : self.rewards,
            "logprobs"     : self.logprobs,
            "actions"      : self.actions,
            "values"       : self.values,
            "advantages"   : advantages,
            "returns"      : returns,
            "dones"        : self.dones,
        }


@ppo.group()
def trajectory(): pass

@trajectory.group(invoke_without_command=True)
@click.option("--gamma"     , "gamma"     , type=float , default=.99)
@click.option("--gaelambda" , "gaelambda" , type=float , default=.95)
@click.option("--steps"     , "steps"     , type=int   , default=64)
@click.option("--feedback"  , "feedback"  , type=bool  , default=False)
@click.option("--reset-prb" , "reset_prb" , type=float , default=None)
@click.pass_obj
def default(trainer, gamma, gaelambda, steps, feedback, reset_prb):
    trainer.algorithm.set_trajectory(
        Trajectory( 
            trainer   = trainer,
            steps     = steps,
            gamma     = gamma,
            gaelambda = gaelambda,
            reset_prb = reset_prb,
            feedback  = feedback
        )
    )
