from algorithms.ppo import ppo
import click, utils, torch 

agents = 3
obs_size = 13
envs = 512

class Trajectory:
    def __init__(self, gamma=.99, gaelambda=.95, steps=64, feedback=False):
        self.gamma, self.gaelambda, self.steps, self.feedback = gamma, gaelambda, steps, feedback

        self.obs      = torch.zeros((self.steps, agents, envs, obs_size)).to("cuda:0")
        self.actions  = torch.zeros((self.steps, agents, envs, 2)).to("cuda:0")
        self.logprobs = torch.zeros((self.steps, agents, envs)).to("cuda:0")
        self.rewards  = torch.zeros((self.steps, agents, envs)).to("cuda:0")
        self.dones    = torch.ones ((self.steps, envs), dtype=torch.bool).to("cuda:0")
        self.values   = torch.zeros((self.steps, agents, envs)).to("cuda:0")

    def __call__(self, environment, agent, storage):

        # TRY NOT TO MODIFY: start the game
        next_obs = environment.reset(prev=self.obs[-1], dones=self.dones[-1]) if self.feedback else environment.reset()
        next_done = torch.zeros(envs).to("cuda:0")

        for step in range(0, self.steps):
            self.obs[step] = torch.stack(next_obs)
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            self.values[step] = value.squeeze(-1)
            self.actions[step] = action
            self.logprobs[step] = logprob

            next_obs, reward, done, info = environment.step(action)
            self.rewards[step] = torch.stack(reward)
            print(self.rewards.sum())
            next_obs, next_done = next_obs, done

        environment.compute_statistics(self.obs)
        self.obs = environment.normalize(self.obs)

        obs      = self.obs       .reshape ((self.steps, agents * envs, obs_size))
        actions  = self.actions   .reshape ((self.steps, agents * envs, 2))
        logprobs = self.logprobs  .reshape ((self.steps, agents * envs))
        rewards  = self.rewards   .reshape ((self.steps, agents * envs))
        dones    = self.dones     .repeat  (1,agents)
        next_done= next_done      .repeat  (1,agents)
        values   = self.values    .reshape ((self.steps, agents * envs))

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs)["values"].reshape(1, -1).flatten()
            advantages = torch.zeros_like(rewards).to("cuda:0")
            lastgaelam = 0
            for t in reversed(range(self.steps)):
                if t == self.steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]

                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gaelambda * nextnonterminal * lastgaelam
            returns = advantages + values


        b_obs        = obs       .reshape ((self.steps, agents, envs, obs_size)).transpose(1,2).flatten(0,1)
        b_actions    = actions   .reshape ((self.steps, agents, envs, 2)).transpose(1,2).flatten(0,1)
        b_logprobs   = logprobs  .reshape ((self.steps, agents, envs)).transpose(1,2).flatten(0,1)
        b_rewards    = rewards   .reshape ((self.steps, agents, envs)).transpose(1,2).flatten(0,1)
        b_values     = values    .reshape ((self.steps, agents, envs)).transpose(1,2).flatten(0,1)
        b_advantages = advantages.reshape ((self.steps, agents, envs)).transpose(1,2).flatten(0,1)
        b_returns    = returns   .reshape ((self.steps, agents, envs)).transpose(1,2).flatten(0,1)

        storage.dictionary = ({"rewards":b_rewards, "observations" : b_obs, "logprobs":b_logprobs, "actions":b_actions, "advantages":b_advantages, "returns":b_returns, "values":b_values})
        return storage


@ppo.group()
def trajectory(): pass

@trajectory.group(invoke_without_command=True)
@click.option("--gamma"     , "gamma"     , type=float , default=.99)
@click.option("--gaelambda" , "gaelambda" , type=float , default=.95)
@click.option("--steps"     , "steps"     , type=int   , default=64)
@click.option("--feedback"  , "feedback"  , type=bool  , default=False)
@click.pass_obj
def default(trainer, gamma, gaelambda, steps, feedback):
    trainer.algorithm.set_trajectory(
        Trajectory( 
            steps     = steps,
            gamma     = gamma,
            gaelambda = gaelambda,
            feedback  = feedback
        )
    )
