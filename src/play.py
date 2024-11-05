from vmas.simulator.environment import Wrapper
from vmas.interactive_rendering import InteractiveEnv
from vmas.make_env import make_env
import models
import click
import torch

class Game(InteractiveEnv):
    def __init__(self, *args, reward_model = None, **kwargs):
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
        self.reward_model = reward_model
        super().__init__(*args, **kwargs)

    def _cycle(self):
        while True:

            if self.reset:
                self.iteration = 0
                self.env.reset()
                self.reset = False

            action_list = [[0.0] * agent.action_size for agent in self.agents]
            action_list[self.current_agent_index][
                : self.agents[self.current_agent_index].dynamics.needed_action_size
            ] = self.u[
                : self.agents[self.current_agent_index].dynamics.needed_action_size
            ]

            obs, rew, done, info = self.env.step(action_list)

            print("="*100)
            print("observations")
            print(torch.stack(obs))

            if self.reward_model is not None:
                observations = torch.stack(obs)
                actions      = torch.tensor(action_list)
                proxy = self.reward_model(observations.float(), actions.float())

                print("proxy:", proxy)
            print("reward:", rew)

            self.env.render(
                mode="rgb_array" if self.save_render else "human",
                visualize_when_rgb=True,
            )

            self.reset = done



@click.command()
@click.option("--name"        , "name"        , type=str          , default="transport" , help="scenario name"               )
@click.option("--agents"      , "agents"      , type=int          , default=3           , help="num. agents"                 )
@click.option("--reward-path" , "reward_path" , type=click.Path() , default=None        , help="reward model statedict path" )
@click.option("--keyarg"      , "keyargs"     , type=(str,int)    , default=None        , help="additional arguments", multiple=True)
def game(
    name        ,
    agents      ,
    reward_path ,
    keyargs     ,
):

    reward_model = None
    if reward_path is not None:
        checkpoint = torch.load(reward_path, weights_only=True) 
        reward_model = models.RewardAFO(
             observation_size = 11     ,
             action_size      = 2      ,
             agents           = agents ,
             steps            = 32     ,
             layers           = 1      ,
             hidden_size      = 2048   ,
             dropout          = 0      ,
             activation       = "ReLU" ,
             device           = "cpu"
        )
        reward_model.load_state_dict({k.replace("_orig_mod.",""):v for k,v in checkpoint["reward_state_dict"].items()})

    Game(
        make_env(
            scenario = name        ,
            num_envs = 1           ,
            n_agents = agents      ,
            device   = "cpu"       ,
            seed     = 0           ,
            wrapper  = Wrapper.GYM ,
            **dict(keyargs)
        ),
        render_name = name          ,
        reward_model = reward_model ,
    )

if __name__ == "__main__":
    game()
    
    
