from vmas.simulator.environment import Wrapper
from vmas.interactive_rendering import InteractiveEnv
from vmas.make_env import make_env
from vmas.simulator import rendering
import models
import click
import torch

def compute_angle(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    dot_product = torch.dot(v1, v2)
    norm_v1 = torch.norm(v1)
    norm_v2 = torch.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta

def cartesian_to_polar(cartesian_coords):
    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    polar_coords = torch.stack((r, theta), dim=-1)
    return polar_coords

class Game(InteractiveEnv):
    def __init__(self, *args, reward_model = None, value_model = None, **kwargs):
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
        self.reward_model = reward_model
        self.value_model  = value_model
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)

        super().__init__(*args, **kwargs)

    def _cycle(self):
        obs = [torch.tensor(o) for o in self.env.reset()]

        self.lineforms = [rendering.Transform() for agent in self.agents]
        self.lines = [rendering.Line(
            (0,0),
            (0,.1),
            width=2,
        ) for agent in self.agents]
        for line,form in zip(self.lines, self.lineforms): line.add_attr(form)
        for line in self.lines: self.env.unwrapped.viewer.add_geom(line)
        

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

            for agent, lineform in zip(self.agents,self.lineforms):
                lineform.set_translation(*agent.state.pos[0])

            prev = torch.stack(obs)
            obs, rew, done, info = self.env.step(action_list)
            obs = [torch.tensor(o) for o in obs]
            self._write_values(1, "rews: " + str(list(map(lambda x:f"{x:1.3f}", rew))))

            print("="*100)
            print("observations")
            print(torch.stack(obs))

            if self.reward_model is not None:
                observations = torch.stack(obs)
                actions      = torch.tensor(action_list, requires_grad=True)
                proxy = self.reward_model(observations.float(), actions.float(), prev.float())
                self._write_values(0, "alts: " + str(list(map(lambda x:f"{x:1.3f}", proxy.tolist()))))

                self.optimizer.zero_grad()
                proxy.mean().neg().backward()

                grads = -actions.grad
                angles = cartesian_to_polar(grads.float())
                for angle,lineform in zip(angles,self.lineforms): 
                    lineform.set_rotation(angle[1]-3.14/2 )
                print()

            if self.value_model is not None:
                observations = torch.stack(obs).unsqueeze(0)
                values = self.value_model(observations.float()).squeeze(0)
                self._write_values(2, "vals: " + str(list(map(lambda x:f"{x:1.3f}", values.tolist()))))


                print("proxy:", proxy)
            print("reward:", rew)


            self.env.render(
                mode="rgb_array" if self.save_render else "human",
                visualize_when_rgb=True,
            )

            self.reset = done



@click.command()
@click.option("--name"        , "name"        , type=str          , default="transport" , help="scenario name"               )
@click.option("--seed"        , "seed"        , type=int          , default=0           , help="random seed"                 )
@click.option("--agents"      , "agents"      , type=int          , default=3           , help="num. agents"                 )
@click.option("--reward-path" , "reward_path" , type=click.Path() , default=None        , help="reward model statedict path" )
@click.option("--value-path"  , "value_path"  , type=click.Path() , default=None        , help="value  model statedict path" )
@click.option("--keyarg"      , "keyargs"     , type=(str,int)    , default=None        , help="additional arguments", multiple=True)
def game(
    name        ,
    seed        ,
    agents      ,
    reward_path ,
    value_path  ,
    keyargs     ,
):

    reward_model = None
    value_model  = None
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
        value_model = models.ValueAFO(
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
        value_model .load_state_dict({k.replace("_orig_mod.",""):v for k,v in checkpoint[ "value_state_dict"].items()})

    Game(
        make_env(
            scenario = name        ,
            num_envs = 1           ,
            n_agents = agents      ,
            device   = "cpu"       ,
            seed     = seed        ,
            wrapper  = Wrapper.GYM ,
            **dict(keyargs)
        ),
        render_name = name          ,
        reward_model = reward_model ,
        value_model  =  value_model ,
    )

if __name__ == "__main__":
    game()
    
    
