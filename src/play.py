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
    def __init__(self, *args, reward_model = None, value_model = None, policy_model = None, **kwargs):
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
        self.reward_model = reward_model
        self.value_model  =  value_model
        self.policy_model = policy_model
        

        super().__init__(*args, **kwargs)

    def _cycle(self):
        obs = [torch.tensor(o) for o in self.env.reset()]

        # policy lines ############################################################
        self.lineforms_policy = [rendering.Transform() for agent in self.agents]
        self.lines_policy = [rendering.Line(
            (0,0),
            (0,.1),
            width=2,
        ) for agent in self.agents]
        for line in self.lines_policy: line.set_color(0,0,1)
        for line,form in zip(self.lines_policy, self.lineforms_policy): line.add_attr(form)
        for line in self.lines_policy: self.env.unwrapped.viewer.add_geom(line)

        # reward_lines ############################################################
        self.lineforms_reward = [rendering.Transform() for agent in self.agents]
        self.lines_reward = [rendering.Line(
            (0,0),
            (0,.1),
            width=2,
        ) for agent in self.agents]
        for line in self.lines_reward: line.set_color(1,0,0)
        for line,form in zip(self.lines_reward, self.lineforms_reward): line.add_attr(form)
        for line in self.lines_reward: self.env.unwrapped.viewer.add_geom(line)

        # value_lines ############################################################
        self.lineforms_value = [rendering.Transform() for agent in self.agents]
        self.lines_value = [rendering.Line(
            (0,0),
            (0,.1),
            width=2,
        ) for agent in self.agents]
        for line in self.lines_value: line.set_color(0,1,0)
        for line,form in zip(self.lines_value, self.lineforms_value): line.add_attr(form)
        for line in self.lines_value: self.env.unwrapped.viewer.add_geom(line)


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

            for agent, lineform_policy, lineform_reward, lineform_value in zip(self.agents, self.lineforms_policy, self.lineforms_reward, self.lineforms_value):
                lineform_policy.set_translation(*agent.state.pos[0])
                lineform_reward.set_translation(*agent.state.pos[0])
                lineform_value .set_translation(*agent.state.pos[0])

            obs, rew, done, info = self.env.step(action_list)
            act = torch.tensor(action_list).float()
            obs = torch.stack([torch.tensor(o) for o in obs]).float()
            self._write_values(1, "rews: " + str(list(map(lambda x:f"{x:1.3f}", rew))))
            self._write_values(4, "obs1: " + str(list(map(lambda x:f"{x:1.3f}", obs[0,:5].tolist()))))
            self._write_values(3, "obs2: " + str(list(map(lambda x:f"{x:1.3f}", obs[0,5:].tolist()))))

            if self.reward_model is not None:
                obs = obs.clone().detach().requires_grad_(True)
                proxy_rews = self.reward_model(obs, act, None)
                proxy_rews.sum().neg().backward()
                angles = cartesian_to_polar(obs.grad[:,:2])
                for angle,lineform in zip(angles,self.lineforms_reward): lineform.set_rotation(angle[1]-3.14/2 )
                self._write_values(0, "alts: " + str(list(map(lambda x:f"{x:1.3f}", proxy_rews.squeeze(0).tolist()))))

            if self.value_model is not None:
                obs = obs.clone().detach().requires_grad_(True)
                proxy_vals = self.value_model(obs)
                obs.sum().neg().backward()
                angles = cartesian_to_polar(obs.grad[:,:2])
                for angle,lineform in zip(angles,self.lineforms_value): lineform.set_rotation(angle[1]-3.14/2 )
                self._write_values(2, "vals: " + str(list(map(lambda x:f"{x:1.3f}", proxy_vals.squeeze(0).tolist()))))

            if self.policy_model is not None:
                actions = self.policy_model.act(obs)["actions"].squeeze(0)
                angles  = cartesian_to_polar(actions)
                for angle,lineform in zip(angles,self.lineforms_policy): lineform.set_rotation(angle[1]-3.14/2 )

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
@click.option("--policy-path" , "policy_path" , type=click.Path() , default=None        , help="policy model statedict path" )
@click.option("--keyarg"      , "keyargs"     , type=(str,int)    , default=None        , help="additional arguments", multiple=True)
def game(
    name        ,
    seed        ,
    agents      ,
    reward_path ,
    value_path  ,
    policy_path ,
    keyargs     ,
):

    reward_model = None
    value_model  = None
    policy_model = None
    if reward_path is not None:
        reward_checkpoint = torch.load(reward_path, weights_only=True) 
        reward_model = models.rewards.TransformerReward(
             observation_size = 11     ,
             action_size      = 2      ,
             agents           = agents ,
             steps            = 32     ,
             layers           = 1      ,
             hidden_size      = 64     ,
             feedforward_size = 128    ,
             heads            = 1      ,
             dropout          = 0      ,
             activation       = "GELU" ,
             device           = "cpu"
        )
        reward_model.load_state_dict({k.replace("_orig_mod.",""):v for k,v in reward_checkpoint["reward_state_dict"].items()})

    if value_path is not None: 
        value_checkpoint = torch.load(value_path, weights_only=True) 
        value_model = models.values.TransformerValue(
             observation_size = 11     ,
             action_size      = 2      ,
             agents           = agents ,
             steps            = 32     ,
             layers           = 1      ,
             hidden_size      = 64     ,
             feedforward_size = 128    ,
             heads            = 1      ,
             dropout          = 0      ,
             activation       = "GELU" ,
             device           = "cpu"

        )
        value_model .load_state_dict({k.replace("_orig_mod.",""):v for k,v in value_checkpoint[ "value_state_dict"].items()})

    if policy_path is not None:
        policy_checkpoint = torch.load(policy_path, weights_only=True)
        policy_model = models.policies.TransformerPolicy(
            observation_size = 11     ,
            action_size      = 2      ,
            agents           = agents ,
            steps            = 32     ,
            layers           = 1      ,
            hidden_size      = 64     ,
            feedforward_size = 128    ,
            heads            = 1      ,
            dropout          = 0      ,
            activation       = "GELU" ,
            device           = "cpu"
        )
        policy_model.load_state_dict({k.replace("_orig_mod.",""):v for k,v in policy_checkpoint["policy_state_dict"].items()})

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
        policy_model = policy_model ,
    )

if __name__ == "__main__":
    game()
    
    
