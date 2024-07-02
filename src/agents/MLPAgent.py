from agents import agent
import utils, click, math, torch, copy

import torch.nn as nn
from utils import layer_init
import numpy as np

from torch.distributions.normal import Normal

class MLPAgent(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size, device):
        super().__init__()
        agents = 3
        self.critic = nn.Sequential(
            utils.Lambda(lambda x:x.flatten(1,2)),
            layer_init(nn.Linear(observation_size * agents, hidden_size, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, agents, device=device), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            utils.Lambda(lambda x:x.flatten(1,2)),
            layer_init(nn.Linear(observation_size * agents, hidden_size, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_size * agents, device=device), std=0.01),
            nn.Tanh(),
            utils.Lambda(lambda x:x.view(x.size(0),agents,action_size)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size)).to(device)

        print(self)

    def get_value(self, x):
        flag = type(x) == list
        x = torch.stack(x).transpose(0,1) if type(x) == list else x
        return {"values" : self.critic(x).transpose(0,1) if flag else self.critic(x)}

    def get_action(self, x, action=None):
        flag = type(x) == list
        x = torch.stack(x).transpose(0,1) if flag else x
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = torch.clamp(probs.rsample(),-1,1)
        
        return {
            "logits"   : action_mean,
            "actions"  : action.transpose(0,1) if flag else action,
            "logprobs" : probs.log_prob(action).sum(-1).transpose(0,1) if flag else probs.log_prob(action).sum(-1),
            "entropy"  : probs.entropy().sum(-1).transpose(0,1) if flag else probs.entropy().sum(-1)
        }

    def get_action_and_value(self, observation, action=None):
        result = self.get_action(observation, action=action) | self.get_value(observation)
        return result["actions"], result["logprobs"], result["entropy"], result["values"]



@agent.group(invoke_without_command=True)
@click.option("--observation-size" , "observation_size" , type=int , default=2)
@click.option("--action-size"      , "action_size"      , type=int , default=2)
@click.option("--hidden-size"      , "hidden_size"      , type=int , default=64)
@click.option("--device"           , "device"           , type=str , default="cuda:0")
@click.option("--state-dict-path"  , "state_dict_path"  , type=click.Path(), default=None)
@click.pass_obj
def mlp_agent(trainer, observation_size, action_size, hidden_size, device, state_dict_path):
    trainer.set_agent(
        MLPAgent(
            observation_size = observation_size,
            action_size      = action_size,
            hidden_size      = hidden_size,
            device           = device
        )
    )

    if state_dict_path: trainer.agent.load_state_dict(torch.load(state_dict_path)["agentsd"])
