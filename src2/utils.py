import logging
import random
import numpy
import torch
import click
import json

def layer_init(layer, std=1.141, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None: torch.nn.init.constant_(layer.bias, bias_const)
    return layer

@torch.no_grad()
def pert(low:torch.Tensor, peak:torch.Tensor, high:torch.Tensor, lamb:int=8):
    """ pert distribution   : https://en.wikipedia.org/wiki/PERT_distribution 
        implementation from : https://stackoverflow.com/questions/68476485/random-values-from-a-pert-distribution-in-python """
    r = high - low
    alpha = 1 + lamb * (peak - low) / r
    beta  = 1 + lamb * (high - peak) / r
    return low + torch.distributions.Beta(alpha, beta).sample() * r

@torch.no_grad()
def compute_values(values, rewards, dones, slam, gamma):
    steps, envirs, agents = values.shape

    target_values = torch.zeros(steps, envirs, agents, dtype=torch.float32, device=values.device)
    Ai = torch.zeros(envirs, agents, dtype=torch.float32, device=values.device)
    Bi = torch.zeros(envirs, agents, dtype=torch.float32, device=values.device)
    lam = torch.ones(envirs, agents, dtype=torch.float32, device=values.device)

    for i in reversed(range(steps)):
        lam = slam * lam * (1. - dones[i]) + dones[i]
        Ai = (1.0 - dones[i]) * (slam * gamma * Ai + gamma * values[i] + (1. - lam) / (1. - slam) * rewards[i])
        Bi = gamma * (values[i] * dones[i] + Bi * (1.0 - dones[i])) + rewards[i]
        target_values[i] = (1.0 - slam) * Ai + lam * Bi

    return target_values

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): 
        return self.func(x)

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def get_file_logger(path):
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(json.dumps(json.loads("""
        {
            "data"    : "%(asctime)s", 
            "level"   : "%(levelname)s", 
            "process" : { 
                "id"   : "%(process)d", 
                "name" : "%(processName)s"
            }, 
            "thread"  : {
                "id"   : "%(thread)d", 
                "name" : "%(threadName)s"
            }, 
            "message" : "MESSAGE" 
        }
    """), indent=None, separators=(",",":")).replace("\"MESSAGE\"","%(message)s"))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def chain(*decs):
    def deco(f):
        for dec in reversed(decs): f = dec(f)
        return f
    return deco

common_options = chain(
    click.option("--device"            , "device"            , type=str          , default="cuda:0" , help="random device"                                 ),
    click.option("--seed"              , "seed"              , type=int          , default=42       , help="random seed"                                   ),
    click.option("--episodes"          , "episodes"          , type=int          , default=500      , help="episodes before resetting the environement"    ),
    click.option("--observation-size"  , "observation_size"  , type=int          , default=2        , help="observation size"                              ),
    click.option("--action-size"       , "action_size"       , type=int          , default=11       , help="action size"                                   ),
    click.option("--agents"            , "agents"            , type=int          , default=5        , help="number of agents"                              ),
    click.option("--train-envs"        , "train_envs"        , type=int          , default=512      , help="number of train environments"                  ),
    click.option("--eval-envs"         , "eval_envs"         , type=int          , default=512      , help="number of evaluation environments"             ),
    click.option("--train-steps"       , "train_steps"       , type=int          , default=32       , help="number of steps for the training rollout"      ),
    click.option("--eval-steps"        , "eval_steps"        , type=int          , default=64       , help="number of steps for the evaluation rollout"    ),
    click.option("--dir"               , "dir"               , type=click.Path() , default="./"     , help="directory in which store logs and checkpoints" ),
    click.option("--etr"               , "etr"               , type=int          , default=5        , help="epochs between environment resets"             ),
    click.option("--etv"               , "etv"               , type=int          , default=10       , help="epochs between evaluations"                    ),
)


@torch.no_grad
def compute_advantages(value_model, rewards, next_obs, values, dones, next_done, gamma=.99, gaelambda=.95):

    next_value = value_model(next_obs)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    for t in reversed(range(rewards.size(0))):
        if t == rewards.size(0) - 1:
            nextnonterminal = 1.0 - next_done.float()
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].float()
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gaelambda * nextnonterminal * lastgaelam

    return advantages

@torch.no_grad()
def compute_returns(advantages, values):
    return advantages + values


def value_loss(newvalue, oldvalues, returns, clipcoef=None):
    v_loss_unclipped = (newvalue - returns) ** 2
    if clipcoef is not None:
        v_clipped = oldvalues + torch.clamp(newvalue - oldvalues, -clipcoef, clipcoef)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        return 0.5 * v_loss_max.mean()
    return 0.5 * v_loss_unclipped.mean()

def policy_loss(advantages, ratio, clipcoef):
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clipcoef, 1 + clipcoef)
    return torch.max(pg_loss1, pg_loss2).mean()

def ppo_loss(new_values, old_values, new_logprobs, old_logprobs, advantages, returns, entropy, vclip, clipcoef, vfcoef, entcoef, normadv = True):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) if normadv else advantages
    ratio = (new_logprobs - old_logprobs).exp()

    ploss = policy_loss(advantages, ratio, clipcoef)
    vloss = value_loss(new_values, old_values, returns, clipcoef if vclip else None)
    eloss = entropy.mean()

    return ploss + vloss * vfcoef - entcoef * eloss 

