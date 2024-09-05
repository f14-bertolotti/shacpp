from Dispersion import Dispersion as DispersionScenario
from unroll import Nunroll, unroll
from evaluate import Nevaluate, evaluate

evaluate

import trainers
import models
import numpy
import torch
import utils
import click
import utils
import vmas
import tqdm
import os


@click.command
@click.option("--device"            , "device"            , type=str          , default="cuda:0" , help="random device")
@click.option("--seed"              , "seed"              , type=int          , default=42       , help="random seed")
@click.option("--episodes"          , "episodes"          , type=int          , default=500      , help="episodes before resetting the environement")
@click.option("--etr"               , "etr"               , type=int          , default=5        , help="training etr between evaluations")
@click.option("--value-batch-size"  , "value_batch_size"  , type=int          , default=512      , help="value model batch size")
@click.option("--reward-batch-size" , "reward_batch_size" , type=int          , default=512      , help="reward model batch size")
@click.option("--value-epochs"      , "value_epochs"      , type=int          , default=4        , help="value model epochs")
@click.option("--reward-epochs"     , "reward_epochs"     , type=int          , default=4        , help="reward model epochs")
@click.option("--etv"               , "etv"               , type=int          , default=10       , help="number of etv")
@click.option("--agents"            , "agents"            , type=int          , default=5        , help="number of agents")
@click.option("--train-envs"        , "train_envs"        , type=int          , default=512      , help="number of train environments")
@click.option("--eval-envs"         , "eval_envs"         , type=int          , default=512      , help="number of evaluation environments")
@click.option("--train-steps"       , "train_steps"       , type=int          , default=32       , help="number of steps for the training rollout")
@click.option("--eval-steps"        , "eval_steps"        , type=int          , default=64       , help="number of steps for the evaluation rollout")
@click.option("--enable-grads"      , "enable_grads"      , type=bool         , default=False    , help="Shold grads be enabled")
@click.option("--gamma"             , "gamma_factor"      , type=float        , default=.99      , help="reward discount factor")
@click.option("--lambda"            , "lambda_factor"     , type=float        , default=.95      , help="td-lambda factor")
@click.option("--dir"               , "dir"               , type=click.Path() , default="./"     , help="directory in which store logs and checkpoints")
def run(
        dir,
        seed,
        episodes,
        agents,
        train_envs,
        train_steps,
        eval_envs,
        eval_steps,
        value_batch_size,
        value_epochs,
        reward_batch_size,
        reward_epochs,
        etr,
        gamma_factor,
        lambda_factor,
        enable_grads,
        etv,
        device
    ):
    torch.set_float32_matmul_precision('high')
    utils.seed_everything(seed)

    eval_logger   = utils.get_file_logger(os.path.join(dir,  "eval.log"))
    reward_logger = utils.get_file_logger(os.path.join(dir,"reward.log"))
    value_logger  = utils.get_file_logger(os.path.join(dir, "value.log"))
    policy_logger = utils.get_file_logger(os.path.join(dir,"policy.log"))
    world_logger  = utils.get_file_logger(os.path.join(dir, "world.log"))

    train_world = vmas.simulator.environment.Environment(
        DispersionScenario(
            device = device,
            radius = .05,
            agents = agents,
        ),
        n_agents           = agents        ,
        num_envs           = train_envs    ,
        device             = device        ,
        shared_reward      = False         ,
        grad_enabled       = enable_grads  ,
        continuous_actions = True          ,
        dict_spaces        = False         ,
        seed               = seed          ,
    )
    
    eval_world = vmas.simulator.environment.Environment(
        DispersionScenario(
            device = device,
            radius = .05,
            agents = agents,
        ),
        n_agents           = agents        ,
        num_envs           = eval_envs     ,
        device             = device        ,
        shared_reward      = False         ,
        grad_enabled       = False         ,
        continuous_actions = True          ,
        dict_spaces        = False         ,
        seed               = seed          ,
    )
    
    reward_model_dataset_size    = 10000
    
    gammas = torch.ones(train_steps, device=device, dtype=torch.float)
    gammas[1:] = gamma_factor
    gammas = gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,train_envs,agents).unsqueeze(-1)
    
    observation_size:int = numpy.prod(train_world.get_observation_space()[0].shape)
    action_size     :int = numpy.prod(train_world.get_action_space()[0].shape)
    
    #value_model =models.Value  (observation_size=observation_size, action_size=action_size, agents=agents, steps=train_steps, layers=1, hidden_size=128 , dropout=0.0, activation="Tanh", device=device)
    policy_model=models.NPolicy(observation_size=observation_size, action_size=action_size, agents=agents, steps=train_steps, layers=1, hidden_size=128, dropout=0.0, activation="Tanh", device=device)
    reward_model=models.Reward2(observation_size=observation_size, action_size=action_size, agents=agents, steps=train_steps, layers=1, hidden_size=128 , feedforward_size=512, heads=4, dropout=0.0, activation="gelu", device=device)
    #world_model =models.World  (observation_size=observation_size, action_size=action_size, agents=agents, steps=train_steps, layers=1, hidden_size=128 , feedforward_size=512, heads=2, dropout=0.0, activation="relu", device=device)

    
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.001) 
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.001)
    #value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=0.001)
    #world_model_optimizer  = torch.optim.Adam( world_model.parameters(), lr=0.001)

    policy_model = torch.compile(policy_model)
    reward_model = torch.compile(reward_model)
    
    rewval_model_cache = {
        "observations" : torch.zeros(reward_model_dataset_size, agents, observation_size, device=device),
        "actions"      : torch.zeros(reward_model_dataset_size, train_steps, agents, action_size, device=device),
        "rewards"      : torch.zeros(reward_model_dataset_size, train_steps, agents,   1, device=device),
        "values"       : torch.zeros(reward_model_dataset_size, train_steps, agents,   1, device=device),
        "mask"         : torch.zeros(reward_model_dataset_size, dtype=torch.bool, device=device),
        "pert_low"     : torch.zeros(train_envs, dtype = torch.float32 , device=device, requires_grad=False),
        "pert_high"    : torch.ones (train_envs, dtype = torch.float32 , device=device, requires_grad=False) * (reward_model_dataset_size-1)
    }
   
    prev_obs, prev_act = None, None
    
    for episode in (bar:=tqdm.tqdm(range(1, episodes))):
        
        # unroll episode #############################################
        episode_data = Nunroll(
            observations  = (None if episode == 1 or episode % etr == 0 else prev_obs), 
            world         = train_world,
            unroll_steps  = train_steps,
            reward_model  = reward_model,
            policy_model  = policy_model.sample,
            world_model   = None, #world_model,
            value_model   = None, #value_model
        )
        
   
        # train actor model ###########################################
        trainers.train_policy(
            episode      = episode               ,
            model        = policy_model          ,
            episode_data = episode_data          ,
            optimizer    = policy_model_optimizer,
            gammas       = gammas                ,
            logger       = policy_logger         ,
        )

 
        # train reward model ##########################################
        trainers.train_rewval(
            episode         = episode                   ,
            model           = reward_model              ,
            optimizer       = reward_model_optimizer    ,
            episode_data    = episode_data              ,
            batch_size      = reward_batch_size         ,
            training_epochs = reward_epochs             ,
            logger          = reward_logger,
            cache           = rewval_model_cache,
            slam = lambda_factor,
            gamma = gamma_factor
        )


    
    
        if episode % etv == 0:
            torch.save({
                "policy_state_dict" : policy_model.state_dict(),   
                "reward_state_dict" : reward_model.state_dict(),
                #"value_state_dict"  : value_model.state_dict(),
                #"world_state_dict"  : world_model.state_dict(),
            }, os.path.join(dir,"models.pkl"))

    
        if episode % etv == 0:
            eval_data = Nevaluate(
                episode      = episode      ,
                policy_model = policy_model  ,
                reward_model = reward_model ,
                value_model  = None, #value_model  ,
                world        = eval_world   ,
                steps        = eval_steps   ,
                envs         = eval_envs    ,
                logger       = eval_logger
            ) 
            eval_reward = eval_data["rewards"]
            bar.set_description(f"reward:{eval_reward:5.3f}")

        prev_obs = episode_data["observations"][-1].detach().clone()
        del episode_data

if __name__ == "__main__":
    run()
