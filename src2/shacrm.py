from Dispersion import Dispersion as DispersionScenario
from unroll import unroll
from evaluate import evaluate

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
@utils.common_options
@click.option("--value-batch-size"  , "value_batch_size"  , type=int  , default=512, help="value model batch size")
@click.option("--reward-batch-size" , "reward_batch_size" , type=int  , default=512, help="reward model batch size")
@click.option("--value-epochs"      , "value_epochs"      , type=int  , default=4  , help="value model epochs")
@click.option("--reward-epochs"     , "reward_epochs"     , type=int  , default=4  , help="reward model epochs")
@click.option("--gamma"             , "gamma_factor"      , type=float, default=.99, help="reward discount factor")
@click.option("--lambda"            , "lambda_factor"     , type=float, default=.95, help="td-lambda factor")
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
        observation_size,
        action_size,
        etr,
        gamma_factor,
        lambda_factor,
        etv,
        compile,
        restore_path,
        device
    ):

    utils.save_locals(dir, locals())
    utils.seed_everything(seed)

    eval_logger   = utils.get_file_logger(os.path.join(dir,  "eval.log"))
    reward_logger = utils.get_file_logger(os.path.join(dir,"reward.log"))
    value_logger  = utils.get_file_logger(os.path.join(dir, "value.log"))
    policy_logger = utils.get_file_logger(os.path.join(dir,"policy.log"))

    train_world = vmas.simulator.environment.Environment(
        DispersionScenario(
            device = device,
            radius = .05,
            agents = agents,
        ),
        n_agents           = agents        ,
        num_envs           = train_envs    ,
        device             = device        ,
        grad_enabled       = True          ,
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
        grad_enabled       = False         ,
        continuous_actions = True          ,
        dict_spaces        = False         ,
        seed               = seed          ,
    )
    
    reward_model_dataset_size    = 40000
    
    gammas = torch.ones(train_steps, device=device, dtype=torch.float)
    gammas[1:] = gamma_factor
    gammas = gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,train_envs,agents)
    
    value_model  = models.Value (observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device)
    policy_model = models.Policy(observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device, shared=[True, True, False])
    reward_model = models.Reward(observation_size = observation_size, action_size = action_size, agents = agents, layers = 3, hidden_size = 512, dropout=0.0, activation="Tanh", device = device)

    if compile:
        policy_model = torch.compile(policy_model)
        reward_model = torch.compile(reward_model)
        value_model  = torch.compile(value_model)

    if restore_path:
        checkpoint = torch.load(restore_path)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        reward_model.load_state_dict(checkpoint["reward_state_dict"])
        value_model .load_state_dict(checkpoint["value_state_dict"])
    
    
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.0001) 
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.001)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=0.001)
    
    reward_model_cache = {
        "observations" : torch.zeros(reward_model_dataset_size, agents, observation_size, device=device),
        "actions"      : torch.zeros(reward_model_dataset_size, agents,      action_size, device=device),
        "rewards"      : torch.zeros(reward_model_dataset_size, agents, device=device),
        "mask"         : torch.zeros(reward_model_dataset_size, dtype=torch.bool, device=device),
        "pert_low"     : torch.zeros(train_envs * train_steps, dtype = torch.float32 , device=device, requires_grad=False),
        "pert_high"    : torch.ones (train_envs * train_steps, dtype = torch.float32 , device=device, requires_grad=False) * (reward_model_dataset_size-1)
    }
   
    prev_observations, prev_dones = None, None
    
    for episode in (bar:=tqdm.tqdm(range(1, episodes))):
        
        # unroll episode #############################################
        episode_data = unroll(
            observations  = (None if episode == 1 or episode % etr == 0 else prev_observations), 
            dones         = (None if episode == 1 or episode % etr == 0 else prev_dones),
            world         = train_world,
            unroll_steps  = train_steps,
            policy_model  = policy_model.sample,
        )
        episode_data["proxy_rewards"] = reward_model(episode_data["observations"], episode_data["actions"])
        episode_data["values"]        = value_model(episode_data["observations"].flatten(0,1)).view(episode_data["rewards"].shape)
    
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
        trainers.train_reward(
            episode         = episode                   ,
            model           = reward_model              ,
            optimizer       = reward_model_optimizer    ,
            episode_data    = episode_data              ,
            cached_data     = reward_model_cache        ,
            batch_size      = reward_batch_size         ,
            dataset_size    = reward_model_dataset_size ,
            training_epochs = reward_epochs             ,
            logger          = reward_logger
        )

        # train value model ###########################################
        trainers.train_value(
            episode         = episode               ,
            model           = value_model           ,
            optimizer       = value_model_optimizer ,
            episode_data    = episode_data          ,
            training_epochs = value_epochs          ,
            batch_size      = value_batch_size      ,
            slam            = lambda_factor         ,
            gamma           = gamma_factor          ,
            logger          = value_logger
        )
    
    
        if episode % etv == 0:
            torch.save({
                "policy_state_dict" : policy_model.state_dict(),   
                "reward_state_dict" : reward_model.state_dict(),
                "value_state_dict"  : value_model.state_dict(),
            }, os.path.join(dir,"models.pkl"))

    
        if episode % etv == 0:
            eval_data = evaluate(
                episode      = episode      ,
                policy_model = policy_model ,
                reward_model = reward_model ,
                world        = eval_world   ,
                steps        = eval_steps   ,
                envs         = eval_envs    ,
                logger       = eval_logger
            ) 
            eval_reward = eval_data["rewards"]
            bar.set_description(f"reward:{eval_reward:5.3f}")
            del eval_data

        prev_observations = episode_data["last_observations"].detach().clone()
        prev_dones        = episode_data["last_dones"       ].detach().clone()
        del episode_data

if __name__ == "__main__":
    run()
