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
@click.option("--world-batch-size"  , "world_batch_size"  , type=int  , default=2048, help="world model batch size")
@click.option("--world-epochs"      , "world_epochs"      , type=int  , default=4   , help="world model epochs")
@click.option("--gamma"             , "gamma_factor"      , type=float, default=.99 , help="reward discount factor")
@click.option("--lambda"            , "lambda_factor"     , type=float, default=.95 , help="td-lambda factor")
def run(
        dir,
        seed,
        episodes,
        agents,
        train_envs,
        train_steps,
        eval_envs,
        eval_steps,
        world_batch_size,
        world_epochs,
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
    torch.set_float32_matmul_precision('high')

    utils.save_locals(dir, locals())
    utils.seed_everything(seed)

    eval_logger   = utils.get_file_logger(os.path.join(dir,  "eval.log"))
    world_logger  = utils.get_file_logger(os.path.join(dir, "world.log"))
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
        grad_enabled       = False         ,
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
    
    world_model_dataset_size = 10000
    
    gammas = torch.ones(train_steps, device=device, dtype=torch.float)
    gammas[1:] = gamma_factor
    gammas = gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,train_envs,agents)
    
    policy_model = models.PolicyAFO(observation_size = observation_size, action_size = action_size, agents = agents, steps=train_steps, layers = 1, hidden_size = 2048, dropout=0.0, activation="Tanh", device = device)
    world_model = models.worlds.TransformerWorld(observation_size = observation_size, action_size = action_size, agents = agents, steps=train_steps, layers=3, hidden_size=32, heads=1, feedforward_size=256, dropout=0.0, activation="ReLU", device=device)

    if compile:
        policy_model = torch.compile(policy_model)
        world_model  = torch.compile(world_model)

    if restore_path:
        checkpoint = torch.load(restore_path)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        world_model .load_state_dict(checkpoint[ "world_state_dict"])
    
    world_model_optimizer  = torch.optim.Adam( world_model.parameters(), lr=0.001) 
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.001)
    
    world_model_cache = {
        "observations" : torch.zeros(world_model_dataset_size, train_steps, agents, observation_size, device=device),
        "last_obs"     : torch.zeros(world_model_dataset_size, agents, observation_size, device=device),
        "actions"      : torch.zeros(world_model_dataset_size, train_steps, agents, action_size, device=device),
        "rewards"      : torch.zeros(world_model_dataset_size, train_steps, agents, device=device),
        "values"       : torch.zeros(world_model_dataset_size, train_steps, agents, device=device),
        "mask"         : torch.zeros(world_model_dataset_size, dtype=torch.bool, device=device),
        "pert_low"     : torch.zeros(train_envs, dtype = torch.float32 , device=device, requires_grad=False),
        "pert_high"    : torch.ones (train_envs, dtype = torch.float32 , device=device, requires_grad=False) * (world_model_dataset_size-1)
    }
   
    prev_observations, prev_dones, eval_reward = None, None, 0
    
    for episode in (bar:=tqdm.tqdm(range(1, episodes))):
        
        # unroll episode #############################################
        episode_data = unroll(
            observations  = (None if episode == 1 or episode % etr == 0 or prev_dones[:,0].all() else prev_observations), 
            dones         = (None if episode == 1 or episode % etr == 0 or prev_dones[:,0].all() else prev_dones),
            world         = train_world,
            unroll_steps  = train_steps,
            policy_model  = policy_model.sample,
        )

        results = world_model(episode_data["observations"][0].unsqueeze(1), episode_data["actions"].transpose(0,1))
        episode_data["proxy_rewards"], episode_data["values"] = results[0].transpose(0,1), results[1].transpose(0,1)
    
        # train actor model ###########################################
        trainers.train_policy(
            episode      = episode               ,
            policy_model = policy_model          ,
            episode_data = episode_data          ,
            optimizer    = policy_model_optimizer,
            gammas       = gammas                ,
            logger       = policy_logger         ,
        )
         
        # train world model ##########################################
        trainers.train_world(
            episode         = episode                   ,
            model           = world_model              ,
            optimizer       = world_model_optimizer    ,
            episode_data    = episode_data              ,
            cached_data     = world_model_cache        ,
            batch_size      = world_batch_size         ,
            training_epochs = world_epochs             ,
            logger          = world_logger
        )

        ## train value model ###########################################
        #trainers.train_value(
        #    episode         = episode               ,
        #    model           = value_model           ,
        #    optimizer       = value_model_optimizer ,
        #    episode_data    = episode_data          ,
        #    training_epochs = value_epochs          ,
        #    batch_size      = value_batch_size      ,
        #    slam            = lambda_factor         ,
        #    gamma           = gamma_factor          ,
        #    logger          = value_logger
        #)
    
    
        if episode % etv == 0:
            torch.save({
                "policy_state_dict" : policy_model.state_dict(),   
                "world_state_dict"  : world_model .state_dict(),
            }, os.path.join(dir,"models.pkl"))

    
        if episode % etv == 0:
            eval_data = evaluate(
                episode      = episode      ,
                policy_model = policy_model ,
                world_model  = world_model  ,
                world        = eval_world   ,
                steps        = eval_steps   ,
                envs         = eval_envs    ,
                logger       = eval_logger
            ) 
            eval_reward = eval_data["rewards"]
            del eval_data
        
        done_train_envs = episode_data["last_dones"][:,0].sum().int().item()
        bar.set_description(f"reward:{eval_reward:5.3f}, dones:{done_train_envs:3d}")

        prev_observations = episode_data["last_observations"].detach()
        prev_dones        = episode_data["last_dones"       ].detach()
        del episode_data

if __name__ == "__main__":
    run()
