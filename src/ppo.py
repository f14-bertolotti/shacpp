from Dispersion import Dispersion as DispersionScenario
from unroll import unroll
from evaluate import evaluate

import trainers
import models
import torch
import utils
import click
import utils
import vmas
import tqdm
import os


@click.command
@utils.common_options
@click.option("--gamma"      , "gamma_factor" , type=float , default=.99 , help="reward discount factor")
@click.option("--batch-size" , "batch_size"   , type=int   , default=512 , help="batch size")
@click.option("--epochs"     , "epochs"       , type=int   , default=4   , help="epochs")
def run(
        dir,
        restore_path,
        seed,
        episodes,
        agents,
        observation_size,
        action_size,
        train_envs,
        train_steps,
        eval_envs,
        eval_steps,
        etr,
        etv,
        device,
        gamma_factor,
        batch_size,
        epochs,
        compile,
    ):
    torch.set_float32_matmul_precision('high')
    utils.seed_everything(seed)

    ppo_logger    = utils.get_file_logger(os.path.join(dir,"ppo.log"))
    eval_logger   = utils.get_file_logger(os.path.join(dir,"eval.log"))

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
        shared_reward      = False         ,
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
        shared_reward      = False         ,
        seed               = seed          ,
    )
    
    policy_model = models.Policy(observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", shared=[False, False, False],  device = device)
    value_model  = models.Value (observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device)

    if compile:
        policy_model = torch.compile(policy_model)
        value_model  = torch.compile(value_model)

    if restore_path:
        checkpoint = torch.load(restore_path)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        value_model.load_state_dict(checkpoint["value_state_dict"])
    
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.001)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=0.001)

    
    prev_dones, prev_observations = None, None
    
    for episode in (bar:=tqdm.tqdm(range(1, episodes))):
        
        # unroll episode #############################################
        episode_data = unroll(
            observations  = (None if episode == 1 or episode % etr == 0 else prev_observations), 
            dones         = (None if episode == 1 or episode % etr == 0 else prev_dones),
            world         = train_world,
            unroll_steps  = train_steps,
            policy_model  = policy_model.sample,
        )


        trainers.train_ppo(
            policy_model     = policy_model,
            value_model      = value_model,
            policy_optimizer = policy_model_optimizer,
            value_optimizer  = value_model_optimizer,
            episode_data     = episode_data,
            batch_size       = batch_size,
            ppo_logger       = ppo_logger,
            gamma            = gamma_factor,
            epochs           = epochs
        )

        if episode % etv == 0:
            torch.save({
                "policy_state_dict" : policy_model.state_dict(),   
                "value_state_dict"  : value_model.state_dict(),
            }, os.path.join(dir,"models.pkl"))

        if episode % etv == 0:
            policy_model.eval()
            eval_data = evaluate(
                episode      = episode      ,
                policy_model = policy_model ,
                world        = eval_world   ,
                steps        = eval_steps   ,
                envs         = eval_envs    ,
                logger       = eval_logger
            ) 
            policy_model.train()
            eval_reward = eval_data["rewards"]
            bar.set_description(f"reward:{eval_reward:5.3f}")
            del eval_data

        prev_observations = episode_data["last_observations"].detach().clone()
        prev_dones        = episode_data["last_dones"       ].detach().clone()
        del episode_data

if __name__ == "__main__":
    run()
