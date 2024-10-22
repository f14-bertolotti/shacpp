from unroll import unroll
from evaluate import evaluate

import trainers
import models
import torch
import utils
import vmas
import tqdm
import os

def ppo(
        dir                    : str                                    ,
        episodes               : int                                    ,
        observation_size       : int                                    ,
        action_size            : int                                    ,
        agents                 : int                                    ,
        train_envs             : int                                    ,
        train_steps            : int                                    ,
        eval_envs              : int                                    ,
        eval_steps             : int                                    ,
        policy_model           : models.Model                           ,
        value_model            : models.Model                           ,
        policy_model_optimizer : torch.optim.Optimizer                  ,
        value_model_optimizer  : torch.optim.Optimizer                  ,
        train_world            : vmas.simulator.environment.Environment ,
        eval_world             : vmas.simulator.environment.Environment ,
        batch_size             : int                                    ,
        epochs                 : int                                    ,
        gamma_factor           : float                                  ,
        etr                    : int                                    ,
        etv                    : int                                    ,
        compile                : bool                                   ,
        restore_path           : str                                    ,
        early_stopping         : dict                                   ,
    ):

    ppo_logger    = utils.get_file_logger(os.path.join(dir,"ppo.log"))
    eval_logger   = utils.get_file_logger(os.path.join(dir,"eval.log"))

    if compile:
        policy_model = torch.compile(policy_model)
        value_model  = torch.compile(value_model)

    checkpoint = dict()
    if restore_path:
        checkpoint = torch.load(restore_path)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        value_model .load_state_dict(checkpoint["value_state_dict"])
    
    prev_observations : torch.Tensor = torch.empty(train_envs, agents, observation_size)
    prev_dones        : torch.Tensor = torch.empty(train_envs, agents, 1)
    eval_reward       : float        = 0
    best_reward       : float        = checkpoint.get("best_reward", float("-inf"))
    patience          : int          = 0
    max_reward        : float        = float("-inf")
    for episode in (bar:=tqdm.tqdm(range(checkpoint.get("episode", 0)+1, episodes))):
        
        # unroll episode #############################################
        episode_data = unroll(
            observations  = (None if episode == 1 or episode % etr == 0 else prev_observations), 
            dones         = (None if episode == 1 or episode % etr == 0 else prev_dones),
            world         = train_world,
            unroll_steps  = train_steps,
            policy_model  = policy_model.sample,
        )

        # train policy and value models ##############################
        trainers.ppo_policy_value(
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

        # checkpoint ##################################################
        if episode % etv == 0:
            torch.save({
                "policy_state_dict" : policy_model.state_dict(),   
                "value_state_dict"  : value_model .state_dict(),
                "best_reward"       : best_reward,
                "episode"           : episode,
            }, os.path.join(dir,"models.pkl"))

        # evaluation #################################################
        if episode % etv == 0:
            eval_data = evaluate(
                episode      = episode      ,
                policy_model = policy_model ,
                world        = eval_world   ,
                steps        = eval_steps   ,
                envs         = eval_envs    ,
                logger       = eval_logger
            ) 
            eval_reward = eval_data["rewards"]
            max_reward  = eval_data["max_reward"]

            # save best model ##########################################
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save({
                    "policy_state_dict" : policy_model.state_dict(),   
                    "value_state_dict"  : value_model .state_dict(),
                    "best_reward"       : best_reward,
                    "episode"           : episode,
                }, os.path.join(dir,"best.pkl"))

            # early_stopping #########################################
            patience = patience + 1 if eval_reward >= max_reward * early_stopping["max_reward_fraction"] else 0
            if patience >= early_stopping["patience"]: break

            del eval_data

        # update progress bar ########################################
        done_train_envs = episode_data["last_dones"][:,0].sum().int().item()
        bar.set_description(f"reward:{eval_reward:5.3f}, max:{max_reward:5.3f}, dones:{done_train_envs:3d}, episode:{episode:5d}")
        # set up next iteration ######################################
        prev_observations = episode_data["last_observations"].detach()
        prev_dones        = episode_data["last_dones"       ].detach()

        # clean up ###################################################
        del episode_data

    torch.save({
        "policy_state_dict" : policy_model.state_dict(),   
        "value_state_dict"  : value_model .state_dict(),
        "episode"           : episodes,
        "best_reward"       : best_reward
    }, os.path.join(dir,"last.pkl"))


