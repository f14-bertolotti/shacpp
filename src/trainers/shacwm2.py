from evaluate import evaluate
from unroll import unroll

import trainers
import models
import torch
import utils
import tqdm
import vmas
import os

def shacwm2(
        dir                    : str                                    ,
        episodes               : int                                    ,
        observation_size       : int                                    ,
        action_size            : int                                    ,
        agents                 : int                                    ,
        train_envs             : int                                    ,
        train_steps            : int                                    ,
        eval_steps             : int                                    ,
        eval_envs              : int                                    ,
        policy_model           : models.Model                           ,
        world_model            : models.Model                           ,
        reward_model           : models.Model                           ,
        value_model            : models.Model                           ,
        world_model_optimizer  : torch.optim.Optimizer                  ,
        policy_model_optimizer : torch.optim.Optimizer                  ,
        reward_model_optimizer : torch.optim.Optimizer                  ,
        value_model_optimizer  : torch.optim.Optimizer                  ,
        train_world            : vmas.simulator.environment.Environment ,
        eval_world             : vmas.simulator.environment.Environment ,
        cache_size             : int                                    ,
        reward_bins            : int                                    ,
        world_batch_size       : int                                    ,
        reward_batch_size      : int                                    ,
        value_batch_size       : int                                    ,
        world_epochs           : int                                    ,
        reward_epochs          : int                                    ,
        value_epochs           : int                                    ,
        gamma_factor           : float                                  ,
        lambda_factor          : float                                  ,
        etr                    : int                                    ,
        etv                    : int                                    ,
        compile                : bool                                   ,
        restore_path           : str                                    ,
        device                 : str                                    ,
        early_stopping         : dict                                   ,
    ):

    eval_logger   = utils.get_file_logger(os.path.join(dir,  "eval.log"))
    world_logger  = utils.get_file_logger(os.path.join(dir, "world.log"))
    policy_logger = utils.get_file_logger(os.path.join(dir,"policy.log"))
    reward_logger = utils.get_file_logger(os.path.join(dir,"reward.log"))
    value_logger  = utils.get_file_logger(os.path.join(dir, "value.log"))

    gammas = utils.gamma_tensor(train_steps, train_envs, agents, gamma_factor).to(device)

    world_cache = {
        "observations" : torch.zeros(cache_size, train_steps, agents, observation_size, device=device),
        "actions"      : torch.zeros(cache_size, train_steps, agents, action_size, device=device),
        "last_obs"     : torch.zeros(cache_size, agents, observation_size, device=device),
        "mask"         : torch.zeros(cache_size, dtype=torch.bool, device=device),
    }

    reward_cache = {
        "observations" : torch.zeros(cache_size, agents, observation_size, device=device),
        "actions"      : torch.zeros(cache_size, agents,      action_size, device=device),
        "rewards"      : torch.zeros(cache_size, agents, device=device),
        "mask"         : torch.zeros(cache_size, dtype=torch.bool, device=device),
    }

    if compile:
        policy_model = torch.compile(policy_model)
        value_model  = torch.compile(value_model)
        reward_model = torch.compile(reward_model)
        world_model  = torch.compile(world_model)

    checkpoint = dict()
    if restore_path:
        checkpoint = torch.load(restore_path, weights_only=False)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        world_model .load_state_dict(checkpoint[ "world_state_dict"])
        reward_model.load_state_dict(checkpoint["reward_state_dict"])
        value_model .load_state_dict(checkpoint[ "value_state_dict"])

    prev_observations : torch.Tensor = torch.empty(train_envs, agents, observation_size)
    prev_dones        : torch.Tensor = torch.empty(train_envs, agents, 1)
    eval_reward       : float        = 0
    best_reward       : float        = checkpoint.get("best_reward", float("-inf"))
    max_reward        : float        = torch.tensor([float("-inf")])
    for episode in (bar:=tqdm.tqdm(range(checkpoint.get("episode", 0)+1, episodes))):
        
        # unroll episode #############################################
        episode_data = unroll(
            observations  = (None if episode == 1 or episode % etr == 0 or prev_dones[:,0].all() else prev_observations), 
            dones         = (None if episode == 1 or episode % etr == 0 or prev_dones[:,0].all() else prev_dones),
            world         = train_world,
            unroll_steps  = train_steps,
            policy_model  = policy_model.sample,
        )

        # compute rewards and values #################################
        obs = world_model(episode_data["observations"][0].unsqueeze(1), episode_data["actions"].transpose(0,1))[2].transpose(0,1)[1:]
        episode_data["proxy_rewards"] = reward_model(obs, episode_data["actions"])
        episode_data["values"]        = value_model (obs)
    
        # train actor model ##########################################
        trainers.train_policy(
            episode      = episode               ,
            policy_model = policy_model          ,
            episode_data = episode_data          ,
            optimizer    = policy_model_optimizer,
            gammas       = gammas                ,
            logger       = policy_logger         ,
        )
         
        # train world model ##########################################
        trainers.train_world2(
            episode         = episode               ,
            model           = world_model           ,
            optimizer       = world_model_optimizer ,
            episode_data    = episode_data          ,
            cached_data     = world_cache           ,
            batch_size      = world_batch_size      ,
            cache_size      = cache_size            ,
            bins            = reward_bins           ,
            training_epochs = world_epochs          ,
            slam            = lambda_factor         ,
            gamma           = gamma_factor          ,
            logger          = world_logger
        )

        # train reward model ##########################################
        trainers.train_reward(
            episode         = episode                   ,
            model           = reward_model              ,
            optimizer       = reward_model_optimizer    ,
            episode_data    = episode_data              ,
            cached_data     = reward_cache              ,
            batch_size      = reward_batch_size         ,
            cache_size      = cache_size                ,
            training_epochs = reward_epochs             ,
            bins            = reward_bins               ,
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
 

        # checkpoint ##################################################
        if episode % etv == 0:
            torch.save({
                "policy_state_dict" : policy_model.state_dict(),
                "world_state_dict"  :  world_model.state_dict(),
                "best_reward"       : best_reward,
                "episode"           : episode,
            }, os.path.join(dir,"models.pkl"))

        # evaluation #################################################
        if episode % etv == 0:
            eval_data = evaluate(
                episode      = episode      ,
                policy_model = policy_model ,
                reward_model = reward_model  ,
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
                    "world_state_dict"  :  world_model.state_dict(),
                    "best_reward"       : best_reward,
                    "episode"           : episode,
                }, os.path.join(dir,"best.pkl"))

            # early_stopping #########################################
            if (eval_reward >= (max_reward * early_stopping["max_reward_fraction"])).all(): break

            del eval_data

        # update progress bar ########################################
        done_train_envs = episode_data["last_dones"][:,0].sum().int().item()
        bar.set_description(f"reward:{eval_reward:5.3f}, max:{max_reward.mean():5.3f}, dones:{done_train_envs:3d}, episode:{episode:5d}")
        # set up next iteration ######################################
        prev_observations = episode_data["last_observations"].detach()
        prev_dones        = episode_data["last_dones"       ].detach()

        # clean up ###################################################
        del episode_data

    torch.save({
        "policy_state_dict" : policy_model.state_dict(),
        "world_state_dict"  :  world_model.state_dict(),
        "episode"           : episodes,
        "best_reward"       : best_reward
    }, os.path.join(dir,"last.pkl"))


