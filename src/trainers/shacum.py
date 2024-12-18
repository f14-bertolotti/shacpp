from evaluate import evaluate
from unroll import unroll

import trainers
import models
import torch
import utils
import tqdm
import vmas
import os

def shacum(
        dir                     : str                                    ,
        episodes                : int                                    ,
        observation_size        : int                                    ,
        action_size             : int                                    ,
        agents                  : int                                    ,
        train_envs              : int                                    ,
        train_steps             : int                                    ,
        eval_steps              : int                                    ,
        eval_envs               : int                                    ,
        policy_model            : models.Model                           ,
        world_model             : models.Model                           ,
        world_model_optimizer   : torch.optim.Optimizer                  ,
        policy_model_optimizer  : torch.optim.Optimizer                  ,
        train_world             : vmas.simulator.environment.Environment ,
        eval_world              : vmas.simulator.environment.Environment ,
        world_cache_size        : int                                    ,
        world_bins              : int                                    ,
        world_batch_size        : int                                    ,
        world_epochs            : int                                    ,
        gamma_factor            : float                                  ,
        lambda_factor           : float                                  ,
        etr                     : int                                    ,
        etv                     : int                                    ,
        compile                 : bool                                   ,
        restore_path            : str|None                               ,
        device                  : str                                    ,
        early_stopping          : dict                                   ,
        world_clip_coefficient  : float|None                             ,
        policy_clip_coefficient : float|None                             ,
        out_coefficient         : float                                  ,
        world_tolerance         : float                                  ,
        world_stop_threshold    : float                                  ,
        world_ett               : int                                    ,
    ):

    eval_logger   = utils.get_file_logger(os.path.join(dir,  "eval.log"))
    world_logger  = utils.get_file_logger(os.path.join(dir, "world.log"))
    policy_logger = utils.get_file_logger(os.path.join(dir,"policy.log"))

    gammas = utils.gamma_tensor(train_steps, train_envs, agents, gamma_factor).to(device)

    universe_cache = {
        "observations" : torch.zeros(world_cache_size, train_steps+1, agents, observation_size, device=device),
        "actions"      : torch.zeros(world_cache_size, train_steps  , agents, action_size     , device=device),
        "rewards"      : torch.zeros(world_cache_size, train_steps  , agents, device=device),
        "values"       : torch.zeros(world_cache_size, train_steps  , agents, device=device),
        "mask"         : torch.zeros(world_cache_size, dtype=torch.bool, device=device),
    } if world_cache_size is not None else None

    if compile:
        policy_model = torch.compile(policy_model)
        world_model  = torch.compile(world_model)

    checkpoint = dict()
    if restore_path is not None:
        checkpoint = torch.load(restore_path, weights_only=False)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        world_model .load_state_dict(checkpoint[ "world_state_dict"])

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
        result = world_model(episode_data["observations"].transpose(0,1), episode_data["actions"].transpose(0,1))
        episode_data["proxy_rewards"] = result["rewards"][:,1:].transpose(0,1)
        episode_data["values"]        = result["values" ][:,1:].transpose(0,1)

        # train actor model ##########################################
        trainers.routines.train_policy(
            episode          = episode                 ,
            policy_model     = policy_model            ,
            episode_data     = episode_data            ,
            optimizer        = policy_model_optimizer  ,
            gammas           = gammas                  ,
            logger           = policy_logger           ,
            clip_coefficient = policy_clip_coefficient ,
            out_coefficient  = out_coefficient
        )
         
        # train world model ##########################################
        trainers.routines.train_universe(
            episode          = episode                ,
            model            = world_model            ,
            optimizer        = world_model_optimizer  ,
            episode_data     = episode_data           ,
            cached_data      = universe_cache         ,
            batch_size       = world_batch_size       ,
            cache_size       = world_cache_size       ,
            bins             = world_bins             ,
            training_epochs  = world_epochs           ,
            logger           = world_logger           ,
            ett              = world_ett              ,
            stop_threshold   = world_stop_threshold   ,
            tolerance        = world_tolerance        ,
            clip_coefficient = world_clip_coefficient ,
            slam             = lambda_factor          ,
            gamma            = gamma_factor           ,
        )

        # checkpoint ##################################################
        if episode % etv == 0:
            torch.save({
                "policy_state_dict"           : policy_model.state_dict()           ,
                "world_state_dict"            : world_model.state_dict()            ,
                "policy_optimizer_state_dict" : policy_model_optimizer.state_dict() ,
                "world_optimizer_state_dict"  : world_model_optimizer.state_dict()  ,
                "best_reward"                 : best_reward                         ,
                "episode"                     : episode                             ,
            }, os.path.join(dir,"models.pkl"))

        # evaluation #################################################
        if episode % etv == 0:
            eval_data = evaluate(
                policy_model     = policy_model     ,
                episode          = episode          ,
                world_model      = world_model      ,
                world            = eval_world       ,
                steps            = eval_steps       ,
                envs             = eval_envs        ,
                world_tolerance  = world_tolerance  ,
                logger           = eval_logger
            ) 
            eval_reward = eval_data["rewards"].sum().item()/eval_envs
            max_reward  = eval_data["max_reward"]

            # save best model ##########################################
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save({
                    "policy_state_dict"           : policy_model.state_dict()           ,
                    "world_state_dict"            : world_model.state_dict()            ,
                    "policy_optimizer_state_dict" : policy_model_optimizer.state_dict() ,
                    "world_optimizer_state_dict"  : world_model_optimizer.state_dict()  ,
                    "best_reward"                 : best_reward                         ,
                    "episode"                     : episode                             ,
                }, os.path.join(dir,"best.pkl"))

            # early_stopping #########################################
            if utils.is_early_stopping(eval_data["rewards"], eval_data["max_reward"], **early_stopping): break
            del eval_data

        # update progress bar ########################################
        done_train_envs = episode_data["dones"][-1,:,0].sum().int().item()
        train_reward    = episode_data["rewards"].sum().item()/train_envs
        bar.set_description(f"evalrew:{eval_reward:5.3f}, trainrew:{train_reward:5.3f}, max:{max_reward.mean():5.3f}, dones:{done_train_envs:3d}, episode:{episode:5d}")
        # set up next iteration ######################################
        prev_observations = episode_data["observations"][-1].detach()
        prev_dones        = episode_data["dones"       ][-1].detach()

        # clean up ###################################################
        del episode_data

    torch.save({
        "policy_state_dict"           : policy_model.state_dict()           ,
        "world_state_dict"            : world_model.state_dict()            ,
        "policy_optimizer_state_dict" : policy_model_optimizer.state_dict() ,
        "world_optimizer_state_dict"  : world_model_optimizer.state_dict()  ,
        "episode"                     : episodes                            ,
        "best_reward"                 : best_reward                         ,
    }, os.path.join(dir,"last.pkl"))


