from unroll import unroll
from evaluate import evaluate

import trainers
import models
import torch
import utils
import vmas
import tqdm
import os


def shac(
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
        value_model            : models.Model                           ,
        policy_model_optimizer : torch.optim.Optimizer                  ,
        value_model_optimizer  : torch.optim.Optimizer                  ,
        train_world            : vmas.simulator.environment.Environment ,
        eval_world             : vmas.simulator.environment.Environment ,
        value_cache_size       : int                                    ,
        value_batch_size       : int                                    ,
        value_epochs           : int                                    ,
        value_bins             : int                                    ,
        gamma_factor           : float                                  ,
        lambda_factor          : float                                  ,
        etr                    : int                                    ,
        etv                    : int                                    ,
        compile                : bool                                   ,
        restore_path           : str|None                               ,
        device                 : str                                    ,
        early_stopping         : dict                                   ,
        value_tolerance        : float                                  ,
        value_stop_threshold   : float                                  ,
        policy_clip_coefficient: float                                  ,
        value_clip_coefficient : float                                  ,
        value_ett              : int                                    ,
        use_diffreward         : bool = True                            ,
    ):

    eval_logger   = utils.get_file_logger(os.path.join(dir,  "eval.log"))
    value_logger  = utils.get_file_logger(os.path.join(dir, "value.log"))
    policy_logger = utils.get_file_logger(os.path.join(dir,"policy.log"))

    gammas = utils.gamma_tensor(train_steps, train_envs, agents, gamma_factor).to(device)

    if compile:
        policy_model = torch.compile(policy_model)
        value_model  = torch.compile(value_model)

    checkpoint = dict()
    if restore_path is not None:
        checkpoint = torch.load(restore_path, weights_only=False)
        policy_model.load_state_dict(checkpoint["policy_state_dict"])
        value_model .load_state_dict(checkpoint["value_state_dict" ])
    
    prev_observations : torch.Tensor = torch.zeros(train_envs, observation_size, device=device)
    prev_dones        : torch.Tensor = torch.ones (train_envs, agents, device=device)
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
            use_diffreward= use_diffreward,
        )

        value_model.eval()
        episode_data["proxy_rewards"] = episode_data["rewards"]
        episode_data["values"]        = value_model (episode_data["observations"].flatten(0,1)).view(episode_data["rewards"].shape)
        value_model.train()

        # train actor model ###########################################
        trainers.routines.train_policy(
            episode          = episode                 ,
            policy_model     = policy_model            ,
            episode_data     = episode_data            ,
            optimizer        = policy_model_optimizer  ,
            gammas           = gammas                  ,
            logger           = policy_logger           ,
            clip_coefficient = policy_clip_coefficient ,
        )

        # train value model ###########################################
        trainers.routines.train_value(
            episode                = episode                ,
            model                  = value_model            ,
            optimizer              = value_model_optimizer  ,
            episode_data           = episode_data           ,
            cached_data            = None                   ,
            training_epochs        = value_epochs           ,
            batch_size             = value_batch_size       ,
            cache_size             = None                   ,
            bins                   = value_bins             ,
            slam                   = lambda_factor          ,
            gamma                  = gamma_factor           ,
            logger                 = value_logger           ,
            stop_threshold         = value_stop_threshold   ,
            tolerance              = value_tolerance        ,
            clip_coefficient       = value_clip_coefficient ,
            ett                    = value_ett              ,
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
        "value_state_dict"  : value_model .state_dict(),
        "episode"           : episodes,
        "best_reward"       : best_reward
    }, os.path.join(dir,"last.pkl"))


