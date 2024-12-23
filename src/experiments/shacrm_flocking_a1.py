import environments
import experiments
import trainers
import models
import torch
import utils
import utils
import os

def run():

    config = experiments.configs.shacrm # base configuration
    config.seed             = 42
    config.environment      = "flocking"
    config.observation_size = 18
    config.action_size      = 2
    config.agents           = 1
    config.action_space     = [-1.0,+1.0]
    config.dir              = f"data/shacrm-{config.environment}-a{config.agents}-s{config.seed}"

    # redefine configuration for transport environment
    config.eval_steps = 512
    config.etr        = 10

    # setup
    os.makedirs(config.dir, exist_ok=False)
    os.system(f"cp -r src {config.dir}")
    utils.save_config(config.dir, config.__dict__)
    torch.set_float32_matmul_precision("high")
    utils.seed_everything(config.seed)
    torch.use_deterministic_algorithms(config.is_deterministic)

    # get environments
    train_world = environments.get_environment(
        name         = config.environment ,
        envs         = config.train_envs  ,
        agents       = config.agents      ,
        device       = config.device      ,
        seed         = config.seed        ,
        grad_enabled = True               ,
    )

    eval_world = environments.get_environment(
        name         = config.environment ,
        envs         = config.eval_envs   ,
        agents       = config.agents      ,
        device       = config.device      ,
        seed         = config.seed        ,
        grad_enabled = False              ,
    )

    # get models
    value_model  = models.values.Transformer(
        observation_size = config.observation_size  ,
        action_size      = config.action_size       ,
        agents           = config.agents            ,
        steps            = config.train_steps       ,
        layers           = config.value_layers      ,
        hidden_size      = config.value_hidden_size ,
        feedforward_size = config.value_feedforward ,
        heads            = config.value_heads       ,
        dropout          = config.value_dropout     ,
        activation       = config.value_activation  ,
        device           = config.device
    )

    policy_model = models.policies.Transformer(
        observation_size = config.observation_size   ,
        action_size      = config.action_size        ,
        agents           = config.agents             ,
        steps            = config.train_steps        ,
        action_space     = config.action_space       ,
        layers           = config.policy_layers      ,
        hidden_size      = config.policy_hidden_size ,
        feedforward_size = config.policy_feedforward ,
        dropout          = config.policy_dropout     ,
        heads            = config.policy_heads       ,
        activation       = config.policy_activation  ,
        var              = config.policy_var         ,
        device           = config.device
    )

    reward_model = models.rewards.Transformer(
        observation_size = config.observation_size   ,
        action_size      = config.action_size        ,
        agents           = config.agents             ,
        steps            = config.train_steps        ,
        layers           = config.reward_layers      ,
        hidden_size      = config.reward_hidden_size ,
        feedforward_size = config.reward_feedforward ,
        heads            = config.reward_heads       ,
        dropout          = config.reward_dropout     ,
        activation       = config.reward_activation  ,
        device           = config.device
    )

    # get optimizers
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=config.reward_learning_rate) 
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_learning_rate)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=config. value_learning_rate)

    # train
    trainers.shacrm(
        dir                     = config.dir                     ,
        episodes                = config.episodes                ,
        observation_size        = config.observation_size        ,
        action_size             = config.action_size             ,
        agents                  = config.agents                  ,
        train_envs              = config.train_envs              ,
        train_steps             = config.train_steps             ,
        eval_steps              = config.eval_steps              ,
        eval_envs               = config.eval_envs               ,
        policy_model            = policy_model                   ,
        reward_model            = reward_model                   ,
        value_model             = value_model                    ,
        reward_model_optimizer  = reward_model_optimizer         ,
        policy_model_optimizer  = policy_model_optimizer         ,
        value_model_optimizer   = value_model_optimizer          ,
        train_world             = train_world                    ,
        eval_world              = eval_world                     ,
        reward_cache_size       = config.reward_cache_size       ,
        reward_batch_size       = config.reward_batch_size       ,
        reward_epochs           = config.reward_epochs           ,
        reward_bins             = config.reward_bins             ,
        value_cache_size        = config.value_cache_size        ,
        value_batch_size        = config.value_batch_size        ,
        value_epochs            = config.value_epochs            ,
        value_bins              = config.value_bins              ,
        gamma_factor            = config.gamma_factor            ,
        lambda_factor           = config.lambda_factor           ,
        etr                     = config.etr                     ,
        etv                     = config.etv                     ,
        compile                 = config.compile                 ,
        restore_path            = config.restore_path            ,
        device                  = config.device                  ,
        early_stopping          = config.early_stopping          ,
        value_tolerance         = config.value_tolerance         ,
        reward_tolerance        = config.reward_tolerance        ,
        value_stop_threshold    = config.value_stop_threshold    ,
        reward_stop_threshold   = config.reward_stop_threshold   ,
        reward_clip_coefficient = config.reward_clip_coefficient ,
        policy_clip_coefficient = config.policy_clip_coefficient ,
        value_clip_coefficient  = config.value_clip_coefficient  ,
        out_coefficient         = config.out_coefficient         ,
        reward_ett              = config.reward_ett              ,
        value_ett               = config.value_ett               ,
    ) 

if __name__ == "__main__":
    run()

