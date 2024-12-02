import environments
import experiments
import trainers
import models
import torch
import utils
import utils
import os

def run():

    config = experiments.configs.shacrm
    config.dir              = "data/shacrm-transport-a3-1"
    config.observation_size = 11
    config.action_size      = 2
    config.agents           = 3

    config.eval_steps = 512

    config.value_activation   = "GELU"
    config.value_hidden_size  = 64
    config.value_feedforward  = 128
    config.value_heads        = 1
    config.reward_activation  = "GELU"
    config.reward_hidden_size = 64
    config.reward_feedforward = 128
    config.reward_heads       = 1
    config.policy_activation  = "GELU"
    config.policy_hidden_size = 64
    config.policy_feedforward = 128
    config.policy_heads       = 1
    config.policy_dropout     = 0.1
    config.reward_ett = 10
    config.value_ett = 10
    config.reward_epochs = 30
    config.value_epochs  = 30
    config.value_cache_size = 10000
    config.reward_cache_size = 10000
    config.reward_clip_coefficient = None
    config.policy_clip_coefficient = 1
    config. value_clip_coefficient = None
    config.value_stop_threshold = .99
    config.reward_stop_threshold = .99
    config.value_learning_rate = 0.001
    config.reward_learning_rate = 0.001
    config.etv = 100

    config.reward_batch_size = 1000
    config.value_batch_size  = 1000

    os.makedirs(config.dir, exist_ok=False)
    utils.save_config(config.dir, config.__dict__)
    torch.set_float32_matmul_precision("high")
    utils.seed_everything(config.seed)

    value_model  = models.values.TransformerValue(
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

    policy_model = models.policies.TransformerPolicy(
        observation_size = config.observation_size   ,
        action_size      = config.action_size        ,
        agents           = config.agents             ,
        steps            = config.train_steps        ,
        layers           = config.policy_layers      ,
        hidden_size      = config.policy_hidden_size ,
        feedforward_size = config.policy_feedforward ,
        heads            = config.policy_heads       ,
        dropout          = config.policy_dropout     ,
        activation       = config.policy_activation  ,
        device           = config.device
    )

    reward_model = models.rewards.TransformerReward(
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

    train_world = environments.get_environment(
        name         = "transport"       ,
        envs         = config.train_envs ,
        agents       = config.agents     ,
        device       = config.device     ,
        grad_enabled = True              ,
        seed         = config.seed
    )

    eval_world = environments.get_environment(
        name         = "transport"      ,
        envs         = config.eval_envs ,
        agents       = config.agents    ,
        device       = config.device    ,
        grad_enabled = False            ,
        seed         = config.seed
    )

    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=config.reward_learning_rate) 
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_learning_rate)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=config. value_learning_rate)

    trainers.shacrm(
        dir                    = config.dir               ,
        episodes               = config.episodes          ,
        observation_size       = config.observation_size  ,
        action_size            = config.action_size       ,
        agents                 = config.agents            ,
        train_envs             = config.train_envs        ,
        train_steps            = config.train_steps       ,
        eval_steps             = config.eval_steps        ,
        eval_envs              = config.eval_envs         ,
        policy_model           = policy_model             ,
        reward_model           = reward_model             ,
        value_model            = value_model              ,
        reward_model_optimizer = reward_model_optimizer   ,
        policy_model_optimizer = policy_model_optimizer   ,
        value_model_optimizer  = value_model_optimizer    ,
        train_world            = train_world              ,
        eval_world             = eval_world               ,
        reward_cache_size      = config.reward_cache_size ,
        reward_batch_size      = config.reward_batch_size ,
        reward_epochs          = config.reward_epochs     ,
        reward_bins            = config.reward_bins       ,
        value_cache_size       = config.value_cache_size  ,
        value_batch_size       = config.value_batch_size  ,
        value_epochs           = config.value_epochs      ,
        value_bins             = config.value_bins        ,
        gamma_factor           = config.gamma_factor      ,
        lambda_factor          = config.lambda_factor     ,
        etr                    = config.etr               ,
        etv                    = config.etv               ,
        compile                = config.compile           ,
        restore_path           = config.restore_path      ,
        device                 = config.device            ,
        early_stopping         = config.early_stopping    ,
        value_tolerance        = config.value_tolerance   ,
        reward_tolerance       = config.reward_tolerance  ,
        value_stop_threshold   = config.value_stop_threshold,
        reward_stop_threshold  = config.reward_stop_threshold,
        reward_clip_coefficient= config.reward_clip_coefficient,
        policy_clip_coefficient= config.policy_clip_coefficient,
        value_clip_coefficient = config.value_clip_coefficient,
        reward_ett             = config.reward_ett         ,
        value_ett              = config.value_ett          ,
    ) 

if __name__ == "__main__":
    run()

