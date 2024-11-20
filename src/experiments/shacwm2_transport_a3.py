import environments
import experiments
import trainers
import models
import torch
import utils
import utils
import os

def run():

    config = experiments.configs.shacwm2
    config.dir              = "data/shacwm2-transport-a3-2-r"
    config.observation_size = 11
    config.action_size      = 2
    config.agents           = 3

    config.eval_steps = 512

    config.episodes = 100000
    config.value_cache_size = 10000
    config.reward_cache_size = 10000
    config.world_cache_size = 100000
    config.value_bins = 10
    config.reward_bins = 10
    config.world_bins = 100
    config.etv = 100
    config.world_ett = 10
    config.reward_ett = 10
    config.value_ett = 10
    config.value_stop_threshold = None
    config.reward_stop_threshold = None
    config.world_stop_threshold = None
    config.var = 5
    config.value_activation = "ReLU"
    config.value_hidden_size = 64
    config.value_feedforward = 128
    #config.reward_activation = "ReLU"
    #config.reward_hidden_size = 32
    #config.reward_feedforward = 64



    os.makedirs(config.dir, exist_ok=False)
    utils.save_config(config.dir, config.__dict__)
    torch.set_float32_matmul_precision("high")
    utils.seed_everything(config.seed)

    value_model  = models.TransformerValue(
        observation_size = config.observation_size  ,
        action_size      = config.action_size       ,
        agents           = config.agents            ,
        steps            = config.train_steps       ,
        layers           = config.value_layers      ,
        hidden_size      = config.value_hidden_size ,
        feedforward_size = config.value_feedforward ,
        heads            = 1,
        dropout          = config.value_dropout     ,
        activation       = config.value_activation  ,
        device           = config.device
    )

    policy_model = models.PolicyAFO(
        observation_size = config.observation_size   ,
        action_size      = config.action_size        ,
        agents           = config.agents             ,
        steps            = config.train_steps        ,
        layers           = config.policy_layers      ,
        hidden_size      = config.policy_hidden_size ,
        dropout          = config.policy_dropout     ,
        activation       = config.policy_activation  ,
        var              = config.var                ,
        device           = config.device
    )

    reward_model = models.RewardAFO(
        observation_size = config.observation_size   ,
        action_size      = config.action_size        ,
        agents           = config.agents             ,
        steps            = config.train_steps        ,
        layers           = config.reward_layers      ,
        hidden_size      = config.reward_hidden_size ,
        #feedforward_size = config.reward_feedforward ,
        #heads            = 1                         ,
        dropout          = config.reward_dropout     ,
        activation       = config.reward_activation  ,
        device           = config.device
    )

    world_model = models.worlds.AxisTransformerWorld(
        observation_size = config.observation_size       ,
        action_size      = config.action_size            ,
        agents           = config.agents                 ,
        steps            = config.train_steps            ,
        layers           = 2*config.world_layers         ,
        hidden_size      = config.world_hidden_size      ,
        dropout          = config.world_dropout          ,
        device           = config.device                 ,
        compute_reward   = False                         ,
        compute_value    = False                         ,
    )

    train_world = environments.get_environment(
        name         = "transport"       ,
        envs         = config.train_envs ,
        agents       = config.agents     ,
        device       = config.device     ,
        grad_enabled = False              ,
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

    world_model_optimizer  = torch.optim.Adam( world_model.parameters(), lr=config. world_learning_rate) 
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_learning_rate)
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=config.reward_learning_rate)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=config. value_learning_rate)

    trainers.shacwm2(
        dir                    = config.dir                      ,
        episodes               = config.episodes                 ,
        observation_size       = config.observation_size         ,
        action_size            = config.action_size              ,
        agents                 = config.agents                   ,
        train_envs             = config.train_envs               ,
        train_steps            = config.train_steps              ,
        eval_steps             = config.eval_steps               ,
        eval_envs              = config.eval_envs                ,
        policy_model           = policy_model                    ,
        value_model            = value_model                     ,
        reward_model           = reward_model                    ,
        world_model            = world_model                     ,
        world_model_optimizer  = world_model_optimizer           ,
        policy_model_optimizer = policy_model_optimizer          ,
        reward_model_optimizer = reward_model_optimizer          ,
        value_model_optimizer  = value_model_optimizer           ,
        train_world            = train_world                     ,
        eval_world             = eval_world                      ,
        world_cache_size       = config.world_cache_size         ,
        value_cache_size       = config.value_cache_size         ,
        reward_cache_size      = config.reward_cache_size        ,
        world_batch_size       = config.world_batch_size         ,
        reward_batch_size      = config.reward_batch_size        ,
        value_batch_size       = config.value_batch_size         ,
        world_epochs           = config.world_epochs             ,
        reward_epochs          = config.reward_epochs            ,
        value_epochs           = config.value_epochs             ,
        gamma_factor           = config.gamma_factor             ,
        lambda_factor          = config.lambda_factor            ,
        world_bins             = config.world_bins               ,
        reward_bins            = config.reward_bins              ,
        value_bins             = config.value_bins               ,
        etr                    = config.etr                      ,
        etv                    = config.etv                      ,
        compile                = config.compile                  ,
        restore_path           = config.restore_path             ,
        device                 = config.device                   ,
        early_stopping         = config.early_stopping           ,
        value_clip_coefficient  = config.value_clip_coefficient  ,
        reward_clip_coefficient = config.reward_clip_coefficient ,
        world_clip_coefficient  = config.world_clip_coefficient  ,
        policy_clip_coefficient = config.policy_clip_coefficient ,
        world_ett               = config.world_ett               ,
        reward_ett              = config.reward_ett              ,
        value_ett               = config.value_ett               ,
        world_stop_threshold    = config.world_stop_threshold    ,
        reward_stop_threshold   = config.reward_stop_threshold   ,
        value_stop_threshold    = config.value_stop_threshold    ,
        world_tolerance         = config.world_tolerance         ,
        reward_tolerance        = config.reward_tolerance        ,
        value_tolerance         = config.value_tolerance         ,
    )

if __name__ == "__main__":
    run()

