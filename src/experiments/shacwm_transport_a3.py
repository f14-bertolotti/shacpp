import environments
import experiments
import trainers
import models
import torch
import utils
import utils
import os

def run():

    config = experiments.configs.shacwm
    config.dir              = "data/shacwm-transport-a3"
    config.observation_size = 11
    config.action_size      = 2
    config.agents           = 3

    config.eval_steps = 512
    config.policy_dropout = 0.3
    config.polict_hidden_size = 64

    os.makedirs(config.dir, exist_ok=False)
    utils.save_config(config.dir, config.__dict__)
    torch.set_float32_matmul_precision("high")
    utils.seed_everything(config.seed)

    policy_model = models.PolicyAFO(
        observation_size = config.observation_size   ,
        action_size      = config.action_size        ,
        agents           = config.agents             ,
        steps            = config.train_steps        ,
        layers           = config.policy_layers      ,
        hidden_size      = config.policy_hidden_size ,
        dropout          = config.policy_dropout     ,
        activation       = config.policy_activation  ,
        device           = config.device
    )

    world_model = models.worlds.AxisTransformerWorld(
        observation_size = config.observation_size       ,
        action_size      = config.action_size            ,
        agents           = config.agents                 ,
        steps            = config.train_steps            ,
        layers           = config.world_layers           ,
        hidden_size      = config.world_hidden_size      ,
        feedforward_size = config.world_feedforward_size ,
        dropout          = config.world_dropout          ,
        activation       = config.world_activation       ,
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

    world_model_optimizer  = torch.optim.Adam( world_model.parameters(), lr=config. world_learning_rate) 
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_learning_rate)

    trainers.shacwm(
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
        world_model             = world_model                    ,
        world_model_optimizer   = world_model_optimizer          ,
        policy_model_optimizer  = policy_model_optimizer         ,
        train_world             = train_world                    ,
        eval_world              = eval_world                     ,
        cache_size              = config.cache_size              ,
        world_batch_size        = config.world_batch_size        ,
        world_epochs            = config.world_epochs            ,
        gamma_factor            = config.gamma_factor            ,
        lambda_factor           = config.lambda_factor           ,
        reward_bins             = config.reward_bins             ,
        etr                     = config.etr                     ,
        etv                     = config.etv                     ,
        compile                 = config.compile                 ,
        restore_path            = config.restore_path            ,
        device                  = config.device                  ,
        early_stopping          = config.early_stopping          ,
        policy_clip_coefficient = config.policy_clip_coefficient ,
        world_clip_coefficient  = config.world_clip_coefficient  ,
        world_ett               = config.world_ett               ,
    )

if __name__ == "__main__":
    run()

