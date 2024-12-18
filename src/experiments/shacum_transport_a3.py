import environments
import experiments
import trainers
import models
import torch
import utils
import utils
import os

def run():

    config = experiments.configs.shacwm # base configuration
    config.seed             = 42
    config.environment      = "transport"
    config.observation_size = 11
    config.action_size      = 2
    config.agents           = 3
    config.action_space     = [-1,+1]
    config.dir              = f"data/shacwm-{config.environment}-a{config.agents}-s{config.seed}"

    # redefine configuration for transport environment
    config.eval_steps = 512
    config.etr        = 10

    # To Be Removed
    config.compile    = False
    config.etv = 10
    config.is_deterministic = False

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
        grad_enabled = False              ,
    )

    eval_world = environments.get_environment(
        name         = config.environment ,
        envs         = config.eval_envs   ,
        agents       = config.agents      ,
        device       = config.device      ,
        seed         = config.seed        ,
        grad_enabled = False              ,
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

    world_model = models.worlds.AxisTransformer(
        observation_size = config.observation_size       ,
        action_size      = config.action_size            ,
        agents           = config.agents                 ,
        steps            = config.train_steps            ,
        layers           = config.world_layers           ,
        hidden_size      = config.world_hidden_size      ,
        dropout          = config.world_dropout          ,
        heads            = config.world_heads            ,
        device           = config.device                 ,
        compute_reward   = True                          ,
        compute_value    = True                          ,
    )

    # get optimizers
    world_model_optimizer  = torch.optim.Adam( world_model.parameters(), lr=config. world_learning_rate) 
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_learning_rate)

    # train
    trainers.shacum(
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
        world_cache_size        = config.world_cache_size        ,
        world_batch_size        = config.world_batch_size        ,
        world_epochs            = config.world_epochs            ,
        gamma_factor            = config.gamma_factor            ,
        lambda_factor           = config.lambda_factor           ,
        world_bins              = config.world_bins              ,
        etr                     = config.etr                     ,
        etv                     = config.etv                     ,
        compile                 = config.compile                 ,
        restore_path            = config.restore_path            ,
        device                  = config.device                  ,
        early_stopping          = config.early_stopping          ,
        world_clip_coefficient  = config.world_clip_coefficient  ,
        policy_clip_coefficient = config.policy_clip_coefficient ,
        out_coefficient         = config.out_coefficient         ,
        world_ett               = config.world_ett               ,
        world_stop_threshold    = config.world_stop_threshold    ,
        world_tolerance         = config.world_tolerance         ,
    )

if __name__ == "__main__":
    run()

