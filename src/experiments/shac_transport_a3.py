import environments
import experiments
import trainers
import models
import torch
import utils
import utils
import os

def run():

    config = experiments.configs.shac
    config.dir              = "data/shac-transport-a3"
    config.observation_size = 11
    config.action_size      = 2
    config.agents           = 3

    config.eval_steps = 512

    os.makedirs(config.dir, exist_ok=False)
    utils.save_config(config.dir, config.__dict__)
    torch.set_float32_matmul_precision("high")
    utils.seed_everything(config.seed)

    value_model  = models.ValueAFO (
        observation_size = config.observation_size  ,
        action_size      = config.action_size       ,
        agents           = config.agents            ,
        steps            = config.train_steps       ,
        layers           = config.value_layers      ,
        hidden_size      = config.value_hidden_size ,
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

    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_learning_rate)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=config. value_learning_rate)

    trainers.shac(
        dir                    = config.dir                     ,
        episodes               = config.episodes                ,
        observation_size       = config.observation_size        ,
        action_size            = config.action_size             ,
        agents                 = config.agents                  ,
        train_envs             = config.train_envs              ,
        train_steps            = config.train_steps             ,
        eval_steps             = config.eval_steps              ,
        eval_envs              = config.eval_envs               ,
        policy_model           = policy_model                   ,
        value_model            = value_model                    ,
        policy_model_optimizer = policy_model_optimizer         ,
        value_model_optimizer  = value_model_optimizer          ,
        train_world            = train_world                    ,
        eval_world             = eval_world                     ,
        value_cache_size       = config.value_cache_size        ,
        value_batch_size       = config.value_batch_size        ,
        value_epochs           = config.value_epochs            ,
        value_bins             = config.value_bins              ,
        value_tolerance        = config.value_tolerance         ,
        value_stop_threshold   = config.value_stop_threshold    ,
        policy_clip_coefficient= config.policy_clip_coefficient ,
        value_clip_coefficient = config.value_clip_coefficient  ,
        value_ett              = config.value_ett               ,
        gamma_factor           = config.gamma_factor            ,
        lambda_factor          = config.lambda_factor           ,
        etr                    = config.etr                     ,
        etv                    = config.etv                     ,
        compile                = config.compile                 ,
        restore_path           = config.restore_path            ,
        device                 = config.device                  ,
        early_stopping         = config.early_stopping          ,
    ) 

if __name__ == "__main__":
    run()

