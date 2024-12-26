import environments
import experiments
import trainers
import pathlib
import models
import torch
import utils
import os

def shacrm(config):

    os.makedirs(config.dir, exist_ok=False)
    torch.set_float32_matmul_precision("high")
    utils.seed_everything(config.seed)
    torch.use_deterministic_algorithms(config.is_deterministic)

    train_world = environments.get_environment(
        name         = config.env_name   ,
        envs         = config.train_envs ,
        agents       = config.agents     ,
        device       = config.device     ,
        grad_enabled = True              ,
        seed         = config.seed
    )

    eval_world = environments.get_environment(
        name         = config.env_name  ,
        envs         = config.eval_envs ,
        agents       = config.agents    ,
        device       = config.device    ,
        grad_enabled = False            ,
        seed         = config.seed
    )

    config.observation_size = train_world.observation_space[0].shape[0]
    config.action_size      = train_world.action_space[0].shape[0]
    config.action_space     = [train_world.action_space[0].low[0].item(), train_world.action_space[0].high[1].item()]
    utils.save_config(config.dir, config.__dict__)

    policy_model = models.policies.MLPAFO(
        observation_size = config.observation_size   ,
        action_size      = config.action_size        ,
        agents           = config.agents             ,
        steps            = config.train_steps        ,
        action_space     = config.action_space       ,
        layers           = config.policy_layers      ,
        hidden_size      = config.policy_hidden_size ,
        dropout          = config.policy_dropout     ,
        activation       = config.policy_activation  ,
        device           = config.device
    ) if config.policy_nn == "mlp" else (models.policies.Transformer(
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
    ) if config.policy_nn == "transformer" else None)

    value_model  = models.values.MLPAFO(
        observation_size = config.observation_size  ,
        action_size      = config.action_size       ,
        agents           = config.agents            ,
        steps            = config.train_steps       ,
        layers           = config.value_layers      ,
        hidden_size      = config.value_hidden_size ,
        dropout          = config.value_dropout     ,
        activation       = config.value_activation  ,
        device           = config.device
    ) if config.value_nn == "mlp" else (models.values.Transformer(
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
    ) if config.value_nn == "transformer" else None)

    reward_model  = models.rewards.MLPAFO(
        observation_size = config.observation_size  ,
        action_size      = config.action_size       ,
        agents           = config.agents            ,
        steps            = config.train_steps       ,
        layers           = config.reward_layers      ,
        hidden_size      = config.reward_hidden_size ,
        dropout          = config.reward_dropout     ,
        activation       = config.reward_activation  ,
        device           = config.device
    ) if config.reward_nn == "mlp" else (models.rewards.Transformer(
        observation_size = config.observation_size  ,
        action_size      = config.action_size       ,
        agents           = config.agents            ,
        steps            = config.train_steps       ,
        layers           = config.reward_layers      ,
        hidden_size      = config.reward_hidden_size ,
        feedforward_size = config.reward_feedforward ,
        heads            = config.reward_heads       ,
        dropout          = config.reward_dropout     ,
        activation       = config.reward_activation  ,
        device           = config.device
    ) if config.reward_nn == "transformer" else None)

    assert policy_model is not None
    assert value_model  is not None
    assert reward_model is not None

    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_learning_rate)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=config. value_learning_rate)
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=config.reward_learning_rate)

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

    pathlib.Path(os.path.join(config.dir, "done")).touch()


if __name__ == "__main__":
    config = experiments.configs.shacrm
    config.agents           = 3
    config.env_name         = "transport"
    config.dir              = f"data/shacrm/{config.env_name}-a{config.agents}"
    shacrm(config)
