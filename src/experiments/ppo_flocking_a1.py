import environments
import experiments
import trainers
import models
import torch
import utils
import utils
import os

def run():

    config = experiments.configs.ppo # base configuration
    config.seed             = 42
    config.environment      = "flocking"
    config.observation_size = 11
    config.action_size      = 2
    config.agents           = 3
    config.action_space     = [-1.0,+1.0]
    config.dir              = f"data/ppo-{config.environment}-a{config.agents}-s{config.seed}"

    # redefine configuration for transport environment
    config.eval_steps = 512
    config.etr        = 10
    config.compile = False
    config.etv = 10

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

    # get optimizers
    policy_model_optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.policy_learning_rate)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=config. value_learning_rate)

    trainers.ppo(
        dir                    = config.dir              ,
        episodes               = config.episodes         ,
        observation_size       = config.observation_size ,
        action_size            = config.action_size      ,
        agents                 = config.agents           ,
        train_envs             = config.train_envs       ,
        train_steps            = config.train_steps      ,
        eval_envs              = config.eval_envs        ,
        eval_steps             = config.eval_steps       ,
        policy_model           = policy_model            ,
        value_model            = value_model             ,
        policy_model_optimizer = policy_model_optimizer  ,
        value_model_optimizer  = value_model_optimizer   ,
        train_world            = train_world             ,
        eval_world             = eval_world              ,
        batch_size             = config.batch_size       ,
        epochs                 = config.epochs           ,
        gamma_factor           = config.gamma_factor     ,
        etr                    = config.etr              ,
        etv                    = config.etv              ,
        compile                = config.compile          ,
        restore_path           = config.restore_path     ,
        early_stopping         = config.early_stopping   ,
    )

 

if __name__ == "__main__":
    run()

