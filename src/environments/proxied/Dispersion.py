from optimizers import add_adam_command, add_sgd_command
from schedulers import add_constant_command, add_cosine_command
from environments import Dispersion, Options, Proxify, proxied
from nn import MLP
import torch, click


@proxied.group(invoke_without_command=True)
@Options.environment
@Options.proxy_reward
@click.pass_obj
def dispersion(trainer, envs, agents, seed, grad_enabled, device, shared_reward, rms, state_dict_path, dataset_size, batch_size, lamb, atol, threshold, shuffle, drop_last, epochs):
    if hasattr(trainer, "environment"): return
    trainer.set_environment(
        Proxify(
            environment  = Dispersion(
                envs         = envs          ,
                agents       = agents        ,
                seed         = seed          ,
                device       = device        ,
                shared_reward= shared_reward ,
                rms          = rms           ,
                grad_enabled = grad_enabled
            ),
            trainer      = trainer      ,
            dataset_size = dataset_size ,
            batch_size   = batch_size   ,
            lamb         = lamb         ,
            atol         = atol         ,
            threshold    = threshold    ,
            shuffle      = shuffle      ,
            drop_last    = drop_last    ,
            epochs       = epochs
        )
    )
    if state_dict_path: trainer.environment.rms = torch.load(state_dict_path)["rms"]

proxied.add_command(dispersion)

@dispersion.group()
def optimizer(): pass
add_adam_command(optimizer, srcnav=lambda x:x.environment, tgtnav=lambda x:x.environment.rewardnn, attrname="set_optimizer")
add_sgd_command (optimizer, srcnav=lambda x:x.environment, tgtnav=lambda x:x.environment.rewardnn, attrname="set_optimizer")

@dispersion.group()
def scheduler(): pass
add_cosine_command  (scheduler, srcnav=lambda x:x.environment, tgtnav=lambda x:x.environment, attrname="set_scheduler")
add_constant_command(scheduler, srcnav=lambda x:x.environment, tgtnav=lambda x:x.environment, attrname="set_scheduler")


@dispersion.group(invoke_without_command=True)
@click.option("--layers"          , "layers"          , type=int          , default=1      , help="layers"     )
@click.option("--hidden-size"     , "hidden_size"     , type=int          , default=64     , help="hidden size")
@click.option("--dropout"         , "dropout"         , type=float        , default=.1     , help="dropout")
@click.option("--activation"      , "activation"      , type=str          , default="ReLU" , help="activation")
@click.option("--state-dict-path" , "state_dict_path" , type=click.Path() , default=None   , help="path to restore the model state")
@click.pass_obj
def mlp(trainer, hidden_size, layers, dropout, activation, state_dict_path):
    env = trainer.environment
    env.set_reward_nn(
        MLP(
            output_size = 1                                                  ,
            input_size  = env.get_action_size() + env.get_observation_size() ,
            layers      = layers                                             ,
            dropout     = dropout                                            ,
            hidden_size = hidden_size                                        ,
            activation  = activation                                         ,
        )
    )

    if state_dict_path: trainer.agent.load_state_dict(torch.load(state_dict_path)["rewardsd"])

