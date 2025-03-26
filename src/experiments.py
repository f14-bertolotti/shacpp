import experiments
import click

ENVNAMES = [
    "dispersion",
    "transport" ,
    "sampling"  ,
    "discovery" ,
    "ant"       ,
]

NNS = [
    "mlp"  ,
    "transformer",
]

ALGNAMES = [
    "ppo"   ,
    "shac"  ,
    "shacrm",
    "shacwm",
]

MODELS = [
    "policy",
    "value" ,
    "reward",
    "world" ,
]

@click.group()
def cli(): pass

@click.command()
@click.option("--alg-name"      , "algname"       , type=click.Choice(ALGNAMES) , default="ppo"        , help="Algorithm name"                           )
@click.option("--env-name"      , "envname"       , type=click.Choice(ENVNAMES) , default="dispersion" , help="Environment name"                         )
@click.option("--agents"        , "agents"        , type=int                    , default=1            , help="Number of agents"                         )
@click.option("--episodes"      , "episodes"      , type=int                    , default=10000        , help="Number of episodes"                       )
@click.option("--seed"          , "seed"          , type=int                    , default=42           , help="Random seed"                              )
@click.option("--etr"           , "etr"           , type=int                    , default=10           , help="Epochs before envs. reset"                )
@click.option("--etv"           , "etv"           , type=int                    , default=100          , help="Epochs before evaluation"                 )
@click.option("--compile"       , "compile"       , type=bool                   , default=True         , help="Compile the model"                        )
@click.option("--device"        , "device"        , type=str                    , default="cuda:0"     , help="Device to use"                            )
@click.option("--deterministic" , "deterministic" , type=bool                   , default=True         , help="Force deterministic"                      )
@click.option("--dir"           , "dir"           , type=click.Path()           , default=None         , help="Directory to save the results"            )
@click.option("--early-stopping", "early_stopping", type=(float,float)          , default=(.9,.9)      , help="% of env that should achieve % of reward" )
@click.option("--out-coeff"     , "out_coeff"     , type=float                  , default=1            , help="Coefficient for actions outside the space")
@click.option("--var"           , "var"           , type=float                  , default=1            , help="Variance for the policy model"            )
@click.option("--restore-path"  , "restore_path"  , type=click.Path()           , default=None         , help="Path to restore the model"                )
@click.option("--lambda"        , "lambda_factor" , type=float                  , default=0.95         , help="Lambda factor for the RL model"           )
@click.option("--gamma"         , "gamma_factor"  , type=float                  , default=0.99         , help="Gamma factor for the RL model"            )
@click.option("--train-envs"    , "train_envs"    , type=int                    , default=512          , help="Number of environments to train"          )
@click.option("--train-steps"   , "train_steps"   , type=int                    , default=32           , help="Number of steps to train"                 )
@click.option("--eval-envs"     , "eval_envs"     , type=int                    , default=512          , help="Number of environments to evaluate"       )
@click.option("--eval-steps"    , "eval_steps"    , type=int                    , default=512          , help="Number of steps to evaluate"              )
@click.option("--log-grads"     , "log_grads"     , type=bool                   , default=False        , help="Log gradients"                            )
@click.pass_context
def run(ctx, algname, envname, agents, episodes, seed, etr, etv, compile, device, deterministic, dir, early_stopping, out_coeff, var, restore_path, lambda_factor, gamma_factor, train_envs, train_steps, eval_envs, eval_steps, log_grads):
    config = getattr(experiments.configs, algname)
    config.compile  = compile
    config.device   = device
    config.env_name = envname
    config.agents   = agents
    config.episodes = episodes
    config.seed     = seed
    config.etr      = etr
    config.etv      = etv
    config.var      = var
    config.train_envs  = train_envs
    config.train_steps = train_steps
    config.eval_envs   = eval_envs
    config.eval_steps  = eval_steps
    config.gamma_factor = gamma_factor
    config.lambda_factor = lambda_factor
    config.is_deterministic = deterministic
    config.out_coefficient  = out_coeff
    config.early_stopping = {
        "max_reward_fraction" : early_stopping[0],
        "max_envs_fraction"   : early_stopping[1]
    }
    config.restore_path = restore_path
    config.log_grads    = log_grads

    if ctx.obj is not None:
        for key, value in ctx.obj.items():
            setattr(config, key, value)

    config.dir      = f"data/{algname}/{envname}/{agents}/{config.policy_nn}/{seed}" if dir is None else dir
    getattr(experiments,algname)(config)

@click.group(invoke_without_command=True)
@click.option("--model"         , "model"         , type=click.Choice(MODELS) , default="policy", help="Model to configure"               )
@click.option("--layers"        , "layers"        , type=int                  , default=None    , help="Number of layers for the model"   )
@click.option("--hidden-size"   , "hidden_size"   , type=int                  , default=None    , help="Hidden size for the model"        )
@click.option("--feedforward"   , "feedforward"   , type=int                  , default=None    , help="Feedforward size for the model"   )
@click.option("--heads"         , "heads"         , type=int                  , default=None    , help="Number of heads for the model"    )
@click.option("--dropout"       , "dropout"       , type=float                , default=None    , help="Dropout for the model"            )
@click.option("--activation"    , "activation"    , type=str                  , default=None    , help="Activation for the model"         )
@click.option("--learning-rate" , "learning_rate" , type=float                , default=None    , help="Learning rate for the model"      )
@click.option("--epochs"        , "epochs"        , type=int                  , default=None    , help="Number of epochs for the model"   )
@click.option("--ett"           , "ett"           , type=int                  , default=None    , help="Epochs before training the model" )
@click.option("--clip-coeff"    , "clip_coeff"    , type=float                , default=None    , help="Clip coefficient for the model"   )
@click.option("--tolerance"     , "tolerance"     , type=float                , default=None    , help="Tolerance for the model"          )
@click.option("--stop-threshold", "stop_threshold", type=float                , default=None    , help="Stop threshold for the model"     )
@click.option("--cache-size"    , "cache_size"    , type=int                  , default=None    , help="Cache size for the model"         )
@click.option("--bins"          , "bins"          , type=int                  , default=None    , help="Number of bins for the cach"      )
@click.option("--nn"            , "nn"            , type=click.Choice(NNS)    , default="transformer", help="Neural network to use"       )
@click.pass_context
def modelcfg(ctx, model, layers, hidden_size, feedforward, heads, dropout, activation, learning_rate, epochs, ett, clip_coeff, tolerance, stop_threshold, cache_size, bins, nn):
    """ command to specify model configuration parameter if needed """
    if ctx.obj is None:
        ctx.obj = {}

    if layers         is not None: ctx.obj[f"{model}_layers"         ] =  layers
    if hidden_size    is not None: ctx.obj[f"{model}_hidden_size"    ] =  hidden_size
    if feedforward    is not None: ctx.obj[f"{model}_feedforward"    ] =  feedforward
    if heads          is not None: ctx.obj[f"{model}_heads"          ] =  heads
    if dropout        is not None: ctx.obj[f"{model}_dropout"        ] =  dropout
    if activation     is not None: ctx.obj[f"{model}_activation"     ] =  activation
    if learning_rate  is not None: ctx.obj[f"{model}_learning_rate"  ] =  learning_rate
    if epochs         is not None: ctx.obj[f"{model}_epochs"         ] =  epochs
    if ett            is not None: ctx.obj[f"{model}_ett"            ] =  ett
    if clip_coeff     is not None: ctx.obj[f"{model}_clip_coeff"     ] =  clip_coeff
    if tolerance      is not None: ctx.obj[f"{model}_tolerance"      ] =  tolerance
    if stop_threshold is not None: ctx.obj[f"{model}_stop_threshold" ] =  stop_threshold
    if cache_size     is not None: ctx.obj[f"{model}_cache_size"     ] =  cache_size
    if bins           is not None: ctx.obj[f"{model}_bins"           ] =  bins
    if nn             is not None: ctx.obj[f"{model}_nn"             ] =  nn

cli.add_command(modelcfg)
cli.add_command(run)
modelcfg.add_command(run)
modelcfg.add_command(modelcfg)

if __name__ == "__main__":
    cli()


