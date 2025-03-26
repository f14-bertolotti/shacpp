import click
import utils

common_options = utils.chain(
    click.option("--device"            , "device"            , type=str          , default="cuda:0" , help="random device"                                 ),
    click.option("--seed"              , "seed"              , type=int          , default=42       , help="random seed"                                   ),
    click.option("--episodes"          , "episodes"          , type=int          , default=500      , help="episodes before resetting the environement"    ),
    click.option("--observation-size"  , "observation_size"  , type=int          , default=2        , help="observation size"                              ),
    click.option("--action-size"       , "action_size"       , type=int          , default=11       , help="action size"                                   ),
    click.option("--agents"            , "agents"            , type=int          , default=5        , help="number of agents"                              ),
    click.option("--train-envs"        , "train_envs"        , type=int          , default=512      , help="number of train environments"                  ),
    click.option("--eval-envs"         , "eval_envs"         , type=int          , default=512      , help="number of evaluation environments"             ),
    click.option("--train-steps"       , "train_steps"       , type=int          , default=32       , help="number of steps for the training rollout"      ),
    click.option("--eval-steps"        , "eval_steps"        , type=int          , default=64       , help="number of steps for the evaluation rollout"    ),
    click.option("--dir"               , "dir"               , type=click.Path() , default="./"     , help="directory in which store logs and checkpoints" ),
    click.option("--restore-path"      , "restore_path"      , type=click.Path() , default=None     , help="path to a checkpoint to restore"               ),
    click.option("--etr"               , "etr"               , type=int          , default=5        , help="epochs between environment resets"             ),
    click.option("--etv"               , "etv"               , type=int          , default=10       , help="epochs between evaluations"                    ),
    click.option("--compile"           , "compile"           , type=bool         , default=False    , help="compile the model"                             ),
)
