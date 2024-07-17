import utils, click

environment = utils.decochain(
    click.option("--envs"            , "envs"            , type=int          , default=64      , help="number of parallel environments."                                             ),
    click.option("--device"          , "device"          , type=str          , default="cuda:0", help="device for the environments: cuda or cpu."                                    ),
    click.option("--seed"            , "seed"            , type=int          , default=None    , help="environment seed."                                                            ),
    click.option("--agents"          , "agents"          , type=int          , default=3       , help="number of agents per environment."                                            ),
    click.option("--shared-reward"   , "shared_reward"   , type=bool         , default=False   , help="True if all agents share the reward."                                         ),
    click.option("--requires-grad"   , "grad_enabled"    , type=bool         , default=False   , help="True if one can backpropagate in the simulator."                              ),
    click.option("--rms"             , "rms"             , type=bool         , default=False   , help="True if statisticts and normalization of the observation should be computed." ),
    click.option("--state-dict-path" , "state_dict_path" , type=click.Path() , default=None    , help="Path to where environment data has been stored."                              )
)

proxy_reward = utils.decochain(
    click.option("--dataset-size" , "dataset_size" , type=int           , default = 10000 , help="maximum dataset size to train the reward function."                   ),
    click.option("--batch-size"   , "batch_size"   , type=int           , default = 500   , help="batch size to train the reward function."                             ),
    click.option("--lamb"         , "lamb"         , type=int           , default = 8     , help="lambda factor in pert distribution of the reward dataset."            ),
    click.option("--atol"         , "atol"         , type=float         , default = .1    , help="tolerance for the accuracy."                                          ),
    click.option("--threshold"    , "threshold"    , type=float         , default = .9    , help="threshold accuracy to end reward training."                           ),
    click.option("--epochs"       , "epochs"       , type=int           , default = None  , help="maximum number of epoch to train the reward function (None means inf)"),
    click.option("--drop-last"    , "drop_last"    , type=bool          , default = True  , help="Drops the last batch in reward training."                             ),
    click.option("--shuffle"      , "shuffle"      , type=bool          , default = True  , help="Shuffles the dataset before each epoch in reward training."           ),
)
                                                                                                                                                                
