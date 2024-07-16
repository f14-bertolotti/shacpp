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

