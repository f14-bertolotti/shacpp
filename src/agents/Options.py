import utils, click

transformer = utils.decochain(
    click.option("--layers"           , "layers"           , type=int          , default=3      , help="Number of layers."                                            ) ,
    click.option("--embedding-size"   , "embedding_size"   , type=int          , default=64     , help="Embedding layer size."                                        ) ,
    click.option("--feedforward-size" , "feedforward_size" , type=int          , default=256    , help="Feed Forward layer size."                                     ) ,
    click.option("--heads"            , "heads"            , type=int          , default=2      , help="Number of attention heads."                                   ) ,
    click.option("--activation"       , "activation"       , type=str          , default="gelu" , help="Activation function."                                         ) ,
    click.option("--shared"           , "shared"           , type=bool         , default=True   , help="True if parameter should be shared between actor and critic." ) ,
    click.option("--state-dict-path"  , "state_dict_path"  , type=click.Path() , default=None   , help="Path from where the parameters should be loaded from."        ) ,
    click.option("--compile"          , "compile"          , type=bool         , default=True   , help="True if the model should be compiled before being returned."  ) ,
    click.option("--actor-init-gain"  , "actor_init_gain"  , type=float        , default=0.1    , help="Actor orthogonal initialization gain."                        ) ,
    click.option("--critic-init-gain" , "critic_init_gain" , type=float        , default=1.41   , help="Critic orthogonal initialization gain."                       )
)

mlp = utils.decochain(
    click.option("--hidden-size"      , "hidden_size"      , type=int          , default=64    , help="Hidden number of neurons."                                           ) ,
    click.option("--shared"           , "shared"           , type=bool         , default=False , help="True if the first layers should be shared between critic and actor." ) ,
    click.option("--layers"           , "layers"           , type=int          , default=1     , help="Number of layer of the network."                                     ) ,
    click.option("--state-dict-path"  , "state_dict_path"  , type=click.Path() , default=None  , help="Path to the agent to be loaded."                                     ) ,
    click.option("--compile"          , "compile"          , type=bool         , default=True  , help="True if the agent should be compiled"                                ) ,
    click.option("--actor-init-gain"  , "actor_init_gain"  , type=float        , default=0.1   , help="Actor orthogonal initialization gain."                               ) ,
    click.option("--critic-init-gain" , "critic_init_gain" , type=float        , default=1.41  , help="Critic orthogonal initialization gain."                              )
)
