import utils, click

transformer = utils.decochain(
    click.option("--layers"           , "layers"           , type=int          , default=3     , help="Number of layers."                                            ),
    click.option("--embedding-size"   , "embedding_size"   , type=int          , default=64    , help="Embedding layer size."                                        ),
    click.option("--feedforward-size" , "feedforward_size" , type=int          , default=256   , help="Feed Forward layer size."                                     ),
    click.option("--heads"            , "heads"            , type=int          , default=2     , help="Number of attention heads."                                   ),
    click.option("--activation"       , "activation"       , type=str          , default="gelu", help="Activation function."                                         ),
    click.option("--shared"           , "shared"           , type=bool         , default=True  , help="True if parameter should be shared between actor and critic." ),
    click.option("--state-dict-path"  , "state_dict_path"  , type=click.Path() , default=None  , help="Path from where the parameters should be loaded from."        ),
    click.option("--compile"          , "compile"          , type=bool         , default=True  , help="True if the model should be compiled before being returned."  )
)
