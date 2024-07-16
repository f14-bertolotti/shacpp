import click, utils

trajectory = utils.decochain(
    click.option("--gamma"     , "gamma"     , type=float , default=.99 ),
    click.option("--gaelambda" , "gaelambda" , type=float , default=.95 ),
    click.option("--steps"     , "steps"     , type=int   , default=64  ),
    click.option("--utr"       , "utr"       , type=int   , default=1   )
)
