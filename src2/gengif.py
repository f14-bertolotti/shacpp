from moviepy.editor import ImageSequenceClip
from Dispersion import Dispersion as DispersionScenario
import models, click, utils, torch, vmas

@click.command()
@click.option("--seed"              , "seed"              , type=int          , default=42       , help="random seed"                                )
@click.option("--observation-size"  , "observation_size"  , type=int          , default=2        , help="observation size"                           )
@click.option("--action-size"       , "action_size"       , type=int          , default=11       , help="action size"                                )
@click.option("--agents"            , "agents"            , type=int          , default=5        , help="number of agents"                           )
@click.option("--steps"             , "steps"             , type=int          , default=64       , help="number of steps for the evaluation rollout" )
@click.option("--device"            , "device"            , type=str          , default="cuda:0" , help="device"                                     )
@click.option("--output-path"       , "output_path"       , type=click.Path() , default="x.gif"  , help="output path"                                )
@click.option("--input-path"        , "input_path"        , type=click.Path() , default=None     , help="input model path"                           )
@click.option("--compile"           , "compile"           , type=bool         , default=False    , help="compile the model"                          )
def run(
        seed,
        agents,
        observation_size,
        action_size,
        steps,
        device,
        output_path,
        input_path,
        compile,
    ):
    utils.seed_everything(seed)

    world = vmas.simulator.environment.Environment(
        DispersionScenario(
            device = device,
            radius = .05,
            agents = agents,
        ),
        n_agents           = agents ,
        num_envs           = 1      ,
        device             = device ,
        shared_reward      = False  ,
        grad_enabled       = False  ,
        continuous_actions = True   ,
        dict_spaces        = False  ,
        seed               = None   ,
    )
    
    policy = models.Policy(observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 2048, dropout=0.0, activation="Tanh", device = device, shared=[True, True, False])

    if compile: policy = torch.compile(policy)
    if input_path: policy.load_state_dict(torch.load(input_path)["policy_state_dict"])
    
    frames, observation = [], torch.stack(world.reset()).transpose(0,1)
    actions = torch.zeros((*observation.shape[:-1],2), device=device)
    
    with torch.no_grad():
        policy.eval()
        for step in range(0, steps):
            actions = policy(observation)["actions"]
            frames.append(world.render(mode="rgb_array"))
            observation, reward, done, info = world.step(actions.transpose(0,1))
            observation = torch.stack(observation).transpose(0,1)
    
    
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_gif(output_path, fps=30)
    
if __name__ == "__main__":
    run()
