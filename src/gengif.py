from moviepy import ImageSequenceClip
import environments
import models 
import click
import utils
import torch
import vmas

@click.command()
@click.option("--seed"             , "seed"             , type=int          , default=42           , help="random seed"                                )
@click.option("--observation-size" , "observation_size" , type=int          , default=2            , help="observation size"                           )
@click.option("--action-size"      , "action_size"      , type=int          , default=11           , help="action size"                                )
@click.option("--action-space"     , "action_space"     , type=(float,float), default=(-1,1)       , help="action space"                               )
@click.option("--env-name"         , "env_name"         , type=str          , default="dispersion" , help="environment name"                           )
@click.option("--agents"           , "agents"           , type=int          , default=5            , help="number of agents"                           )
@click.option("--steps"            , "steps"            , type=int          , default=64           , help="number of steps for the evaluation rollout" )
@click.option("--device"           , "device"           , type=str          , default="cuda:0"     , help="device"                                     )
@click.option("--output-path"      , "output_path"      , type=click.Path() , default="x.gif"      , help="output path"                                )
@click.option("--input-path"       , "input_path"       , type=click.Path() , default=None         , help="input model path"                           )
@click.option("--compile"          , "compile"          , type=bool         , default=False        , help="compile the model"                          )
def run(
        seed,
        agents,
        observation_size,
        action_size,
        action_space,
        steps,
        device,
        output_path,
        input_path,
        env_name,
        compile,
    ):
    utils.seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    torch.set_printoptions(precision=3, sci_mode=False)
    
    world = environments.get_environment(
        name         = env_name,
        envs         = 1       ,
        agents       = agents  ,
        device       = device  ,
        grad_enabled = False   ,
        seed         = seed    ,
    )
    
    policy = models.policies.Transformer(
        observation_size = observation_size ,
        action_size      = action_size      ,
        agents           = agents           ,
        steps            = steps            ,
        action_space     = action_space     ,
        layers           = 1                ,
        hidden_size      = 64               ,
        feedforward_size = 128              ,
        heads            = 1                ,
        dropout          = 0.0              ,
        activation       = "ReLU"           ,
        device           = device
    )

    if compile: policy = torch.compile(policy)
    if input_path: 
        checkpoint = torch.load(input_path)
        print("restoring from:", input_path)
        print("checkpoint ep :", checkpoint["episode"])
        policy.load_state_dict({k.replace("_orig_mod.",""):v for k,v in checkpoint["policy_state_dict"].items()})
    
    frames, observation = [], torch.stack(world.reset()).transpose(0,1)
    
    
    with torch.no_grad():
        policy.eval()
        total_reward = 0
        for step in range(0, steps):
            actions = policy.act(observation)["actions"]
            frames.append(world.render(mode="rgb_array"))
            observation, reward, done, info = world.step(actions.transpose(0,1))
            print(world.scenario.max_rewards())
            observation = torch.stack(observation).transpose(0,1)
            total_reward += torch.stack(reward).sum().item()
            
        
        print("total reward", total_reward, world.scenario.max_rewards().sum().item())
    
    
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_gif(output_path, fps=30)
    
if __name__ == "__main__":
    run()
