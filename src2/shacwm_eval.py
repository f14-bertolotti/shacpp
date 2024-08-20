from ActorModel import *
from Dispersion import Dispersion as DispersionScenario
import numpy, torch, vmas
from moviepy.editor import ImageSequenceClip

agents = 10
envs   = 1
device = "cuda:0"
steps = 64

world = vmas.simulator.environment.Environment(
    DispersionScenario(
        device = device,
        radius = .05,
        agents = agents,
    ),
    n_agents           = agents,
    num_envs           = envs  ,
    device             = device,
    shared_reward      = False ,
    grad_enabled       = False  ,
    continuous_actions = True  ,
    dict_spaces        = False ,
    seed               = None  ,
)

observation_size:int = numpy.prod(world.get_observation_space()[0].shape)
action_size     :int = numpy.prod(world.get_action_space()[0].shape)

actor_model  = ActorModel(observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device)
actor_model.load_state_dict(torch.load("actor.pkl")["actor_state_dict"])

frames, observation = [], torch.stack(world.reset()).transpose(0,1)
actions = torch.zeros((*observation.shape[:-1],2), device=device)

with torch.no_grad():
    actor_model.eval()
    for step in range(0, steps):
        actions = actor_model(observation, actions)
        frames.append(world.render(mode="rgb_array"))
        observation, reward, done, info = world.step(actions.transpose(0,1))
        print(done)
        observation = torch.stack(observation).transpose(0,1)


clip = ImageSequenceClip(frames, fps=30)
clip.write_gif("play.gif", fps=30)

