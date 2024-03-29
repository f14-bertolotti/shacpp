from shac2 import Actor, Critic, Environment 
import numpy, torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from RunningMeanStd import RunningMeanStd
import pickle

envs=2
seed=1
device="cuda:0"
steps=1024

actor       = Actor(actions=2,observations=10*2).to(device)
environment = Environment(envs=envs, actor=actor, agents=9, device=device, deterministic=True)
actor.load_state_dict(torch.load("actor.pkl")["actor"])

obss, rews = [], []
obs = environment.init_state()

obss.append(obs[[0]].clone().cpu().squeeze(0).detach())

for step in range(0, steps):

    with torch.no_grad():
        action = actor(torch.cat([obs,obs[:,[0]]],dim=1), deterministic=True)
    
    obs,rew = environment.step(obs,action)

    obss.append(obs[[0]].clone().cpu().squeeze(0).detach())
    rews.append(rew[[0]].clone().cpu().squeeze(0).detach())



obss = torch.stack(obss)
rews = torch.stack(rews)

fig, ax = plt.subplots()
scatterplot = ax.scatter(
    x = environment.points[:,0].cpu().numpy(),
    y = environment.points[:,1].cpu().numpy(),
    color = "red"
)
scatterplot = ax.scatter(
    x = obss[0,:,0].numpy(),
    y = obss[0,:,1].numpy(),
    color = "black"
)

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)

def update(frame):
    x = obss[frame,:,0]
    y = obss[frame,:,1]
    data = numpy.stack([x, y]).T
    scatterplot.set_offsets(data)
    return scatterplot

ani = animation.FuncAnimation(fig=fig, func=update, frames=obss.size(0), interval=120)
plt.show()
