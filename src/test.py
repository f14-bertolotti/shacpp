import numpy, torch
import agents, environments
import matplotlib.pyplot as plt
import matplotlib.animation as animation

envs=1
seed=1
device="cuda:0"
steps=1024

agent       = agents.TransformerAgent().to(device)
environment = environments.Scattered(envs=envs, agents=9, device=device)
agent.load_state_dict(torch.load("models/ppo/agent.pkl")["agentsd"])

obss, rews = [], []
obs = environment.reset()
for step in range(0, steps):

   with torch.no_grad():
       result = agent.get_action(obs)

   res = environment.step(obs,result["actions"])
   obs,rew = res["next_observations"], res["rewards"]
   obss.append(obs.cpu().squeeze(0).detach())
   rews.append(rew.cpu().squeeze(0).detach())

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

ani = animation.FuncAnimation(fig=fig, func=update, frames=obss.size(0), interval=60)
plt.show()
