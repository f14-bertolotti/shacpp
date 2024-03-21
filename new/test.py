from train import Policy, Environment

import torch

device = "cuda:0"
modelsd = torch.load("engine.pkl")["modelsd"]
policy = Policy(agents=8, device=device)
policy.load_state_dict(modelsd)
environment = Environment(policy, batch=1, agents=8, device="cuda:0")

trace = [environment.step()["state"].cpu() for i in range(20)]
print(environment.reward())

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig, ax = plt.subplots()
ax.scatter(
    x = [p[0].item() for p in environment.points],
    y = [p[1].item() for p in environment.points],
    color = "red"
)
scatterplot = ax.scatter(
    x = trace[0][0,:,0],
    y = trace[0][0,:,1],
    color = "black"
)
for point in environment.points:
    ax.add_patch(plt.Circle((point[0],point[1]), 3,alpha=0.5))

ax.set_xlim(-30,30)
ax.set_ylim(-30,30)

def update(frame):
    x = trace[frame][0,:,0]
    y = trace[frame][0,:,1]
    data = np.stack([x, y]).T
    scatterplot.set_offsets(data)
    return scatterplot


ani = animation.FuncAnimation(fig=fig, func=update, frames=len(trace), interval=100)
plt.show()
