from Engine import Engine

import torch

device = "cuda:0"
modelsd = torch.load("engine.pkl")["modelsd"]
engine = Engine(batch_size=1, agents=25, device=device)
engine.load_state_dict(modelsd)
result = engine(
        torch. rand((1, 25, 2), device=device, requires_grad=False)*10-5, 
        1000
    )

losses = engine.lossfn(result)
print(" ".join(f"{l.item():5.3f}" for l in losses))

trace = [{"pos":pos.detach().cpu().numpy()} for pos in result["trace"]]

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig, ax = plt.subplots()
ax.scatter(
    x = [p[0].item() for p in engine.points],
    y = [p[1].item() for p in engine.points],
    color = "red"
)
scatterplot = ax.scatter(
    x     = trace[0]["pos"][0,:,0],
    y     = trace[0]["pos"][0,:,1],
    color = "black"
)

ax.set_xlim(-30,30)
ax.set_ylim(-30,30)

def update(frame):
    x = trace[frame]["pos"][0,:,0]
    y = trace[frame]["pos"][0,:,1]
    data = np.stack([x, y]).T
    scatterplot.set_offsets(data)
    return scatterplot


ani = animation.FuncAnimation(fig=fig, func=update, frames=len(trace), interval=10)
plt.show()
