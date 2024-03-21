from ppo import seed_everything, Agent, Storage, Environment 
import numpy, torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

envs=1
seed=1
device="cuda:0"
steps=1000

seed_everything(seed)
agent       = Agent().to(device)
storage     = Storage(envs=envs, steps=steps, device="cpu")
environment = Environment(envs=envs, device=device)
agent.load_state_dict(torch.load("agent.pkl")["agentsd"])


for step in range(0, steps):

   with torch.no_grad():
       result = agent.get_action_and_value(environment.observation)

   storage[step] = {
       "observation" : environment.observation.cpu(),
       "action"      : result["action"].cpu(),
       "logprob"     : result["logprob"].cpu(),
       "value"       : result["value"].cpu(),
       "entropy"     : result["entropy"].cpu()
   }
   environment.step(result["action"])
   storage[step] = {"reward" : environment.reward().cpu()}


fig, ax = plt.subplots()
scatterplot = ax.scatter(
    x = [storage.observations[0,0,0].numpy()],
    y = [storage.observations[0,0,1].numpy()],
    color = "black"
)

ax.set_xlim(-30,30)
ax.set_ylim(-30,30)

def update(frame):
    x = [storage.observations[0,frame,0]]
    y = [storage.observations[0,frame,1]]
    data = numpy.stack([x, y]).T
    scatterplot.set_offsets(data)
    return scatterplot

print(storage.rewards)
ani = animation.FuncAnimation(fig=fig, func=update, frames=storage.observations.size(1), interval=10)
plt.show()
