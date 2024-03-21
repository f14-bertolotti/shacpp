from Engine import Engine

import random
import torch
import tqdm
torch.set_float32_matmul_precision('high')

epochs = 1000000
horizon = 20
batch_size = 2048
agents = 25
device="cuda:0"
compile=False
raw_engine = Engine(batch_size=batch_size, agents=agents, device=device,trace=False)
optimizer = torch.optim.Adam(raw_engine.parameters(), lr=0.0001) 
engine = torch.compile(raw_engine) if compile else raw_engine

pos = torch.rand ((batch_size, agents, 2), device=device, requires_grad=True)*10-5
for epoch in (bar := tqdm.tqdm(range(epochs))):

    optimizer.zero_grad()
    result = engine(pos,iterations=horizon)
    losses = engine.lossfn(result) 
    sum(losses).backward()
    optimizer.step()

    pos = pos.detach()
    
    restart_mask = torch.rand((batch_size,)) < 0.05
    pos[restart_mask] = torch.rand ((agents, 2), device=device, requires_grad=True)*10-5

    bar.set_description(" ".join(f"{l.item():5.3f}" for l in losses))

    if epoch % 100 == 0: torch.save({"modelsd":raw_engine.state_dict()}, "engine.pkl")

