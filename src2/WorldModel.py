from MultiLinear import MultiLinear
import copy, torch

class WorldModel(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, layers=3, hidden_size=128, heads=2, feedforward_size=512, dropout=0.1, max_positions=512, activation="ReLU", device="cuda:0"):
        super().__init__()

        self.agents = agents

        self.hidden_first = torch.nn.Linear((observation_size+action_size)*agents, hidden_size, device = device)

        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])
        self.hidden_lins   = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) for _ in range(layers)])

        self.hidden_last = torch.nn.Linear(hidden_size, observation_size*agents, device = device)

    def step(self, obs, act):
        oba = torch.cat([obs, act], dim=-1)
        hidden = self.hidden_first(oba.flatten(1,2))
        for act, drp, nrm, lin in zip(self.hidden_acts, self.hidden_drops, self.hidden_norms, self.hidden_lins):
            hidden = nrm(hidden + drp(act(lin(hidden))))

        prd_obs = self.hidden_last(hidden).view(*obs.shape)
        return prd_obs + obs


