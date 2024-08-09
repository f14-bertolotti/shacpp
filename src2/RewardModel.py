from MultiLinear import MultiLinear
from utils import layer_init
from utils import inverse_permutation
import torch

class RewardModel(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, layers = 1, hidden_size=128, dropout=0.1, activation="Tanh", device="cuda:0"):
        super().__init__()

        self.first_layer   = torch.nn.Linear((observation_size + action_size), hidden_size, device   = device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device = device) for _ in range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])
        
        self.last_layer    = torch.nn.Linear(hidden_size, 1, device = device)

    def forward(self, obs, act):
        src = torch.cat([obs, act], dim=-1)
        hidden = self.first_drop(self.first_act(self.first_layer(src)))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = drop(act(layer(hidden)))
        value = self.last_layer(hidden)
        return value

class RewardModel2(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, layers = 1, hidden_size=128, dropout=0.1, activation="Tanh", device="cuda:0"):
        super().__init__()

        self.first_layer   = torch.nn.Linear((observation_size + action_size), hidden_size, device   = device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device = device) for _ in range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])
        
        self.last_layer    = torch.nn.Linear(hidden_size, 1, device = device)

    def forward(self, obs, act):
        src = torch.cat([obs, act], dim=-1)
        hidden = self.first_drop(self.first_act(self.first_layer(src)))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        value = self.last_layer(hidden)
        return value

class RewardModel3(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, layers = 1, hidden_size=128, dropout=0.1, activation="Tanh", device="cuda:0"):
        super().__init__()
        self.agents = agents

        self.first_layer   = torch.nn.Linear((observation_size + action_size)*(agents), hidden_size, device = device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device = device) for _ in range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])
        
        self.last_layer    =  MultiLinear(agents, hidden_size, 1, bias=False, device = device)
        self.idxs = [j for i in range(agents) for j in range(agents) if i != j]

    def forward(self, obs, act):

        detached_obs = obs.detach()[:,self.idxs].view(obs.size(0),self.agents, -1)
        detached_act = act.detach()[:,self.idxs].view(act.size(0),self.agents, -1)

        src = torch.cat([obs, act, detached_obs, detached_act], dim=-1)
        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(src))))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        value = self.last_layer(hidden)

        return value

