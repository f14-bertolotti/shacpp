from MultiLinear import MultiLinear
import torch

#class Reward(torch.nn.Module):
#    def __init__(self, observation_size, action_size, agents, layers = 1, hidden_size=128, activation="Tanh", dropout=0.1, device="cuda:0", shared=False):
#        super().__init__()
#        if type(shared) is not list: shared = [shared] * (layers + 2)
#        assert len(shared) == layers + 2
#
#        self.agents = agents
#        self.first_layer   = torch.nn.Linear((observation_size+action_size)*agents, hidden_size, device=device) if shared[0] else \
#                             MultiLinear(agents, observation_size*agents, hidden_size, device=device)
#        self.first_act     = getattr(torch.nn, activation)()
#        self.first_drop    = torch.nn.Dropout(dropout)
#        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)
#
#        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) if shared[i+1] else \
#                                                  MultiLinear(agents, hidden_size, hidden_size) for i in  range(layers)])
#        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
#        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
#        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])
#
#        self.last_layer    = torch.nn.Linear(hidden_size, 1, bias=False, device=device) if shared[-1] else \
#                             MultiLinear(channels=agents, input_size=hidden_size, output_size=1, bias=False, device=device)
#
#
#    def forward(self, obs, act):
#        obs = obs.flatten(1,2).unsqueeze(1).repeat(1,obs.size(1),1)
#        act = act.flatten(1,2).unsqueeze(1).repeat(1,act.size(1),1)
#
#        src = torch.cat([obs, act], dim=-1)
#        hidden = self.first_drop(self.first_act(self.first_layer(src)))
#        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
#            hidden = ln(hidden + drop(act(layer(hidden))))
#        value = self.last_layer(hidden).squeeze(-1)
#        return value
#
class Reward(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, layers = 1, hidden_size=128, activation="Tanh", dropout=0.1, device="cuda:0", shared=False):
        super().__init__()
        if type(shared) is not list: shared = [shared] * (layers + 2)
        assert len(shared) == layers + 2

        self.agents = agents
        self.first_layer   = torch.nn.Linear((observation_size+action_size), hidden_size, device=device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_layer    = torch.nn.Linear(hidden_size, 1, bias=False, device=device)


    def forward(self, obs, act):

        src = torch.cat([obs, act], dim=-1)
        hidden = self.first_drop(self.first_act(self.first_layer(src)))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        value = self.last_layer(hidden).squeeze(-1)
        return value

