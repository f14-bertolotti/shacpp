from MultiLinear import MultiLinear
import torch

class Value(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, layers = 1, hidden_size=128, activation="Tanh", dropout=0.1, device="cuda:0"):
        super().__init__()

        self.agents = agents
        self.first_layer   = torch.nn.Linear(observation_size*agents, hidden_size, device=device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) for _ in range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_layer    = torch.nn.Linear(hidden_size, self.agents, bias=False, device=device)
        self.last_act    = torch.nn.Tanh()

    #@torch.no_grad()
    #def share(self, first_layer=None, first_norm=None, hidden_layer=None, hidden_norms=None, last_layer=None):
    #    if first_layer is not None: self.first_layer.weight = first_layer.weight
    #    if first_norm  is not None: self.first_norm.weight = first_norm.weight

    #    if hidden_layer is not None:
    #        for layer, norm, other_layer, other_norm in zip(self.hidden_layers, self.hidden_norms, hidden_layer, hidden_norms):
    #            layer.weight = other_layer.weight
    #            norm.weight = other_norm.weight
    #            
    #    if last_layer is not None: self.last_layer.weight = last_layer.weight

    def forward(self, observations):
        observations = observations.flatten(1,2)#.unsqueeze(1).repeat(1,observations.size(1),1)

        #observations = torch.stack([torch.stack([observations[:,i,:] if i==j else observations[:,i,:].detach() for i in range(self.agents)],dim=1).flatten(-2,-1) for j in range(self.agents)],dim=-2)

        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(observations))))
        for layer, act, drop, norm in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = norm(hidden + drop(act(layer(hidden))))
        value = self.last_layer(hidden)
        return value


