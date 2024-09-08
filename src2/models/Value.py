import torch

class Value(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, layers = 1, hidden_size=128, activation="Tanh", dropout=0.1, device="cuda:0"):
        super().__init__()
        self.agents = agents
        self.first_layer   = torch.nn.Linear(observation_size*agents, hidden_size, device = device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)
        
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device = device) for _ in range(layers)])
        self.hidden_act    = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drop   = torch.nn.ModuleList([torch.nn.Dropout(dropout)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_layer    = torch.nn.Linear(hidden_size, agents, bias=False, device = device)

    def forward(self, observations):
        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(observations.flatten(-2,-1)))))
        for layer, act, drop, norm in zip(self.hidden_layers, self.hidden_act, self.hidden_drop, self.hidden_norms):
            hidden = norm(hidden + drop(act(layer(hidden))))
        value = self.last_layer(hidden)
        return value


