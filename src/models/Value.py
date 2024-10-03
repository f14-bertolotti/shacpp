import models
import torch

class Value(models.Model):
    """ base MLP Value """
    def __init__(
        self, 
        observation_size : int              ,
        action_size      : int              ,
        agents           : int              ,
        steps            : int              ,
        layers           : int   = 1        ,
        hidden_size      : int   = 128      ,
        dropout          : float = 0.0      ,
        activation       : str   = "Tanh"   ,
        device           : str   = "cuda:0"
    ):
        super().__init__(observation_size, action_size, agents, steps)

        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) for _ in range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])


class ValueAFO(Value):
    """ 
        Value MLP.
        The MLP sees all observations from all agents.
    """
    def __init__(
        self, 
        observation_size : int              ,
        action_size      : int              ,
        agents           : int              ,
        steps            : int              ,
        layers           : int   = 1        ,
        hidden_size      : int   = 128      ,
        dropout          : float = 0.0      ,
        activation       : str   = "Tanh"   ,
        device           : str   = "cuda:0"
    ):

        super().__init__(observation_size, action_size, agents, steps, layers, hidden_size, dropout, activation, device)

        self.first_layer   = torch.nn.Linear(observation_size*agents, hidden_size, device=device)
        self.last_layer    = torch.nn.Linear(hidden_size, self.agents, bias=False, device=device)

    def forward(self, observations):
        observations = observations.flatten(1,2)

        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(observations))))
        for layer, act, drop, norm in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = norm(hidden + drop(act(layer(hidden))))

        return self.last_layer(hidden)

class ValueOFA(Value):
    """ 
        MLP Value.
        One MLP for each agents.
        MLP params shared by all agents.
        The MLP sees only the observations from its agent.
    """
    def __init__(
        self, 
        observation_size : int              ,
        action_size      : int              ,
        agents           : int              ,
        steps            : int              ,
        layers           : int   = 1        ,
        hidden_size      : int   = 128      ,
        dropout          : float = 0.0      ,
        activation       : str   = "Tanh"   ,
        device           : str   = "cuda:0"
    ):

        super().__init__(observation_size, action_size, agents, steps, layers, hidden_size, dropout, activation, device)

        self.first_layer   = torch.nn.Linear(observation_size, hidden_size, device=device)
        self.last_layer    = torch.nn.Linear(hidden_size, 1, bias=False, device=device)

    def forward(self, observations):
        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(observations))))
        for layer, act, drop, norm in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = norm(hidden + drop(act(layer(hidden))))

        return self.last_layer(hidden).squeeze(-1)


