from models.policies.Policy import Policy
import torch
import utils

class MLPAFO(Policy):
    """ 
        MLP Policy. 
        The MLP see all observations from all agents.
        The MLP outputs actions for all agents.
    """

    def __init__(
        self, 
        observation_size : int              ,
        action_size      : int              ,
        agents           : int              ,
        steps            : int              ,
        action_space     : list[float]      ,
        hidden_size      : int   = 128      ,
        layers           : int   = 1        ,
        dropout          : float = 0.0      ,
        var              : float = 1.0      ,
        activation       : str   = "Tanh"   ,
        device           : str   = "cuda:0"
    ):

        super().__init__(
            observation_size = observation_size ,
            action_size      = action_size      ,
            agents           = agents           ,
            steps            = steps            ,
            var              = var              ,
            action_space     = action_space     ,
            device           = device
        )

        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_act    = torch.nn.Tanh()

        self.first_layer = torch.nn.Linear(observation_size*agents, hidden_size, device=device)
        self.last_layer  = torch.nn.Linear(hidden_size, agents*action_size, bias=False, device=device)
        self.last_layer  = utils.layer_init(self.last_layer, 1.141)

    def forward(self, observations):
        observations = observations.flatten(-2,-1)

        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(observations))))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        actions = self.last_act(logits:=self.last_layer(hidden))

        return  actions.view(-1, self.agents, self.actions_size)

class MLPOFA(Policy):
    """ 
        MLP Policy.
        One MLP shared between all agents.
        The MLP see only the observations from its agent.
        The MLP outputs actions only for its agents.
    """

    def __init__(
        self, 
        observation_size : int              ,
        action_size      : int              ,
        agents           : int              ,
        steps            : int              ,
        hidden_size      : int   = 128      ,
        layers           : int   = 1        ,
        dropout          : float = 0.0      ,
        var              : float = 1.0      ,
        activation       : str   = "Tanh"   ,
        device           : str   = "cuda:0"
    ):

        super().__init__(observation_size, action_size, agents, steps, var, device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_act    = torch.nn.Tanh()

        self.first_layer = torch.nn.Linear(observation_size, hidden_size, device=device)
        self.last_layer  = torch.nn.Linear(hidden_size, action_size, bias=False, device=device)
        self.last_layer  = utils.layer_init(self.last_layer, 1.141)

    def forward(self, observations):
        hidden = self.first_drop(self.first_act(self.first_norm(self.first_layer(observations))))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = hidden + drop(act(layer(ln(hidden))))
        actions = self.last_act(logits:=self.last_layer(hidden))

        return actions.view(-1, self.agents, self.actions_size)


