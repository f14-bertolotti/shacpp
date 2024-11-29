import models
import torch

class MLPReward(models.Model):
    """ base MLP Reward """
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

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])


class MLPRewardAFO(MLPReward):
    """ 
        Reward MLP.
        The MLP sees all observations and all actions from all agents.
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

        self.first_layer = torch.nn.Linear((observation_size*2+action_size)*agents, hidden_size, device=device)
        self.last_layer  = torch.nn.Linear(hidden_size, agents, bias=False, device=device)

    def forward(self, prev_obs, act, next_obs):
        src = torch.cat([prev_obs, act, next_obs], dim=-1).flatten(-2,-1)

        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(src))))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        
        return self.last_layer(hidden)

class MLPRewardOFA(MLPReward):
    """ 
        MLP Reward.
        One MLP for each agents.
        MLP params shared by all agents.
        The MLP sees only the observations and actions from its agent.
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
        
        self.first_layer   = torch.nn.Linear((observation_size*2+action_size), hidden_size, device=device)
        self.last_layer    = torch.nn.Linear(hidden_size, 1, bias=False, device=device)

    def forward(self, prev_obs, act, next_obs):
        src = torch.cat([prev_obs, act, next_obs], dim=-1)

        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(src))))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))

        return self.last_layer(hidden).squeeze(-1)

