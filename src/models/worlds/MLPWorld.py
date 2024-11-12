import models
import torch

class MLPWorld(models.Model):
    """ World Model MLP. """
    def __init__(
        self,
        observation_size : int              ,
        action_size      : int              ,
        agents           : int              ,
        steps            : int              ,
        layers           : int   = 3        ,
        hidden_size      : int   = 128      ,
        dropout          : float = 0.0      ,
        activation       : str   = "ReLU"   ,
        device           : str   = "cuda:0" ,
        compute_reward   : bool  = True     ,
        compute_value    : bool  = True     ,
    ):
        super().__init__(observation_size, action_size, agents, steps)
        self.compute_reward = compute_reward
        self.compute_value  = compute_value

        self.first_layer = torch.nn.Linear(agents * (observation_size + (action_size) * steps), hidden_size, device=device)
        self.first_norm = torch.nn.LayerNorm(hidden_size, device=device)
        self.first_act  = getattr(torch.nn, activation)()

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        if self.compute_reward: self.hid2rew = torch.nn.Linear(hidden_size, agents * steps * 1, device=device)
        if self.compute_value : self.hid2val = torch.nn.Linear(hidden_size, agents * steps * 1, device=device)
        self.hid2obs = torch.nn.Linear(hidden_size, agents * (steps+1) * observation_size, device=device)

    def forward(self, obs, act):
        hidden = self.first_norm(self.first_act(self.first_layer(torch.cat([obs[:,[0]].flatten(1,3), act.flatten(1,3)], dim=1))))

        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))

        return {
        "observations" : self.hid2obs(hidden).view(hidden.size(0), self.steps+1, self.agents, self.observation_size),
        "rewards"      : self.hid2rew(hidden).view(hidden.size(0), self.steps, self.agents) if self.compute_reward else None,
        "values"       : self.hid2val(hidden).view(hidden.size(0), self.steps, self.agents) if self.compute_value  else None
        }


