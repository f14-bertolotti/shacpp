import models
import torch

class LSTMWorld(models.Model):
    """  World Model Transformer. """

    def __init__(
        self, 
        observation_size : int,
        action_size      : int,
        agents           : int,
        steps            : int,
        layers           : int   = 3        ,
        hidden_size      : int   = 128      ,
        bidirectional    : bool  = True     ,
        dropout          : float = 0.0      ,
        device           : str   = "cuda:0" ,
        compute_reward   : bool  = True     ,
        compute_value    : bool  = True     ,
    ):
        super().__init__(observation_size, action_size, agents, steps)
        self.bidirectional = bidirectional
        self.compute_reward = compute_reward
        self.compute_value  = compute_value

        self.obs2hid = torch.nn.Linear(observation_size, hidden_size, device = device)
        self.act2hid = torch.nn.Linear(action_size     , hidden_size, device = device)
        self.actpos = torch.nn.Parameter(torch.empty(1, steps, 1, hidden_size, device = device).normal_(0,0.02))
        self.agnpos = torch.nn.Parameter(torch.empty(1, 1, agents, hidden_size, device = device).normal_(0,0.02))

        self.ln = torch.nn.LayerNorm(hidden_size, device=device)

        self.encoder = torch.nn.LSTM(
            input_size    = hidden_size, 
            hidden_size   = hidden_size, 
            num_layers    = layers,
            batch_first   = True,
            dropout       = dropout,
            bidirectional = bidirectional,
        ).to(device)

        if self.compute_reward: self.hid2rew = torch.nn.Linear(hidden_size * (2 if self.bidirectional else 1), 1, device = device)
        if self.compute_value : self.hid2val = torch.nn.Linear(hidden_size * (2 if self.bidirectional else 1), 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size * (2 if self.bidirectional else 1), observation_size, device = device)

    def forward(self, obs, act):

        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)

        encoded = self.encoder(hidden.flatten(1,2))[0].view(hidden.size(0), hidden.size(1), hidden.size(2), hidden.size(3) * (2 if self.bidirectional else 1))

        return {
            "observations" : self.hid2obs(encoded),
            "rewards"      : self.hid2rew(encoded)[:,1:].squeeze(-1) if self.compute_reward else None,
            "values"       : self.hid2val(encoded)[:,1:].squeeze(-1) if self.compute_value  else None
        }
