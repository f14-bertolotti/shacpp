import models
import torch

class TransformerWorld(models.Model):
    """  World Model Transformer. """

    def __init__(
        self, 
        observation_size : int,
        action_size      : int,
        agents           : int,
        steps            : int,
        layers           : int   = 3        ,
        hidden_size      : int   = 128      ,
        heads            : int   = 2        ,
        feedforward_size : int   = 512      ,
        dropout          : float = 0.0      ,
        activation       : str   = "ReLU"   ,
        device           : str   = "cuda:0" ,
        compute_reward   : bool  = True     ,
        compute_value    : bool  = True     ,
    ):
        super().__init__(observation_size, action_size, agents, steps)
        activation = {"ReLU":"relu", "GELU":"gelu"}[activation]
        self.compute_reward = compute_reward
        self.compute_value  = compute_value

        self.obs2hid = torch.nn.Linear(observation_size, hidden_size, device = device)
        self.act2hid = torch.nn.Linear(action_size     , hidden_size, device = device)
        self.actpos = torch.nn.Parameter(torch.empty(1, steps, 1, hidden_size, device = device).normal_(0,0.02))
        self.agnpos = torch.nn.Parameter(torch.empty(1, 1, agents, hidden_size, device = device).normal_(0,0.02))

        self.ln = torch.nn.LayerNorm(hidden_size, device=device)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                dim_feedforward = feedforward_size ,
                d_model         = hidden_size      ,
                activation      = activation       ,
                device          = device           ,
                nhead           = heads            ,
                dropout         = dropout          ,
                batch_first     = True
            ), 
            num_layers           = layers,
            enable_nested_tensor = False
        )

        if self.compute_reward: self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        if self.compute_value : self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)

        self.mask = torch.tensor([[0 if i1 <= i2 else 1 for i1 in range(steps+1) for j1 in range(agents)] for i2 in range(steps+1) for j2 in range(agents)], dtype=torch.float)
        self.mask[self.mask == 1] = float("-inf")
        self.mask = self.mask.to(device)

    def forward(self, obs, act):

        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)

        encoded = self.encoder(hidden.flatten(1,2), mask=self.mask).view(hidden.shape)

        return {
            "observations" : self.hid2obs(encoded),
            "rewards"      : self.hid2rew(encoded)[:,1:].squeeze(-1) if self.compute_reward else None,
            "values"       : self.hid2val(encoded)[:,1:].squeeze(-1) if self.compute_value  else None,
        }
