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
        device           : str   = "cuda:0"
    ):
        super().__init__(observation_size, action_size, agents, steps)
        activation = {"ReLU":"relu", "GELU":"gelu"}[activation]

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

        self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)

    def forward(self, obs, act):

        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)

        encoded = self.encoder(hidden.flatten(1,2)).view(hidden.shape)

        rew = self.hid2rew(encoded)[:,1:].squeeze(-1)
        val = self.hid2val(encoded)[:,1:].squeeze(-1)
        obs = self.hid2obs(encoded)

        return rew, val, obs


