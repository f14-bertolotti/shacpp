import models
import torch

class TransformerReward(models.Model):
    """  
        Reward Model that is permutation invariant wrt agents.
        It uses a transformer architecture to encode the observations.
        A linear layer to map the encoded observations to the rewards.
        It ignores the resulting observations from applying the actions to the previous observations.
    """

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
    ):
        super().__init__(observation_size, action_size, agents, steps)
        activation = {"ReLU":"relu", "GELU":"gelu"}[activation]

        self.obs2hid     = torch.nn.Linear(observation_size, hidden_size, device=device)
        self.act2hid     = torch.nn.Linear(action_size     , hidden_size, device=device)
        self.first_norm  = torch.nn.LayerNorm(hidden_size, device=device)
        self.first_drop  = torch.nn.Dropout(dropout)

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

    def get_src_mask(self, agents, device="cpu", dtype=torch.float32):
        mask = torch.full((agents*2, agents*2), float('-inf'))
        for i in range(0,agents):
            for j in range(agents):
                mask[i,j] = 0
        for i in range(agents,agents*2):
            for j in range(agents,agents*2):
                mask[i,j] = 0
        for i in range(0,agents):
            mask[i,i+agents] = 0
        for i in range(agents,agents*2):
            mask[i,i-agents] = 0


    def forward(self, prev_obs, act, next_obs):
        hidobs = self.obs2hid(prev_obs)
        hidact = self.act2hid(act)
        hidden = self.first_drop(self.first_norm(hidobs+hidact))
        encoded = self.encoder(hidden)
        rewards = self.hid2rew(encoded).squeeze(-1)
        return rewards

