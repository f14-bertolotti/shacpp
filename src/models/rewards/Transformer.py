import models
import torch

class Transformer(models.Model):
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

        self.src2hid     = torch.nn.Linear(action_size + observation_size, hidden_size, device=device)
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

    def forward(self, prev_obs, act, next_obs):
        src = torch.cat([prev_obs, act], -1) 
        hidden = self.first_drop(self.first_norm(self.src2hid(src)))
        encoded = self.encoder(hidden)
        rewards = self.hid2rew(encoded).squeeze(-1)
        return rewards

