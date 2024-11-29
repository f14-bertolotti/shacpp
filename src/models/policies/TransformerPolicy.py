from models.policies import Policy
import torch

class TransformerPolicy(Policy):
    """ 
        Policy that is permutation invariant wrt. the agents. 
        It uses a transformer architecture to encode the observations.
        A linear layer to map the encoded observations to the actions.
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
        var              : float = 1.0      ,
        device           : str   = "cuda:0" ,
    ):
        super().__init__(observation_size, action_size, agents, steps, var)
        activation = {"ReLU":"relu", "GELU":"gelu"}[activation]

        self.first_layer = torch.nn.Linear(observation_size, hidden_size, device=device)
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

        self.hid2act  = torch.nn.Linear(hidden_size, action_size, device = device)

    def forward(self, observations):
        hidden  = self.first_drop(self.first_norm(self.first_layer(observations)))
        encoded = self.encoder(hidden)
        logits  = self.hid2act(encoded)

        return logits.view(-1, self.agents, self.actions_size)
