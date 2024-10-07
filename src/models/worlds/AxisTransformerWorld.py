import models
import torch

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size, feedforward_size, dropout, device):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_size, feedforward_size, device=device)
        self.lin2 = torch.nn.Linear(feedforward_size, hidden_size, device=device)
        self.lnrm = torch.nn.LayerNorm(hidden_size, device=device)
        self.drop = torch.nn.Dropout(dropout)
    def forward(self, x):
        return self.lnrm(x + self.lin2(self.drop(torch.nn.functional.relu(self.lin1(x)))))

class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, device):
        super().__init__()
        self.qlin = torch.nn.Linear(hidden_size, hidden_size, device=device)
        self.klin = torch.nn.Linear(hidden_size, hidden_size, device=device)
        self.vlin = torch.nn.Linear(hidden_size, hidden_size, device=device)
    def forward(self, x):
        q = self.qlin(x)
        k = self.klin(x)
        v = self.vlin(x)
        s = q @ k.transpose(-2,-1) / (q.shape[-1]**0.5)
        a = torch.nn.functional.softmax(s, dim=-1)
        return a @ v

class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, feedforward_size, dropout, device):
        super().__init__()
        self.attn = SelfAttention(hidden_size, device)
        self.ffwd = FeedForward(hidden_size, feedforward_size, dropout, device)
    def __call__(self, x):
        return self.ffwd(self.attn(x))

class AxisTransformerWorld(models.Model):
    """ World Model AxisTransformer. """
    def __init__(
        self, 
        observation_size : int,
        action_size      : int,
        agents           : int,
        steps            : int,
        layers           : int   = 3        ,
        dropout          : float = 0.0      ,
        hidden_size      : int   = 128      ,
        feedforward_size : int   = 512      ,
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

        self.layers = torch.nn.ModuleList([
            TransformerLayer(
                hidden_size = hidden_size,
                feedforward_size = feedforward_size,
                dropout = dropout,
                device = device
            ) for _ in range(layers*2)
        ])

        self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)

    def forward(self, obs, act):
        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos
        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)

        for layer in self.layers:
            hidden = layer(hidden)
            hidden = hidden.transpose(1,2)

        rew = self.hid2rew(hidden)[:,1:].squeeze(-1)
        val = self.hid2val(hidden)[:,1:].squeeze(-1)
        obs = self.hid2obs(hidden)
        return rew, val, obs


