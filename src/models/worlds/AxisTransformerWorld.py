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
    def forward(self, x, mask):
        q = self.qlin(x)
        k = self.klin(x)
        v = self.vlin(x)
        s = q @ k.transpose(-2,-1) / (q.shape[-1]**0.5)
        a = torch.nn.functional.softmax(s + (0 if mask is None else mask), dim=-1)
        return a @ v

class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, feedforward_size, dropout, device):
        super().__init__()
        self.attn = SelfAttention(hidden_size, device)
        self.ffwd = FeedForward(hidden_size, feedforward_size, dropout, device)
    def __call__(self, x, mask=None):
        return self.ffwd(self.attn(x,mask=mask))

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

        self.layers = torch.nn.ModuleList([
            TransformerLayer(
                hidden_size      = hidden_size,
                feedforward_size = feedforward_size,
                dropout          = dropout,
                device           = device
            ) for _ in range(layers*2)
        ])

        if self.compute_reward: self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        if self.compute_value : self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)
        self.step_mask = torch.nn.Transformer.generate_square_subsequent_mask(steps+1, device=device)

    def forward(self, obs, act):
        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos
        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)

        for i,layer in enumerate(self.layers):
            hidden = layer(hidden, mask=self.step_mask if i%2==1 else None )
            hidden = hidden.transpose(1,2)

        return {
            "observations" : self.hid2obs(hidden),
            "rewards"      : self.hid2rew(hidden)[:,1:].squeeze(-1) if self.compute_reward else None,
            "values"       : self.hid2val(hidden)[:,1:].squeeze(-1) if self.compute_value  else None
        } 


