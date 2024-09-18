import torch

class World(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, steps, layers=3, hidden_size=128, heads=2, feedforward_size=512, dropout=0.1, activation="relu", device="cuda:0"):
        super().__init__()

        self.agents = agents
        self.steps  =  steps

        self.obs2hid = torch.nn.Linear(observation_size, hidden_size, device = device)
        self.act2hid = torch.nn.Linear(action_size     , hidden_size, device = device)
        self.actpos = torch.nn.Parameter(torch.empty(1, steps, 1, hidden_size, device = device).normal_(0,0.02))
        self.agnpos = torch.nn.Parameter(torch.empty(1, 1, agents, hidden_size, device = device).normal_(0,0.02))

        self.ln = torch.nn.LayerNorm(hidden_size, device=device)

        self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)


class WorldAFOLong(World):
    def __init__(self, observation_size, action_size, agents, steps, layers=3, hidden_size=128, heads=2, feedforward_size=512, dropout=0.1, activation="relu", device="cuda:0"):
        super().__init__(observation_size, action_size, agents, steps, layers, hidden_size, heads, feedforward_size, dropout, activation, device)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                dim_feedforward = feedforward_size ,
                d_model         = hidden_size      ,
                activation      = activation       ,
                device          = device           ,
                nhead           = heads            ,
                dropout         = dropout          ,
                batch_first     = True,
            ), 
            num_layers = layers,
            enable_nested_tensor = False if heads % 2 == 1 else True,
        )

    def forward(self, obs, act):

        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)

        encoded = self.encoder(hidden.flatten(1,2)).view(hidden.shape)

        rew = self.hid2rew(encoded)[:,1:].squeeze(-1)
        val = self.hid2val(encoded)[:,1:].squeeze(-1)
        return rew, val


class WorldAFOWide(World):
    def __init__(self, observation_size, action_size, agents, steps, layers=3, hidden_size=128, heads=2, feedforward_size=512, dropout=0.1, activation="relu", device="cuda:0"):
        super().__init__(observation_size, action_size, agents, steps, layers, hidden_size, heads, feedforward_size, dropout, activation, device)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                dim_feedforward = feedforward_size ,
                d_model         = hidden_size * agents,
                activation      = activation       ,
                device          = device           ,
                nhead           = heads            ,
                dropout         = dropout          ,
                batch_first     = True,
            ), 
            num_layers = layers,
            enable_nested_tensor = False if heads % 2 == 1 else True,
        )

    def forward(self, obs, act):

        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = self.ln(torch.cat([hidobs, hidact], dim=1))

        encoded = self.encoder(hidden.flatten(-2,-1)).view(hidden.shape)

        rew = self.hid2rew(encoded)[:,1:].squeeze(-1)
        val = self.hid2val(encoded)[:,1:].squeeze(-1)
        return rew, val


