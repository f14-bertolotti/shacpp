import torch

class World(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, steps, layers=3, hidden_size=128, heads=2, feedforward_size=512, dropout=0.1, max_positions=512, activation="relu", device="cuda:0"):
        super().__init__()

        self.agents = agents

        self.obs2hid = torch.nn.Linear(observation_size, hidden_size, device = device)
        self.act2hid = torch.nn.Linear(action_size     , hidden_size, device = device)
        self.actpos = torch.nn.Parameter(torch.empty(1, steps, 1, hidden_size, device = device).normal_(0,0.02))
        self.agnpos = torch.nn.Parameter(torch.empty(1, 1, agents, hidden_size, device = device).normal_(0,0.02))

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                dim_feedforward = feedforward_size ,
                d_model         = hidden_size      ,
                activation      = activation       ,
                device          = device           ,
                nhead           = heads            ,
                batch_first     = True
            ), 
            num_layers = layers
        )

        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)

    def forward(self, obs, act):
        act = act.transpose(0,1) # ENVS x STEPS x AGENTS x ACTSIZE
        obs = obs.unsqueeze(1)   # ENVS x 1     x AGENTS x OBSSIZE

        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = torch.cat([hidobs, hidact], dim=1) + self.agnpos

        encoded = self.encoder(hidden.flatten(1,2)).view(hidden.shape)

        obs = self.hid2obs(encoded) + obs
        return obs[:,:-1,:,:].transpose(0,1)


