import transformers, torch

class FeedForward(torch.nn.Module):
    def __init__(self, input_size, feedforward_size, dropout=0.0, activation="ReLU"):
        super().__init__()
        self.activation = getattr(torch.nn,activation)()
        self.lin1 = torch.nn.Linear(input_size, feedforward_size)
        self.lin2 = torch.nn.Linear(feedforward_size, input_size)
        self.ln  = torch.nn.LayerNorm(input_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.ln(x + self.lin2(self.activation(self.lin1(self.dropout(x)))))

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
            num_layers = layers,
            enable_nested_tensor = False
        )

        self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)

    def forward(self, obs, act):
        # obs ENVS x STEPS x AGENTS x ACTSIZE
        # act ENVS x 1     x AGENTS x OBSSIZE

        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)

        encoded = self.encoder(hidden.flatten(1,2)).view(hidden.shape)

        rew = self.hid2rew(encoded)[:,1:].squeeze(-1)
        val = self.hid2val(encoded)[:,1:].squeeze(-1)
        obs = self.hid2obs(encoded)
        return rew, val, obs

from mamba_ssm import Mamba
class MambaWorld(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, steps, layers=3, hidden_size=128, expansion_factor=16, dropout=0.0, device="cuda:0"):
        super().__init__()
        self.agents = agents
        self.steps  =  steps

        self.obs2hid = torch.nn.Linear(observation_size, hidden_size, device = device)
        self.act2hid = torch.nn.Linear(action_size     , hidden_size, device = device)
        self.actpos = torch.nn.Parameter(torch.empty(1, steps, 1, hidden_size, device = device).normal_(0,0.02))
        self.agnpos = torch.nn.Parameter(torch.empty(1, 1, agents, hidden_size, device = device).normal_(0,0.02))
        self.ln = torch.nn.LayerNorm(hidden_size, device=device)

        self.layers = torch.nn.ModuleList([Mamba(
            d_model = hidden_size ,
            d_state = expansion_factor,
            d_conv  = 4 ,
            expand  = 2 ,
        ).to(device) for layer in range(layers)])
        self.lns    = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)

    def forward(self, obs, act):
        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)
        tmpshape = hidden.shape
        hidden = hidden.flatten(1,2)

        for ln,layer in zip(self.lns, self.layers):
            hidden = ln(hidden + layer(hidden))

        hidden = hidden.view(tmpshape)

        rew = self.hid2rew(hidden)[:,1:].squeeze(-1)
        val = self.hid2val(hidden)[:,1:].squeeze(-1)
        obs = self.hid2obs(hidden)
        return rew, val, obs




class AxisWorld(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, steps, layers=3, hidden_size=128, heads=2, feedforward_size=512, dropout=0.1, activation="relu", device="cuda:0"):
        super().__init__()

        self.agents = agents
        self.steps  =  steps

        self.obs2hid = torch.nn.Linear(observation_size, hidden_size, device = device)
        self.act2hid = torch.nn.Linear(action_size     , hidden_size, device = device)
        self.actpos = torch.nn.Parameter(torch.empty(1, steps, 1, hidden_size, device = device).normal_(0,0.02))
        self.agnpos = torch.nn.Parameter(torch.empty(1, 1, agents, hidden_size, device = device).normal_(0,0.02))

        self.ln = torch.nn.LayerNorm(hidden_size, device=device)

        self.layers = torch.nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                dim_feedforward = feedforward_size ,
                d_model         = hidden_size      ,
                activation      = activation       ,
                device          = device           ,
                nhead           = heads            ,
                dropout         = dropout          ,
                batch_first     = True
            ), 
            num_layers = 1,
            enable_nested_tensor = False
        ) for ax in range(layers*2)])

        self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)

    def forward(self, obs, act):
        hidobs = self.obs2hid(obs)
        hidact = self.act2hid(act) + self.actpos

        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.agnpos)

        for i,layer in enumerate(self.layers):
            shape = hidden.shape
            hidden = layer(hidden.flatten(0,1)).view(shape).transpose(1,2)
            hidden = hidden

        rew = self.hid2rew(hidden)[:,1:].squeeze(-1)
        val = self.hid2val(hidden)[:,1:].squeeze(-1)
        obs = self.hid2obs(hidden)
        return rew, val, obs



class MLPWorld(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, steps, layers=3, hidden_size=128, dropout=0.0, activation="ReLU", device="cuda:0"):
        super().__init__()

        self.observation_size = observation_size
        self.agents = agents
        self.steps  =  steps

        self.first_layer = torch.nn.Linear(agents * (observation_size + (action_size) * steps), hidden_size, device=device)
        self.first_norm = torch.nn.LayerNorm(hidden_size, device=device)
        self.first_act  = getattr(torch.nn, activation)()

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.hid2rew = torch.nn.Linear(hidden_size, agents * steps * 1, device=device)
        self.hid2val = torch.nn.Linear(hidden_size, agents * steps * 1, device=device)
        self.hid2obs = torch.nn.Linear(hidden_size, agents * (steps+1) * observation_size, device=device)

    def forward(self, obs, act):
        hidden = self.first_norm(self.first_act(self.first_layer(torch.cat([obs.flatten(1,3), act.flatten(1,3)], dim=1))))

        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))

        rew = self.hid2rew(hidden).view(hidden.size(0), self.steps, self.agents)
        val = self.hid2val(hidden).view(hidden.size(0), self.steps, self.agents)
        obs = self.hid2obs(hidden).view(hidden.size(0), self.steps+1, self.agents, self.observation_size)
        return rew, val, obs


class MixerWorld(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, steps, layers=3, hidden_size=128, dropout=0.0, activation="ReLU", device="cuda:0"):
        super().__init__()

        self.observation_size = observation_size
        self.agents = agents
        self.steps  =  steps

        self.obs2hid = torch.nn.Linear(observation_size, hidden_size, device = device)
        self.act2hid = torch.nn.Linear(action_size     , hidden_size, device = device)
        self.stp2hid = torch.nn.Linear((steps+1) * agents, hidden_size, device=device)
        self.hid2stp = torch.nn.Linear(hidden_size, (steps+1) * agents, device=device)

        self.ff0 = FeedForward(hidden_size, hidden_size, dropout, activation).to(device)
        self.ff1 = FeedForward(hidden_size, hidden_size, dropout, activation).to(device)

        self.hid2rew = torch.nn.Linear(hidden_size, 1, device=device)
        self.hid2val = torch.nn.Linear(hidden_size, 1, device=device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device=device)

    def forward(self, obs, act):

        hidden = torch.cat([self.obs2hid(obs), self.act2hid(act)], dim=1).flatten(1,2)
        hidden = self.stp2hid(hidden.transpose(1,2)).transpose(1,2)
        hidden = self.ff0(hidden).transpose(1,2)
        hidden = self.hid2stp(self.ff1(hidden)).transpose(1,2)
        rew = self.hid2rew(hidden).view(obs.size(0), self.steps+1, self.agents)[:,1:]
        val = self.hid2val(hidden).view(obs.size(0), self.steps+1, self.agents)[:,1:]
        obs = self.hid2obs(hidden).view(obs.size(0), self.steps+1, self.agents, self.observation_size)
        
        return rew, val, obs


