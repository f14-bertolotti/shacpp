from utils import layer_init
from utils import Lambda
from MultiLinear import MultiLinear
from MLP import MLP
import torch


class ActorModel(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, hidden_size=128, layers = 1, dropout=0.1, activation="Tanh", device="cuda:0"):
        super().__init__()
        self.first_layer   = torch.nn.Linear(observation_size*agents, hidden_size, device=device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) for _ in  range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_layer    = layer_init(MultiLinear(channels=agents, input_size=hidden_size, output_size=action_size, bias=False, device=device), 1.41)
        self.last_act    = torch.nn.Tanh()

        self.logstd = torch.nn.Parameter(torch.zeros(1, action_size).to(device))


    def forward(self, observations):
        observations = observations.flatten(1,2).unsqueeze(1).repeat(1,observations.size(1),1)

        hidden = self.first_drop(self.first_act(self.first_layer(observations)))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = drop(act(layer(hidden)))
        actions = self.last_act(self.last_layer(hidden))
        return actions

    def sample(self, observations):
        action_mean   = self(observations)
        action_std    = torch.exp(self.logstd.expand_as(action_mean))
        probs         = torch.distributions.normal.Normal(action_mean, action_std)
        actions       = torch.clamp(probs.rsample(),-1,1)
        return actions 

class ActorModel2(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, hidden_size=128, layers = 1, dropout=0.1, activation="Tanh", device="cuda:0"):
        super().__init__()
        self.first_layer   = torch.nn.Linear(observation_size*agents, hidden_size, device=device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) for _ in  range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_layer    = layer_init(MultiLinear(channels=agents, input_size=hidden_size, output_size=action_size, bias=False, device=device), 1.41)
        self.last_act    = torch.nn.Tanh()

        self.logstd = torch.nn.Parameter(torch.zeros(1, action_size).to(device))


    def forward(self, observations):
        observations = observations.flatten(1,2).unsqueeze(1).repeat(1,observations.size(1),1)

        hidden = self.first_drop(self.first_act(self.first_layer(observations)))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        actions = self.last_act(self.last_layer(hidden))
        return actions

    def sample(self, observations):
        action_mean   = self(observations)
        action_std    = torch.exp(self.logstd.expand_as(action_mean))
        probs         = torch.distributions.normal.Normal(action_mean, action_std)
        actions       = torch.clamp(probs.rsample(),-1,1)
        return actions 

class ActorModel3(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, hidden_size=128, layers = 1, dropout=0.1, activation="Tanh", device="cuda:0"):
        super().__init__()
        self.agents = agents
        self.first_layer   = MultiLinear(agents, observation_size*(agents), hidden_size, device=device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)

        self.hidden_layers = torch.nn.ModuleList([MultiLinear(agents, hidden_size, hidden_size, device=device) for _ in  range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_layer    = MultiLinear(agents, hidden_size, action_size, bias=False, device=device)
        self.last_act    = torch.nn.Tanh()

        self.logstd = torch.nn.Parameter(torch.zeros(1, action_size).to(device))
        self.idxs   = [j for i in range(agents) for j in range(agents) if i != j]

    #def forward(self, observations, prev_actions):
    #    observations = observations.flatten(1,2).unsqueeze(1).repeat(1,observations.size(1),1)

    #    hidden = self.first_drop(self.first_act(self.first_layer(observations)))
    #    for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
    #        hidden = ln(hidden + drop(act(layer(hidden))))
    #    actions = self.last_act(self.last_layer(hidden) + prev_actions)
    #    return actions

    def forward(self, observations, prev_actions):
 
        detached_obs = observations.detach()[:,self.idxs].view(observations.size(0),self.agents, -1)

        src = torch.cat([observations, detached_obs], dim=-1)

        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(src))))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        actions = self.last_act(self.last_layer(hidden) + prev_actions)
        return actions

    def sample(self, observations, actions):
        action_mean   = self(observations, actions)
        action_std    = torch.exp(self.logstd.expand_as(action_mean))
        probs         = torch.distributions.normal.Normal(action_mean, action_std)
        actions       = torch.clamp(probs.rsample(),-1,1)
        return actions 

