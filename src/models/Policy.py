from utils import layer_init
import torch

class Policy(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, hidden_size=128, layers = 1, dropout=0.1, activation="Tanh", device="cuda:0", shared=False):
        super().__init__()
        if type(shared) is not list: shared = [shared] * (layers + 2)
        assert len(shared) == layers + 2
        self.agents = agents
        self.actions_size = action_size

        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)
        self.first_norm    = torch.nn.LayerNorm(hidden_size, device=device)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_act    = torch.nn.Tanh()

        self.action_var = torch.ones((action_size,)).to(device)

    def sample(self, observations):
        result        = self(observations)
        action_mean   = result["actions"]
        action_std    = torch.diag(self.action_var).unsqueeze(0).unsqueeze(0).repeat(1,action_mean.size(1),1,1)
        probs         = torch.distributions.MultivariateNormal(action_mean, action_std)
        actions       = probs.rsample()

        return {
            "actions"  : actions.view(action_mean.shape),
            "logits"   : result["logits"],
            "logprobs" : probs.log_prob(actions),
            "entropy"  : probs.entropy().sum(-1)
        }

    def eval_action(self, observations, actions):
        result        = self(observations)
        action_mean   = result["actions"]
        action_std    = torch.diag(self.action_var).unsqueeze(0).unsqueeze(0).repeat(1,action_mean.size(1),1,1)
        probs         = torch.distributions.MultivariateNormal(action_mean, action_std)
        
        return {
            "actions"  : actions,
            "logits"   : result["logits"],
            "logprobs" : probs.log_prob(actions),
            "entropy"  : probs.entropy().sum(-1)
        }


class PolicyAFO(Policy):
    def __init__(self, observation_size, action_size, agents, hidden_size=128, layers = 1, dropout=0.1, activation="Tanh", device="cuda:0"):
        super().__init__(observation_size, action_size, agents, hidden_size, layers, dropout, activation, device)
        self.first_layer   = torch.nn.Linear(observation_size*agents, hidden_size, device=device)
        self.last_layer    = torch.nn.Linear(hidden_size, agents*action_size, bias=False, device=device)
        self.last_layer    = layer_init(self.last_layer, 1.141)

    def forward(self, observations):
        observations = observations.flatten(-2,-1)

        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(observations))))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        actions = self.last_act(logits:=self.last_layer(hidden))
        return {
            "actions" : actions.view(-1, self.agents, self.actions_size),
            "logits" : logits.view(-1, self.agents, self.actions_size)
        }

class PolicyOFA(Policy):
    def __init__(self, observation_size, action_size, agents, hidden_size=128, layers = 1, dropout=0.1, activation="Tanh", device="cuda:0"):
        super().__init__(observation_size, action_size, agents, hidden_size, layers, dropout, activation, device)
        self.first_layer   = torch.nn.Linear(observation_size, hidden_size, device=device)
        self.last_layer    = torch.nn.Linear(hidden_size, action_size, bias=False, device=device)
        self.last_layer    = layer_init(self.last_layer, 1.141)

    def forward(self, observations):
        hidden = self.first_drop(self.first_norm(self.first_act(self.first_layer(observations))))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        actions = self.last_act(logits:=self.last_layer(hidden))
        return {
            "actions" : actions.view(-1, self.agents, self.actions_size),
            "logits" : logits.view(-1, self.agents, self.actions_size)
        }


