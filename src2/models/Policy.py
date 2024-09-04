from utils import layer_init
from MultiLinear import MultiLinear
import torch

class Policy(torch.nn.Module):
    def __init__(self, observation_size, action_size, agents, hidden_size=128, layers = 1, dropout=0.1, activation="Tanh", device="cuda:0", shared=False):
        super().__init__()
        if type(shared) is not list: shared = [shared] * (layers + 2)
        assert len(shared) == layers + 2

        self.first_layer   = torch.nn.Linear(observation_size*agents, hidden_size, device=device) if shared[0] else \
                             MultiLinear(agents, observation_size*agents, hidden_size, device=device)
        self.first_act     = getattr(torch.nn, activation)()
        self.first_drop    = torch.nn.Dropout(dropout)

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) if shared[i+1] else \
                                                  MultiLinear(agents, hidden_size, hidden_size) for i in  range(layers)])
        self.hidden_acts   = torch.nn.ModuleList([getattr(torch.nn, activation)() for _ in range(layers)])
        self.hidden_drops  = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(layers)])
        self.hidden_norms  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])

        self.last_layer    = torch.nn.Linear(hidden_size, action_size, bias=False, device=device) if shared[-1] else \
                             MultiLinear(channels=agents, input_size=hidden_size, output_size=action_size, bias=False, device=device)
        self.last_layer    = layer_init(self.last_layer, 1.141)
        self.last_act    = torch.nn.Tanh()

        self.action_var = torch.ones((action_size,)).to(device)

    def forward(self, observations):
        observations = observations.flatten(1,2).unsqueeze(1).repeat(1,observations.size(1),1)

        hidden = self.first_drop(self.first_act(self.first_layer(observations)))
        for layer, act, drop, ln in zip(self.hidden_layers, self.hidden_acts, self.hidden_drops, self.hidden_norms):
            hidden = ln(hidden + drop(act(layer(hidden))))
        actions = self.last_act(self.last_layer(hidden))
        return {
            "actions" : actions,
        }

    def sample(self, observations):
        action_mean   = self(observations)["actions"]
        action_std    = torch.diag(self.action_var).unsqueeze(0).unsqueeze(0).repeat(1,action_mean.size(1),1,1)
        probs         = torch.distributions.MultivariateNormal(action_mean, action_std)
        actions       = torch.clamp(probs.rsample(),-1,1)

        return {
            "actions"  : actions.view(action_mean.shape),
            "logprobs" : probs.log_prob(actions),
            "entropy"  : probs.entropy().sum(-1)
        }

    def eval_action(self, observations, actions):
        action_mean   = self(observations)["actions"]
        action_std    = torch.diag(self.action_var).unsqueeze(0).unsqueeze(0).repeat(1,action_mean.size(1),1,1)
        probs         = torch.distributions.MultivariateNormal(action_mean, action_std)
        
        return {
            "actions"  : actions,
            "logprobs" : probs.log_prob(actions),
            "entropy"  : probs.entropy().sum(-1)
        }


