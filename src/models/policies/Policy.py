import models
import torch

class Policy(models.Model):
    """ base Policy """

    def __init__(
        self,
        observation_size : int              ,
        action_size      : int              ,
        agents           : int              ,
        steps            : int              ,
        var              : float = 1.0      ,
        device           : str   = "cuda:0"
    ):
        super().__init__(observation_size, action_size, agents, steps)

        self.action_var = var * torch.ones((action_size,)).to(device)

    def sample(self, observations):
        action_mean = self(observations)
        action_std  = torch.diag(self.action_var).unsqueeze(0).unsqueeze(0).repeat(1,action_mean.size(1),1,1)
        probs       = torch.distributions.MultivariateNormal(action_mean, action_std)
        actions     = probs.rsample()

        return {
            "actions"  : actions.clamp(-1,+1).view(action_mean.shape),
            "logits"   : action_mean,
            "logprobs" : probs.log_prob(actions),
            "entropy"  : probs.entropy().sum(-1)
        }

    def act(self, observations):
        logits = self(observations)
        return {
            "actions" : logits.clamp(-1,+1),
            "logits"  : logits
        }

    def eval_action(self, observations, actions):
        action_mean = self(observations)
        action_std  = torch.diag(self.action_var).unsqueeze(0).unsqueeze(0).repeat(1,action_mean.size(1),1,1)
        probs       = torch.distributions.MultivariateNormal(action_mean, action_std)
        
        return {
            "logits"   : action_mean,
            "actions"  : action_mean.clamp(-1,+1),
            "logprobs" : probs.log_prob(actions),
            "entropy"  : probs.entropy().sum(-1)
        }


