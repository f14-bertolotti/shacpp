import copy, torch, click

@click.group()
def agent(): pass

class Agent:
    def __init__(
            self, 
            critic : torch.nn.Module,
            actor  : torch.nn.Module
    ):
        self.critic = critic
        self.actor  =  actor

    def get_value(self, x) -> dict[str, torch.Tensor]:
        return {
            "values" : self.critic(x)
        }

    def get_action(self, x, action=None):
        action_mean   = self.actor(x)
        action_logstd = self.actor.logstd.expand_as(action_mean)
        action_std    = torch.exp(action_logstd)
        probs         = torch.distributions.normal.Normal(action_mean, action_std)

        if action is None: action = torch.clamp(probs.rsample(),-1,1)

        return {
            "logits"   : action_mean,
            "actions"  : action,
            "logprobs" : probs.log_prob(action).sum(-1),
            "entropy"  : probs.entropy().sum(-1)
        }

    def get_action_and_value(self, observation:torch.Tensor, action:None|torch.Tensor=None) -> dict[str, torch.Tensor]:
        return self.get_action(observation, action=action) | self.get_value(observation)


