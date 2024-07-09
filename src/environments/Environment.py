from utils import RunningMeanStd
import torch, click


class Environment():
    def __init__(self, rms, envirs, agents, device):
        self.envirs = envirs
        self.agents = agents
        self.device = device
        self.rms = RunningMeanStd() if rms else None
    
    @torch.no_grad
    def update_statistics(self, observations):
        if self.rms: self.rms.update(observations.view(-1,observations.size(-1)))

    @torch.no_grad
    def normalize(self, obs):
        return torch.clip((obs - self.rms.mean) / torch.sqrt(self.rms.var + 1e-4), -10, 10) if self.rms else obs

    @torch.no_grad
    def state_dict(self):
        return self.rms


@click.group()
def environment(): pass
