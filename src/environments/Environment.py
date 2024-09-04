from utils import RunningMeanStd
import torch, numpy, click


class Environment():
    def __init__(self, rms, envirs, agents, device):
        self.envirs = envirs
        self.agents = agents
        self.device = device
        self.rmssts = RunningMeanStd() if rms else None
        self.rms    = rms
    
    def step(self, action:torch.Tensor, oldobs:torch.Tensor|None = None) -> dict[str,torch.Tensor]:
        next_observation, reward, done, info = self.world.step(action.transpose(0,1))
        return {
            "observation" : torch.stack(next_observation).transpose(0,1), 
            "reward"      : torch.stack(reward          ).transpose(0,1), 
            "done"        : done, 
            "info"        : info,
        }

    @torch.no_grad
    def reset(self):
        return torch.stack(self.world.reset()).transpose(0,1)

    @torch.no_grad
    def render(self, *args, **kwargs):
        return self.world.render(*args, **kwargs)

    def update_statistics(self, observations):
        if self.rmssts: self.rmssts.update(observations.view(-1,observations.size(-1)))

    @torch.no_grad
    def normalize(self, obs):
        return torch.clip((obs - self.rmssts.mean) / torch.sqrt(self.rmssts.var + 1e-4), -10, 10) if self.rmssts else obs

    @torch.no_grad
    def state_dict(self):
        return self.rmssts
    
    @torch.no_grad
    def get_action_size(self):
        return 2

    @torch.no_grad
    def get_observation_size(self):
        return 13


@click.group()
def environment(): pass
