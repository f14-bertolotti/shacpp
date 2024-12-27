from vmas.scenarios import discovery
from vmas.simulator.core import Agent
from vmas.simulator.utils import ScenarioUtils
import torch

class Discovery(discovery.Scenario):
    """ 
        This is a modification of the Discovery scenario to allow to max rewards to be calculated.
        This is possible by allowing a maximum number of respawn.
        We hardset this to 10.
    """

    def __init__(self, *args, agents_per_target, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp = agents_per_target

    def make_world(self, *args, **kwargs):
        return super().make_world(*args, agents_per_target=self.tmp, targets_respawn=False, **kwargs)

    def max_rewards(self):
        return torch.ones(self.world.batch_dim, device=self.world._device) * (self.n_targets)

    def zero_grad(self):
        self.world.zero_grad()

    def diffreward(self, prevs, acts, nexts):
        """ This reward is inherently non differentiable """
        raise NotImplementedError

    def reward(self, agent: Agent):
        return super().reward(agent)

