from vmas.scenarios import sampling
from vmas.simulator.utils import Color
import torch

class Sampling(sampling.Scenario):

    def max_rewards(self):
        # todo
        max_rewards = self.shaping_factor * len(self.world.agents)
        return max_rewards


