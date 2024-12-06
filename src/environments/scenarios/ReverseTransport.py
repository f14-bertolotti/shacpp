from vmas.scenarios import reverse_transport
from vmas.simulator.utils import Color
import torch

class ReverseTransport(reverse_transport.Scenario):

    def max_rewards(self):
        # todo
        max_rewards = self.shaping_factor * len(self.world.agents)
        return max_rewards
