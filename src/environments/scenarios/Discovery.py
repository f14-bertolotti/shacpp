from vmas.scenarios import discovery
from vmas.simulator.utils import Color
import torch

class Flocking(discovery.Scenario):

    def max_rewards(self):
        return 0

    def zero_grad(self):
        self.world.zero_grad()