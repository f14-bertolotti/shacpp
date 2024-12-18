from vmas.scenarios import sampling
from vmas.simulator.utils import Color
import torch

class Sampling(sampling.Scenario):

    def max_rewards(self):
        return 100000 # TODO - this is a placeholder value


