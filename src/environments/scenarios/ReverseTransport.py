from vmas.scenarios import reverse_transport
from vmas.simulator.utils import Color
import torch

class ReverseTransport(reverse_transport.Scenario):

    def max_rewards(self):
        return 100000 # TODO - this is a placeholder value
