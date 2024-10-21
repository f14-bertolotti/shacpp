from vmas.scenarios import transport
import torch

class Transport(transport.Scenario):

    def max_reward(self):
        return sum([(torch.linalg.vector_norm(
            package.state.pos - package.goal.state.pos, dim=1
        ) - min(self.package_width/2, self.package_length/2) - 0.15).mean().item() * self.shaping_factor * len(self.world.agents) for package in self.packages])

