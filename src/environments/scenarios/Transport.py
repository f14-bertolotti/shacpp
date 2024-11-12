from vmas.scenarios import transport
import torch

class Transport(transport.Scenario):

    def max_rewards(self):
        distances = sum([self.world.get_distance(package, package.goal) for package in self.packages])
        max_rewards = distances * self.shaping_factor * len(self.world.agents)
        return max_rewards

    def diffreward(self, prevs, acts, nexts):
        prevs_dist_to_goal = torch.linalg.vector_norm(prevs[:,:,4:6], dim=-1)
        nexts_dist_to_goal = torch.linalg.vector_norm(nexts[:,:,4:6], dim=-1)
        return (prevs_dist_to_goal - nexts_dist_to_goal) * 100


