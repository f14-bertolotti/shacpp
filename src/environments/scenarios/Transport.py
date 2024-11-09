from vmas.scenarios import transport
import torch

class Transport(transport.Scenario):

    def max_rewards(self):
        distances = sum([self.world.get_distance(package, package.goal) for package in self.packages])
        max_rewards = distances * self.shaping_factor * len(self.world.agents)
        return max_rewards

    def diffreward(self, prevs, acts, nexts):
        prevs_dist_to_goal = [torch.linalg.vector_norm(prev[:,4:6], dim=-1) for prev in prevs]
        nexts_dist_to_goal = [torch.linalg.vector_norm(next[:,4:6], dim=-1) for next in nexts]
        rewards = [(prev_dist - next_dist)*100 for prev_dist, next_dist in zip(prevs_dist_to_goal, nexts_dist_to_goal)]
        return rewards

