from vmas.simulator.utils import Color
from vmas.scenarios import transport
import torch

class Transport(transport.Scenario):

    def max_rewards(self):
        distances = sum([self.world.get_distance(package, package.goal) for package in self.packages])
        max_rewards = distances * self.shaping_factor * len(self.world.agents)
        return max_rewards

    def reward(self, agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(self.world.batch_dim,device=self.world.device,dtype=torch.float32)

            for package in self.packages:
                package.dist_to_goal = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1)
                package.on_goal = self.world.is_overlapping(package, package.goal)

                # update package color
                red_color     = torch.tensor(Color.RED  .value, device=self.world.device, dtype=torch.float32)
                green_color   = torch.tensor(Color.GREEN.value, device=self.world.device, dtype=torch.float32)
                package.color = torch.where(package.on_goal.unsqueeze(-1).repeat(1,3), green_color, red_color)

                # update reward
                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew = torch.where(package.on_goal, self.rew, self.rew + package.global_shaping - package_shaping)

                package.global_shaping = package_shaping

        return self.rew

    def diffreward(self, prevs, acts, nexts):
        prevs_dist_to_goal = torch.linalg.vector_norm(prevs[:,:,4:6], dim=-1)
        nexts_dist_to_goal = torch.linalg.vector_norm(nexts[:,:,4:6], dim=-1)
        return (prevs_dist_to_goal - nexts_dist_to_goal) * 100


