from vmas.scenarios import transport
from vmas.simulator.utils import Color
import torch

class Transport(transport.Scenario):

    def max_rewards(self):
        distances = sum([self.world.get_distance(package, package.goal) for package in self.packages])
        max_rewards = distances * self.shaping_factor * len(self.world.agents)
        return max_rewards

    def zero_grad(self):
        self.world.zero_grad()
        for package in self.packages:
            package.global_shaping = (
                torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )

    def reward(self, agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
                requires_grad=True,
            )

            for package in self.packages:
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color = torch.tensor(
                    Color.RED.value,
                    device=self.world.device,
                    dtype=torch.float32,
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value,
                    device=self.world.device,
                    dtype=torch.float32,
                )

                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew = self.rew.clone()
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping

        return self.rew
