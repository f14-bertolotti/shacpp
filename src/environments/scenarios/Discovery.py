from vmas.scenarios import discovery
from vmas.simulator.core import Agent
from vmas.simulator.utils import ScenarioUtils
import torch

class Discovery(discovery.Scenario):
    """ 
        This is a modification of the Discovery scenario to allow to max rewards to be calculated.
        This is possible by allowing a maximum number of respawn.
        We hardset this to 10.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets_respawn = 10

    def max_rewards(self):
        return torch.ones(self.world.batch_dim, device=self.world._device) * (self.targets_respawn + self.n_targets)

    def zero_grad(self):
        self.world.zero_grad()

    def diffreward(self, prevs, acts, nexts):
        """ This reward is inherently non differentiable """
        raise NotImplementedError

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            self.time_rew = torch.full(
                (self.world.batch_dim,),
                self.time_penalty,
                device=self.world.device,
            )
            self.agents_pos = torch.stack(
                [a.state.pos for a in self.world.agents], dim=1
            )
            self.targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)
            self.agents_targets_dists = torch.cdist(self.agents_pos, self.targets_pos)
            self.agents_per_target = torch.sum(
                (self.agents_targets_dists < self._covering_range).type(torch.int),
                dim=1,
            )
            self.covered_targets = self.agents_per_target >= self._agents_per_target

            self.shared_covering_rew[:] = 0
            for a in self.world.agents:
                self.shared_covering_rew += self.agent_reward(a)
            self.shared_covering_rew[self.shared_covering_rew != 0] /= 2

        # Avoid collisions with each other
        agent.collision_rew[:] = 0
        for a in self.world.agents:
            if a != agent:
                agent.collision_rew[
                    self.world.get_distance(a, agent) < self.min_collision_distance
                ] += self.agent_collision_penalty

        if is_last:
            if self.targets_respawn > 0:
                self.targets_respawn -= 1
                occupied_positions_agents = [self.agents_pos]
                for i, target in enumerate(self._targets):
                    occupied_positions_targets = [
                        o.state.pos.unsqueeze(1)
                        for o in self._targets
                        if o is not target
                    ]
                    occupied_positions = torch.cat(
                        occupied_positions_agents + occupied_positions_targets,
                        dim=1,
                    )
                    pos = ScenarioUtils.find_random_pos_for_entity(
                        occupied_positions,
                        env_index=None,
                        world=self.world,
                        min_dist_between_entities=self._min_dist_between_entities,
                        x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                        y_bounds=(-self.world.y_semidim, self.world.y_semidim),
                    )

                    target.state.pos[self.covered_targets[:, i]] = pos[
                        self.covered_targets[:, i]
                    ].squeeze(1)
            else:
                self.all_time_covered_targets += self.covered_targets
                for i, target in enumerate(self._targets):
                    target.state.pos[self.covered_targets[:, i]] = self.get_outside_pos(
                        None
                    )[self.covered_targets[:, i]]
        covering_rew = (
            agent.covering_reward
            if not self.shared_reward
            else self.shared_covering_rew
        )

        return agent.collision_rew + covering_rew + self.time_rew
