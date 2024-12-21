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

    def max_rewards(self):
        return torch.ones(self.world.batch_dim, device=self.world._device) * (self.n_targets)

    def zero_grad(self):
        self.world.zero_grad()

    def diffreward(self, prevs, acts, nexts):
        """ This reward is inherently non differentiable """
        raise NotImplementedError

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            self.reward_per_agent = torch.zeros(self.world.batch_dim, len(self.world.agents), self.n_targets, device=self.world._device)
            self.agents_pos  = torch.stack([a.state.pos for a in self.world.agents], dim=1)
            self.targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)
            self.agents_targets_dists = torch.cdist(self.agents_pos, self.targets_pos)
            self.agents_per_target = (self.agents_targets_dists < self._covering_range)
            self.covered_targets = self.agents_per_target.sum(1) >= self._agents_per_target
            tmp = self.covered_targets.sum(-1) > 0 
            self.reward_per_agent[tmp] = (self.reward_per_agent[tmp] + self.agents_per_target[tmp]) * self.covered_targets[tmp].unsqueeze(1)
            self.reward_per_agent = self.reward_per_agent.sum(-1)
            
            #self.reward_per_agent = self.agents_per_target[self.covered_targets.unsqueeze(-2).repeat(1, self.agents_per_target.size(1),1)]
            #print(self.reward_per_agent.shape)

            #print(self.reward_per_agent.shape)
            #.view(self.world.batch_dim, len(self.world.agents),-1).sum(-1)

        else:
            self.all_time_covered_targets += self.covered_targets
            for i, target in enumerate(self._targets):
                target.state.pos[self.covered_targets[:, i]] = self.get_outside_pos(None)[self.covered_targets[:, i]]

        return self.reward_per_agent[:,self.world.agents.index(agent)]
