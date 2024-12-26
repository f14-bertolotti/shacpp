from vmas.scenarios import sampling
import torch

class Sampling(sampling.Scenario):

        
    def compute_reward_grid(self):
        world_size = 2
        cell_size  = self.grid_spacing
        num_cells  = int(1+world_size // cell_size)
        cidx = torch.arange(num_cells).unsqueeze(0).repeat(num_cells,1)
        ridx = cidx.transpose(0,1)
        positions = torch.stack([ridx,cidx],dim=-1)
        positions = positions / num_cells * 2 - self.world.x_semidim
        positions = positions.unsqueeze(0).repeat(self.world.batch_dim,1,1,1).view(self.world.batch_dim,-1,2).transpose(0,1)
        rewards = sum([gaussian.log_prob(positions.to(self.world.device)).exp() for gaussian in self.gaussians])
        rewards = rewards / rewards.max()
        return rewards.transpose(0,1) * self.n_agents

    def reset_world_at(self, *args, **kwargs):
        super().reset_world_at(*args, **kwargs)
        self.reward_grid = self.compute_reward_grid()
        self.reward_grid.requires_grad = True
        self.untaken_grid = torch.ones_like(self.reward_grid, requires_grad=False)

    def max_rewards(self):
        return self.reward_grid.sum(1)

    def diffreward(self, prevs, actions, nexts):
        cell_size = self.grid_spacing
        world_size = 2
        num_cells = int(1+world_size // cell_size)
        positions = ((nexts[:,:,:2] + self.world.x_semidim) / 2 * num_cells).long()
        positions = positions[:,:,0] * num_cells + positions[:,:,1]
        
        rewards = (self.reward_grid * self.untaken_grid.detach().clone())[torch.arange(self.world.batch_dim).unsqueeze(1), positions]
        self.untaken_grid[torch.arange(self.world.batch_dim).unsqueeze(1), positions] = 0
        
        return rewards


