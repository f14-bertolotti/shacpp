from vmas.scenarios import dispersion
import torch

class Dispersion(dispersion.Scenario):
    """ 
    This is almost identical to the dispersion vmas scenarios.
    The main problem with the original scenario is that all agents started at the same location
    with the same exact observation. 

    Here, agents start evenly distributed in circle of radius {self.radius}.
    Set {self.radius} = 0 to recover the original implementation.
    
    If the agents start all equal and share parameters they must come up with the same output (as they have identical inputs).
    """

    def __init__(self, agents, radius, device, **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        self.radius = radius
        self.device = device
        
        # precomputed evenly distributed points a circle
        x = (torch.pi*2/self.agents) * torch.arange(self.agents, device = device, dtype=torch.float32)
        self.start_positions = torch.stack([self.radius * torch.cos(x), self.radius * torch.sin(x)]).transpose(0,1)

    def reset_world_at(self, env_index: int = None):
        for i,agent in enumerate(self.world.agents):
            agent.set_pos(self.start_positions[i], batch_index=env_index)

        # the rest of the code is untouched wrt. the original dispersion implementation.
        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.pos_range,
                    self.pos_range,
                ),
                batch_index=env_index,
            )
            if env_index is None:
                landmark.eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.just_eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
            else:
                landmark.eaten[env_index] = False
                landmark.just_eaten[env_index] = False
                landmark.is_rendering[env_index] = True

    def diffreward(self, prevs, acts, nexts):
        raise NotImplementedError("Dispersion has a sparse, non-differentiable reward")

    def max_rewards(self):
        return torch.ones(self.world.batch_dim, device=self.device) * self.agents

