import torch

class Model(torch.nn.Module):
    """ abstract Model with common attributes """
    def __init__(
        self,
        observation_size : int,
        action_size      : int,
        agents           : int,
        steps            : int
    ):
        super().__init__()
        self.observation_size = observation_size
        self.actions_size     =      action_size
        self.agents           =           agents
        self.steps            =            steps
