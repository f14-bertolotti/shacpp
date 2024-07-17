import torch

class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.nn.Parameter(torch.zeros(10))
        
    def forward(*args, **kwargs):
        pass

