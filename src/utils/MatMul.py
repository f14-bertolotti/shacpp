import torch

class MatMul(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def forward(self, x): 
        return torch.matmul(x, self.x)


