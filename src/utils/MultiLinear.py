import torch

class MultiLinear(torch.nn.Module):

    def __init__(self, channels, in_size, out_size, bias=False, requires_grad=True, device="cpu"):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.empty(channels, in_size, out_size, requires_grad=requires_grad).to(device))
        self.bias   = torch.nn.Parameter(torch.empty(channels, 1, out_size, requires_grad=requires_grad).to(device)) if bias else None

    def forward(self, x): 
        match self.bias:
            case None: return torch.matmul(x, self.weight)  
            case _   : return torch.matmul(x, self.weight) + self.bias


