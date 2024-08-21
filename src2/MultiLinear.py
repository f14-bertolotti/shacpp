import torch

class MultiLinear(torch.nn.Module):

    def __init__(self, channels, input_size, output_size, bias=True, requires_grad=True, device="cuda:0"):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.empty(channels, input_size, output_size, requires_grad=requires_grad).to(device))
        self.bias   = torch.nn.Parameter(torch.empty(channels, 1, output_size, requires_grad=requires_grad).to(device)) if bias else torch.tensor(0, requires_grad=False, device=device)

    def forward(self, x): 
        x = x.unsqueeze(-2)
        x = torch.matmul(x, self.weight) 
        x = x + self.bias
        x = x.squeeze(-2)
        return x

