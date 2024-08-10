import torch

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, layers=1, dropout=.1, hidden_size=64, activation="ReLU", device="cuda:0"):
        super().__init__()

        activation = getattr(torch.nn, activation)

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, device=device),
            activation(),
            torch.nn.Dropout(dropout),
            *[l for _ in range(layers) for l in [torch.nn.Linear(hidden_size, hidden_size , device=device), activation(), torch.nn.Dropout(.1)]],
            activation(),
            torch.nn.Linear(hidden_size, output_size, device=device),
        )

    def forward(self, observation):
        return self.nn(observation).squeeze(-1)
