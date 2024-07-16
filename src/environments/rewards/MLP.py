from environments.rewards.Reward import reward
import torch, click

class MLP(torch.nn.Module):
    def __init__(self, environment, layers=1, hidden_size=64, output_size=1):
        super().__init__()

        observation_size = environment.get_observation_size()
        action_size = environment.get_action_size()
        device           = environment.device

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(observation_size + action_size, hidden_size, device=device),
            torch.nn.ReLU(),
            torch.nn.Dropout(.1),
            *[l for _ in range(layers) for l in [torch.nn.Linear(hidden_size, hidden_size , device=device), torch.nn.ReLU(), torch.nn.Dropout(.1)]],
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, device=device),
        )

    def forward(self, observation):
        return self.nn(observation).squeeze(-1)
        
@reward.group(invoke_without_command=True)
@click.option("--layers"      , "layers"      , type=int , default=1  , help="layers"     )
@click.option("--hidden-size" , "hidden_size" , type=int , default=64 , help="hidden size")
@click.option("--output-size" , "output_size" , type=int , default=1  , help="output size")
@click.pass_obj
def mlp(trainer, hidden_size, output_size, layers):
    trainer.environment.set_reward_nn(
        MLP(
            environment = trainer.environment,
            layers      = layers,
            hidden_size = hidden_size,
            output_size = output_size,
        )
    )

