from storages import storage
import click, torch


class Default:
    def __init__(self, steps=64, envs=64, agents=9, observations=2, device="cuda:0"):
        self.observations = torch.zeros(steps, envs, agents, observations).to(device)
        self.actions      = torch.zeros(steps, envs, agents, observations).to(device)
        self.logprobs     = torch.zeros(steps, envs).to(device)
        self.rewards      = torch.zeros(steps, envs).to(device)
        self.values       = torch.zeros(steps, envs).to(device)
        self.returns      = torch.zeros(steps, envs).to(device)
        self.advantages   = torch.zeros(steps, envs).to(device)

@storage.group(invoke_without_command=True)
@click.option("--observations", "observations", type=int, default=2)
@click.pass_obj
def default(trainer, observations):
    trainer.set_storage(
        Default(
            steps        = trainer.trajectory.steps,
            envs         = trainer.environment.envs,
            agents       = trainer.environment.agents,
            observations = observations,
            device       = trainer.agent.device
        )
    )
