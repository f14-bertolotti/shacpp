from environments import Environment, environment, Options
from scenarios import Dispersion as DispersionScenario
import torch, click, vmas

class Dispersion(Environment):

    def __init__(
        self,
        envs               = 64,
        agents             = 3,
        device             = "cuda:0",
        seed               = None, 
        shared_reward      = False,
        grad_enabled       = False,
        rms                = False,
    ): 
        super().__init__(rms, envs, agents, device)
        
        self.world = vmas.simulator.environment.Environment(
            DispersionScenario(
                device = device,
                radius = .05,
                agents = agents,
            ),
            n_agents           = agents,
            num_envs           = envs,
            device             = device,
            shared_reward      = shared_reward,
            grad_enabled       = grad_enabled,
            continuous_actions = True,
            dict_spaces        = False,
            seed               = seed,
        )

@environment.group(invoke_without_command=True)
@Options.environment
@click.pass_obj
def dispersion(trainer, envs, agents, seed, grad_enabled, device, shared_reward, rms, state_dict_path):
    if hasattr(trainer, "environment"): return
    trainer.set_environment(
        Dispersion(
            envs         = envs,
            agents       = agents,
            seed         = seed,
            device       = device,
            shared_reward= shared_reward,
            rms          = rms,
            grad_enabled = grad_enabled
        )
    )
    if state_dict_path: trainer.environment.rms = torch.load(state_dict_path)["rms"]


