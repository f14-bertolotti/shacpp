from environments import SurrogateReward, Environment, environment, Options
from environments.rewards import reward
from scenarios import Dispersion as DispersionScenario
from optimizers import add_adam_command, add_sgd_command
from schedulers import add_constant_command, add_cosine_command
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

@environment.group(invoke_without_command=True)
@Options.environment
@click.pass_obj
def dispersion_sr(trainer, envs, agents, seed, grad_enabled, device, shared_reward, rms, state_dict_path):
    if hasattr(trainer, "environment"): return
    trainer.set_environment(
        SurrogateReward(
            Dispersion(
                envs         = envs          ,
                agents       = agents        ,
                seed         = seed          ,
                device       = device        ,
                shared_reward= shared_reward ,
                rms          = rms           ,
                grad_enabled = grad_enabled
            )
        )
    )
    if state_dict_path: trainer.environment.rms = torch.load(state_dict_path)["rms"]

dispersion_sr.add_command(reward)

@dispersion_sr.group()
def optimizer(): pass
add_adam_command(optimizer, srcnav=lambda x:x.environment, tgtnav=lambda x:x.environment.rewardnn, attrname="set_optimizer")
add_sgd_command (optimizer, srcnav=lambda x:x.environment, tgtnav=lambda x:x.environment.rewardnn, attrname="set_optimizer")

@dispersion_sr.group()
def scheduler(): pass
add_cosine_command  (scheduler, srcnav=lambda x:x.environment, tgtnav=lambda x:x.environment, attrname="set_scheduler")
add_constant_command(scheduler, srcnav=lambda x:x.environment, tgtnav=lambda x:x.environment, attrname="set_scheduler")

