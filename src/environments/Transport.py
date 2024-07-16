from environments import environment, Environment, Options
import torch, click, vmas

class Transport(Environment):

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
        
        self.world = vmas.make_env(
            "transport"                        ,
            n_agents           = agents        ,
            num_envs           = envs          ,
            device             = device        ,
            shared_reward      = shared_reward ,
            grad_enabled       = grad_enabled  ,
            continuous_actions = True          ,
            dict_spaces        = False         ,
            seed               = seed          ,
        )


@environment.group(invoke_without_command=True)
@Options.environment
@click.pass_obj
def transport(trainer, envs, agents, seed, grad_enabled, device, shared_reward, rms, state_dict_path):
    trainer.set_environment(
        Transport(
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
