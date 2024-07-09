from environments import environment
import numpy, torch, click, vmas
from scenarios import Dispersion as DispersionScenario
from environments import Environment

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


    def get_action_size(self):
        return numpy.prod(self.world.get_action_space     ()[0].shape)

    def get_observation_size(self):
        return numpy.prod(self.world.get_observation_space()[0].shape)

    def step(self, action):
        next_observation, reward, done, info = self.world.step(action.transpose(0,1))
        return {
            "observation" : torch.stack(next_observation).transpose(0,1), 
            "reward"      : torch.stack(reward          ).transpose(0,1), 
            "done"        : done, 
            "info"        : info,
        }

    @torch.no_grad
    def reset(self):
        return torch.stack(self.world.reset()).transpose(0,1)

    @torch.no_grad
    def render(self, *args, **kwargs):
        return self.world.render(*args, **kwargs)


@environment.group(invoke_without_command=True)
@click.option("--envs"            , "envs"            , type=int          , default=64      , help="number of parallel environments.")
@click.option("--device"          , "device"          , type=str          , default="cuda:0", help="device for the environments: cuda or cpu.")
@click.option("--seed"            , "seed"            , type=int          , default=None    , help="environment seed.")
@click.option("--agents"          , "agents"          , type=int          , default=3       , help="number of agents per environment.")
@click.option("--shared-reward"   , "shared_reward"   , type=bool         , default=False   , help="True if all agents share the reward.")
@click.option("--grad-enabled"    , "grad_enabled"    , type=bool         , default=False   , help="True if one can backpropagate in the simulator.")
@click.option("--rms"             , "rms"             , type=bool         , default=False   , help="True if statisticts and normalization of the observation should be computed.")
@click.option("--state-dict-path" , "state_dict_path" , type=click.Path() , default=None    , help="Path to where environment data has been stored.")
@click.pass_obj
def dispersion(trainer, envs, agents, seed, grad_enabled, device, shared_reward, rms, state_dict_path):
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
