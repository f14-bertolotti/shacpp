from environments import environment
import numpy, torch, click, vmas
from utils import RunningMeanStd

class Dispersion:

    def __init__(
        self,
        envs               = 64,
        agents             = 5,
        device             = "cuda:0",
        continuous_actions = True,
        wrapper            = None, 
        max_steps          = None, 
        seed               = None, 
        grad_enabled       = True,
        rms                = False,
    ): 
        self.envs      = envs
        self.agents    = agents
        self.device    = device
        

        self.env = vmas.make_env(
            scenario           = "dispersion"       ,
            num_envs           = envs               ,
            device             = device             ,
            continuous_actions = continuous_actions ,
            wrapper            = wrapper            ,
            max_steps          = max_steps          ,
            seed               = seed               ,
            dict_spaces        = False              ,
            grad_enabled       = grad_enabled       ,
            n_agents           = agents             ,
        )
        self.rms = RunningMeanStd() if rms else None

    def get_action_size(self):
        return numpy.prod(self.env.get_action_space     ()[0].shape)

    def get_observation_size(self):
        return numpy.prod(self.env.get_observation_space()[0].shape)

    def step(self, action):
        next_observation, reward, done, info = self.env.step(action.transpose(0,1))
        return {
            "observation" : torch.stack(next_observation).transpose(0,1), 
            "reward"      : torch.stack(reward          ).transpose(0,1), 
            "done"        : done, 
            "info"        : info,
        }
    
    @torch.no_grad
    def update_statistics(self, observations):
        if self.rms: self.rms.update(observations.view(-1,observations.size(-1)))

    @torch.no_grad
    def normalize(self, obs):
        return torch.clip((obs - self.rms.mean) / torch.sqrt(self.rms.var + 1e-4), -10, 10) if self.rms else obs

    @staticmethod
    def reset_with_prb(prev, environment, reset_prb):
        """ reset all environments from prev with probability reset_prb"""
        for i in (torch.rand(environment.num_envs) <= reset_prb).nonzero().flatten(): 
            prev[i] = torch.stack(environment.reset_at(i)[:,i])
        return prev

    @staticmethod
    def reset_with_prb_and_done(prev, dones, environment, reset_prb):
        """ reset all environments from prev that are done and reset with probability reset_prb """
        for i in (torch.rand(environment.num_envs, device=dones.get_device()) <= reset_prb).logical_or(dones).nonzero().flatten():
            prev[i] = torch.stack(environment.reset_at(i))[:,i]
        return prev

    @staticmethod
    def reset_with_done(prev, dones, environment):
        """ reset all environments from prev that are done and reset with probability reset_prb """
        for i in dones.nonzero().flatten(): prev[i] = torch.stack(environment.reset_at(i))[i]
        return prev

    @torch.no_grad
    def reset(self, prev=None, dones=None, reset_prb=None):
        match (prev is not None, dones is not None, reset_prb is not None and reset_prb > 0.0):
            case (False,False,False): return torch.stack(self.env.reset()).transpose(0,1)
            case (True ,False,False): return torch.stack(self.env.reset()).transpose(0,1)
            case (True ,False,True ): return Dispersion.reset_with_prb         (prev, self.env, reset_prb)
            case (True ,True ,True ): return Dispersion.reset_with_prb_and_done(prev, dones, self.env, reset_prb) 
            case (True ,True ,False): return Dispersion.reset_with_done        (prev, dones, self.env) 
            case (_    ,_    ,_    ): assert False, "Invalid argument combination"

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def state_dict(self):
        return self.rms

@environment.group(invoke_without_command=True)
@click.option("--envs"         , "envs"         , type=int  , default=64)
@click.option("--device"       , "device"       , type=str  , default="cuda:0")
@click.option("--seed"         , "seed"         , type=int  , default=None)
@click.option("--agents"       , "agents"       , type=int  , default=5)
@click.option("--max_steps"    , "max_steps"    , type=int  , default=None)
@click.option("--grad-enabled" , "grad_enabled" , type=bool , default=True)
@click.option("--rms"          , "rms"          , type=bool , default=False)
@click.option("--state-dict-path", "state_dict_path"  , type=click.Path(), default=None)
@click.pass_obj
def dispersion(trainer, envs, agents, seed, max_steps, grad_enabled, device, rms, state_dict_path):
    trainer.set_environment(
        Dispersion(
            envs         = envs,
            agents       = agents,
            seed         = seed,
            device       = device,
            max_steps    = max_steps,
            rms          = rms,
            grad_enabled = grad_enabled
        )
    )
    if state_dict_path: trainer.environment.rms = torch.load(state_dict_path)["rms"]
