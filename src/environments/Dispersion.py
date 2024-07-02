from environments import environment
import random, torch, click, vmas


class Dispersion:

    def __init__(
        self,
        envs               = 64,
        agents             = 5,
        device             = "cuda",
        continuous_actions = True,
        wrapper            = None, 
        max_steps          = None, 
        seed               = None, 
        grad_enabled       = True  ,
        reset_prb          = 0.15
    ): 
        self.envs   = envs
        self.agents = agents
        self.reset_prb = reset_prb
        

        self.env = vmas.make_env(
            scenario           = "dispersion",
            num_envs           = envs,
            device             = device,
            continuous_actions = continuous_actions,
            wrapper            = wrapper,
            max_steps          = max_steps,
            seed               = seed,
            dict_spaces        = False,
            grad_enabled       = grad_enabled,
            n_agents           = agents,
        )

        print(self.get_observation_space())
        print(self.get_action_space())

    def step(self, observation, action):
        #print("actions",action.shape)
        next_observation, reward, done, info = self.env.step(action)
        #print(torch.stack(reward).shape)
        #print(torch.stack(next_observation).shape)
        return {
            "observations"      : observation,
            "next_observations" : torch.stack(next_observation),
            "rewards"           : torch.stack(reward),
            "done"              : done.unsqueeze(0).repeat(self.agents, 1)
        }

    @torch.no_grad
    def reset(self, prev=None, dones=None):
        match (prev is None, dones is None):
            case (True , True): return self.env.reset()
            
            case (True ,False):
                for i in map(lambda x:x[0], filter(lambda x: x[1], enumerate(torch.rand(self.envs) <= self.reset_prb))): 
                    prev[:,i] = torch.stack(self.env.reset_at(i)[:,i])
                return [x for x in prev]

            case (False ,False): 
                for i in map(lambda x:x[0], filter(lambda x: x[1], enumerate((torch.rand(self.envs, device=dones.get_device()) <= self.reset_prb).logical_or(dones)))):
                    prev[:,i] = torch.stack(self.env.reset_at(i))[:,i]
                return [x for x in prev]
            
            case (False, True): assert False, "Previous observations should be provided"

        assert False, "Error: case not covered"

    def get_action_space(self):
        return self.env.get_action_space()

    def get_observation_space(self):
        return self.env.get_observation_space()

    def train_step(self, *args, **kwargs): pass

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

@environment.group(invoke_without_command=True)
@click.option("--envs"         , "envs"         , type=int  , default=64)
@click.option("--device"       , "device"       , type=str  , default="cuda:0")
@click.option("--seed"         , "seed"         , type=int  , default=None)
@click.option("--agents"       , "agents"       , type=int  , default=5)
@click.option("--reset-prb"    , "reset_prb"    , type=float, default=0.15)
@click.option("--max_steps"    , "max_steps"    , type=int  , default=None)
@click.option("--grad-enabled" , "grad_enabled" , type=bool , default=True)
@click.pass_obj
def dispersion(trainer, envs, agents, seed, max_steps, reset_prb, grad_enabled, device):
    trainer.set_environment(
        Dispersion(
            envs   =   envs,
            agents = agents,
            seed   =   seed,
            device = device,
            reset_prb = reset_prb,
            max_steps = max_steps,
            grad_enabled = grad_enabled
        )
    )


