from environments import environment
import random, torch, click, vmas

class RunningMeanStd():
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device="cuda:0"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

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
        self.rms = RunningMeanStd(shape=self.get_observation_space()[0].shape, device=device)

        print(self.get_observation_space())
        print(self.get_action_space())

    def step(self, action):
        next_observation, reward, done, info = self.env.step(action)
        return next_observation, reward, done, info
    
    @torch.no_grad
    def compute_statistics(self, observations):
        pass
        #self.rms.update(observations.view(-1,observations.size(-1)))

    @torch.no_grad
    def normalize(self, obs):
        return obs
        return torch.clip((obs - self.rms.mean) / torch.sqrt(self.rms.var + 1e-4), -10, 10)

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
@click.option("--state-dict-path", "state_dict_path"  , type=click.Path(), default=None)
@click.pass_obj
def dispersion(trainer, envs, agents, seed, max_steps, reset_prb, grad_enabled, device, state_dict_path):
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
    if state_dict_path: trainer.environment.rms = torch.load(state_dict_path)["rms"]
