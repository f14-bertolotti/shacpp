import jsonlines, random, pickle, numpy, tqdm, copy, torch
from RunningMeanStd import RunningMeanStd

def layer_init(layer, std=1.141, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(torch.nn.Module):
    def __init__(self, layers=3, observations=2, actions=2, first=None, hidden=None, device="cuda:0"):
        super().__init__()
        self.logstd = torch.nn.Parameter(torch.ones(actions, requires_grad=True, dtype=torch.float32, device=device) * -1)

        self.first   = layer_init(torch.nn.Linear(observations, 64, device=device)) if first is None else first
        self.hiddens = [layer_init(torch.nn.Linear(64, 64, device=device),std=.01) for _ in range(layers)] if hidden is None else hidden

        self.actor = torch.nn.Sequential(
            self.first,
            *self.hiddens,
            layer_init(torch.nn.Linear(64, actions, device=device),std=.01),
            torch.nn.Tanh()
        )

    def forward(self, state, deterministic = False):
        mu = self.actor(state.view(state.size(0),-1))/20
        if deterministic: return mu

        std = self.logstd.exp() 
        probs  = torch.distributions.Normal(mu, std)
        action = probs.rsample()
        return action

class Critic(torch.nn.Module):
    def __init__(self, layers=3, observations=2, first=None, hidden=None, device="cuda:0"):
        super().__init__()

        self.first   = layer_init(torch.nn.Linear(observations, 64, device=device)) if first is None else first
        self.hiddens = [layer_init(torch.nn.Linear(64, 64, device=device),std=.01) for _ in range(layers)] if hidden is None else hidden

        self.critic = torch.nn.Sequential(
            self.first,
            *self.hiddens,
            layer_init(torch.nn.Linear(64, 1, device=device),std=.01)
        )

    def forward(self, state):
        state = state.view(state.size(0),-1)
        return self.critic(state).squeeze()

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

class Environment:
    def __init__(self, envs, actor, agents, deterministic=False, device="cuda:0"):
        self.envs, self.agents, self.actor, self.device = envs, agents, actor, device
        self.points = torch.tensor([[(i/9)*2-1,(j/9)*2-1] for j in range(10) for i in range(10)],dtype=torch.float32,device=device)
        self.deterministic = deterministic

    def reset(self, state, p=.1):
        mask = torch.randperm(state.size(0)) < p * state.size(0)
        with torch.no_grad(): state[mask] = self.init_state()[mask]
        return state

    def init_state(self):
        #return torch.zeros((self.envs,self.agents,2), requires_grad=True, dtype=torch.float32, device=self.device)
        return torch.rand((self.envs,self.agents,2), requires_grad=True, dtype=torch.float32, device=self.device)*2-1

    def step(self, state, action):
        agents_actions = torch.stack([torch.zeros(state.size(0),2,device=self.device)] + [self.actor(torch.cat([state,state[:,[i]]],dim=1), deterministic=True) for i in range(1,self.agents)],dim=1).detach()

        state = state + agents_actions.detach()
        new_state = state.clone()
        new_state[:,0] = new_state[:,0] + action
        new_state = torch.clamp(new_state, -20, 20)
        return new_state, self.reward(new_state)

    def reward(self, state):
        # redevouz reward
        dist = torch.cdist(state, state)
        dist0 = torch.cdist(state, torch.zeros(1,2,device=self.device,dtype=torch.float32))
        dist0 = dist0.mean(-2).squeeze(-1)
        reward = 1/(dist.mean(-1).mean(-1)+1)
        reward[dist0>1] -= dist0[dist0>1] 
        
        ## scatter loss
        #dist = torch.cdist(self.points,state)
        #reward = (1/(1+dist.min(-1).values)).mean(-1)
    
        return reward


def run_actor(steps, envs, actor, critic, environment, sgamma, state=None, device="cuda:0"):

    if state is None: state = environment.init_state()
    state.clone().detach().requires_grad_(True)

    rewards = torch.zeros((steps+1, envs), dtype = torch.float32, device = device)
    values  = torch.zeros((steps+1, envs), dtype = torch.float32, device = device)
    
    buffer = {
        "observations" : torch.zeros((steps, envs, 9,2), dtype = torch.float32, device = device),
        "rewards"      : torch.zeros((steps, envs)     , dtype = torch.float32, device = device),
        "values"       : torch.zeros((steps, envs)     , dtype = torch.float32, device = device)
    }
    
    gamma = 1
    
    for step in range(steps):
        action = actor(torch.cat([state,state[:,[0]]],dim=1))
        next_state, reward = environment.step(state, action)
        rewards[step+1] = rewards[step] + gamma * reward
        values [step+1] = critic(next_state)
        gamma = gamma * sgamma
    
        with torch.no_grad():
            buffer[      "values"][step] = values[step+1].clone()
            buffer["observations"][step] = state         .clone()
            buffer[     "rewards"][step] = reward        .clone()
        
        state = next_state
    
    loss = -((rewards[steps,:]).mean() + gamma * values[steps,:]).sum() / (steps * envs)
    
    return { "loss" : loss, "buffer" : buffer }

@torch.no_grad()
def compute_target_values(steps, envs, values, rewards, slam, gamma, device="cuda:0"):
    #target_values = rewards + gamma * values
    ###########################################################################
    target_values = torch.zeros(steps, envs, dtype=torch.float32, device=device)
    Ai = torch.zeros(envs, dtype=torch.float32, device=device)
    Bi = torch.zeros(envs, dtype=torch.float32, device=device)
    lam = torch.ones(envs, dtype=torch.float32, device=device)
    for i in reversed(range(steps)):
        lam = lam * slam
        Ai = (slam * gamma * Ai + gamma * values[i] + (1. - lam) / (1. - slam) * rewards[i])
        Bi = Bi + rewards[i]
        target_values[i] = (1.0 - slam) * Ai + lam * Bi
    ###########################################################################
    return target_values

            
def train(
        steps         = 32,
        envs          = 64,
        agents        = 9,
        batch_size    = 512,
        agent_epochs  = 10000,
        critic_epochs = 16,
        actor_lr      = .001,
        critic_lr     = .001,
        alpha         = .4,
        lam           = .95,
        gamma         = .99,
        etc           = 10,
        metrics_path  = "data.jsonl",
        device        = "cuda:0"
    ):

    seed_everything(42)

    actor              = Actor(device=device, observations=(agents+1)*2, actions=2)
    critic             = Critic(device=device, observations=agents*2)
    target_critic      = copy.deepcopy(critic)
    actor_optimizer    = torch.optim.Adam(actor .parameters(), lr=actor_lr)
    critic_optimizer   = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    environment = Environment(envs=envs, agents=agents, actor=actor, device=device)

    # main training process
    state = None

    file, file_step = jsonlines.open(metrics_path, mode="w", flush=True), 0
    for agent_epoch in (tbar:=tqdm.tqdm(range(agent_epochs))):

        # train actor
        actor_optimizer.zero_grad()
        actor_result = run_actor(
            steps       = steps,
            envs        = envs,
            actor       = actor,
            critic      = target_critic,
            sgamma      = gamma,
            environment = environment,
            device      = device,
            state       = state
        )
        state = actor_result["buffer"]["observations"][-1]
        actor_result["loss"].backward()
        actor_optimizer.step()

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                actor_result["buffer"]["observations"].view(envs*steps,9,2),
                compute_target_values(
                    steps   = steps,
                    envs    = envs,
                    values  = actor_result["buffer"][ "values"],
                    rewards = actor_result["buffer"]["rewards"],
                    slam    = lam,
                    gamma   = gamma,
                    device  = device
                ).view(envs*steps)
            ), 
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )
        
        critic_sum_loss, critic_steps = 0, 0
        for critic_epoch in range(critic_epochs):
            for step, (observation, value) in enumerate(dataloader):
                critic_optimizer.zero_grad()
                critic_loss = ((critic(observation) - value)**2).mean()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss, actor_rewards = actor_result["loss"], actor_result["buffer"]["rewards"]
                tbar.set_description(f"{agent_epoch:0>3}-{critic_epoch:0>3}-{step:0>3}, cls:{critic_loss.item():8.5f}, als:{actor_loss.item():8.5f}, rew:{actor_rewards.mean().item():8.5f}")

                critic_sum_loss, critic_steps = critic_sum_loss + critic_loss.item(), critic_steps + 1


            # update target critic
            with torch.no_grad():
                for param, param_targ in zip(critic.parameters(), target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)


        # reset some states
        environment.reset(state, .1)
        file.write({"id"          : file_step,
                    "actor_loss"  : actor_result["loss"].item(),
                    "actor_reward": actor_result["buffer"]["rewards"].mean().item(),
                    "critic_loss" : critic_sum_loss/critic_steps })
        file_step += 1
        
        if agent_epoch % etc == 0: 
            torch.save({"actor" : actor.state_dict()}, "actor.pkl")

    file.close()

if __name__=="__main__":
    train(envs=1024, batch_size=5096, actor_lr=1e-3, critic_lr=1e-3, critic_epochs=4)
