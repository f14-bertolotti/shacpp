import random, numpy, tqdm, copy, torch

def layer_init(layer, std=1.141, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(torch.nn.Module):
    def __init__(self, layers=3, device="cuda:0"):
        super().__init__()
        self.logstd = torch.nn.Parameter(torch.ones(2, requires_grad=True, dtype=torch.float32, device=device) * -1)

        self.actor = torch.nn.Sequential(
            layer_init(torch.nn.Linear(2, 64, device=device)),
            *[layer_init(torch.nn.Linear(64, 64, device=device),std=.01) for _ in range(layers)],
            layer_init(torch.nn.Linear(64, 2, device=device),std=.01),
            torch.nn.Tanh()
        )

    def forward(self, state, deterministic = False):
        mu = self.actor(state)
        if deterministic: return mu

        std = self.logstd.exp() 
        probs  = torch.distributions.Normal(mu, std)
        action = probs.rsample()
        return action

class Critic(torch.nn.Module):
    def __init__(self, layers=3, device="cuda:0"):
        super().__init__()
        self.critic = torch.nn.Sequential(
            layer_init(torch.nn.Linear(2, 64, device=device)),
            *[layer_init(torch.nn.Linear(64, 64, device=device),std=.01) for _ in range(layers)],
            layer_init(torch.nn.Linear(64, 1, device=device),std=.01)
        )

    def forward(self, state):
        return self.critic(state).squeeze()

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

class Environment:
    def __init__(self, envs, agents, device="cuda:0"):
        self.envs, self.agents, self.device = envs, agents, device
        self.obj = torch.tensor([[-5,10]],requires_grad=True,device=self.device,dtype=torch.float32)

    def init_state(self):
        return torch.zeros((self.envs,2), requires_grad=True, dtype=torch.float32, device=self.device)

    def step(self, state, action):
        print(state[0],action[0])
        return (next := state + action), self.reward(next)

    def reward(self, state):
        dist = 1/(torch.cdist(state, self.obj)+1)
        return dist.mean(-1)


def run_actor(steps, envs, actor, critic, environment, sgamma, device="cuda:0"):

        state = environment.init_state()

        rewards = torch.zeros((steps+1, envs), dtype = torch.float32, device = device)
        values  = torch.zeros((steps+1, envs), dtype = torch.float32, device = device)

        buffer = {
            "observations" : torch.zeros((steps, envs, 2), dtype = torch.float32, device = device),
            "rewards"      : torch.zeros((steps, envs)   , dtype = torch.float32, device = device),
            "values"       : torch.zeros((steps, envs)   , dtype = torch.float32, device = device)
        }

        gamma = 1

        for step in range(steps):
            action = actor(state)
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
        actor_lr      = 2e-3,
        critic_lr     = 2e-3,
        alpha         = .4,
        lam           = .95,
        gamma         = .99,
        etc           = 10,
        device        = "cuda:0"
    ):

    environment = Environment(envs=envs, agents=agents, device=device)
    actor = Actor(device=device)
    critic = Critic(device=device)
    target_critic = copy.deepcopy(critic)
    actor_optimizer  = torch.optim.Adam(actor .parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    # main training process
    for agent_epoch in (tbar:=tqdm.tqdm(range(agent_epochs))):

        # train actor
        actor_optimizer.zero_grad()
        actor_result = run_actor(steps=steps, envs=envs, actor=actor, critic=target_critic, sgamma=gamma, environment=environment, device=device)
        actor_result["loss"].backward()
        actor_optimizer.step()

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                actor_result["buffer"]["observations"].view(envs*steps,2),
                compute_target_values(
                    steps   = steps,
                    envs    = envs,
                    values  = actor_result["buffer"][ "values"],
                    rewards = actor_result["buffer"]["rewards"],
                    slam    = lam,
                    gamma   = gamma,
                    device  = device
                ).view(envs*steps),
            ), 
            batch_size = batch_size,
            drop_last  = False,
            shuffle    = True
        )

        for critic_epoch in range(critic_epochs):
            for step, (observation, value) in enumerate(dataloader):
                critic_optimizer.zero_grad()
                critic_loss = ((critic(observation) - value)**2).mean()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss, actor_rewards = actor_result["loss"], actor_result["buffer"]["rewards"]
                tbar.set_description(f"{agent_epoch:0>3}-{critic_epoch:0>3}-{step:0>3}, cls:{critic_loss.item():8.5f}, als:{actor_loss.item():8.5f}, rew:{actor_rewards.mean().item():8.5f}")


            # update target critic
            with torch.no_grad():
                for param, param_targ in zip(critic.parameters(), target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)
        
        if agent_epoch % etc == 0: torch.save({"actor" : actor.state_dict()}, "actor.pkl")

if __name__=="__main__":
    train(envs=512, batch_size=2048, actor_lr=1e-4, critic_lr=1e-4, critic_epochs=4)
