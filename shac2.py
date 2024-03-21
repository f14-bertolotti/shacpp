import random, numpy, tqdm, copy, torch

def layer_init(layer, std=1.141, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x): return self.func(x)

class Agent(torch.nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.cov_var = torch.full(size=(2,), fill_value=0.5, device=device)
        self.cov_mat = torch.diag(self.cov_var)

        self.embeddings = layer_init(torch.nn.Linear(2, 64, device=device))
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model         = 64,
                nhead           = 2,
                dim_feedforward = 256,
                activation      = "gelu",
                device          = device,
                batch_first     = True
            ), 
            num_layers=4
        )

        self.critic = torch.nn.Sequential(
            self.embeddings,
            self.encoder,
            Lambda(lambda x:x.sum(-2)),
            layer_init(torch.nn.Linear(64, 1, device=device), std=1),
        )
        
        self.actor = torch.nn.Sequential(
            self.embeddings,
            self.encoder,
            layer_init(torch.nn.Linear(64, 4, device=device),std=.01),
            torch.nn.Tanh()
        )


    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs  = torch.distributions.MultivariateNormal(logits[:,:,:2], self.cov_mat)
        action = probs.sample() if action is None else action
        return action

    def get_action_and_value(self, x, action=None):
        return self.get_action(x, action=action), self.get_value(x).squeeze(-1)

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

class Environment:
    def __init__(self, envs, agents, device="cuda:0"):
        self.envs, self.agents, self.device = envs, agents, device
        self.reset()
        self.points = torch.tensor([[(i/9)*20-10,(j/9)*20-10] for j in range(10) for i in range(10)], dtype=torch.float32, device=device)

    def reset(self):
        self.observation = torch.rand((self.envs,self.agents,2), requires_grad=True, device=self.device)*20-10
        return self.observation

    def step(self, observation, action):
        return observation + action, self.reward()

    def reward(self):
        dists = torch.cdist(self.points, self.observation)
        return (dists.min(-1).values  < 3.333).float().mean(-1) - \
               (dists.min(-1).values >= 3.333).float().mean(-1)

def train(steps=64, envs=32, agents=9, batch_size=64, epochs=10000, episodes=4, actor_lr=.0001, critic_lr=.0001, alpha=.4, lam=.95, gamma=.99, device="cuda:0"):

    environment = Environment(envs=envs, agents=agents, device=device)
    agent = Agent(device=device)
    target_agent = copy.deepcopy(agent)
    actor_optimizer  = torch.optim.Adam(agent.actor .parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=critic_lr)

    # main training process
    for epoch in (tbar:=tqdm.tqdm(range(epochs))):
        environment.reset()

        # train actor
        actor_optimizer.zero_grad()

        observation = environment.reset()
        actor_loss = torch.tensor(0., dtype = torch.float32, device = device)
        rewards = torch.zeros(steps+1, envs, dtype = torch.float32, device = device)
        values  = torch.zeros(steps+1, envs, dtype = torch.float32, device = device)
        gamma=1
        for step in range(64):
            actions = agent.get_action(observation)
            observation, r = environment.step(observation, actions)
            rewards[step+1] = rewards[step] + gamma * r
            values [step+1] = agent.get_value(observation)
            gamma = gamma * .99

        # terminate all envs at the end of optimization iteration
        actor_loss = (actor_loss + (- rewards[steps,:] - gamma * values[steps,:]).sum())/ (steps * envs)

        actor_loss.backward()
        actor_optimizer.step()


if __name__=="__main__":
    train()
