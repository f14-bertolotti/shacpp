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
        return action, probs.log_prob(action).sum(-1), probs.entropy()

    def get_action_and_value(self, x, action=None):
        return *self.get_action(x, action=action), self.get_value(x).squeeze(-1)

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

class Environment:
    def __init__(self, envs, agents, device="cuda:0"):
        self.envs, self.agents, self.device = envs, agents, device
        self.reset()
        self.points = torch.tensor([[(i/9)*20-10,(j/9)*20-10] for j in range(10) for i in range(10)],dtype=torch.float32,device=device)

    def reset(self):
        self.observation = torch.rand((self.envs,self.agents,2), device=self.device)*20-10
        return self.observation

    def step(self, action):
        self.observation += action
        return self.observation, self.reward()

    def reward(self):
        dists = torch.cdist(self.points, self.observation)
        return (dists.min(-1).values  < 3.333).float().mean(-1) - \
               (dists.min(-1).values >= 3.333).float().mean(-1)

##################################################################################
def compute_actor_loss(steps, envs, agents, environment, agent, sgamma, device="cuda:0"):
    observations = torch.zeros  (steps  , envs, agents, 2, dtype=torch.float32, device=device)
    rewards      = torch.zeros  (steps+1, envs, dtype=torch.float32, device=device)
    values       = torch.zeros  (steps+1, envs, dtype=torch.float32, device=device)
    actor_loss   = torch.tensor (0.           , dtype=torch.float32, device=device)

    # initialize trajectory to cut off gradients between episodes.
    observations[0] = environment.reset().detach()
    gamma = 1
    for i in range(0, steps-1):

        actions, _, _ = agent.get_action(observations[i])
        obs, r = environment.step(actions)
        observations[i + 1] = obs.detach()
        values[i + 1] = agent.get_value(observations[i+1]).squeeze(-1)
        rewards[i + 1] = rewards[i, :] + gamma * r
        gamma = gamma * sgamma

    # terminate all envs at the end of optimization iteration
    loss = (actor_loss + (- rewards[steps, :] - gamma * values[steps, :]).sum())/ (steps * envs)

    return observations.detach(), values, rewards, loss
    
@torch.no_grad()
def compute_target_values(steps, envs, values, rewards, slam, gamma, device="cuda:0"):
    target_values = torch.zeros(steps, envs, dtype=torch.float32, device=device)
    Ai = torch.zeros(envs, dtype=torch.float32, device=device)
    Bi = torch.zeros(envs, dtype=torch.float32, device=device)
    lam = torch.ones(envs, dtype=torch.float32, device=device)
    for i in reversed(range(steps)):
        lam = lam * slam
        Ai = (slam * gamma * Ai + gamma * values[i] + (1. - lam) / (1. - slam) * rewards[i])
        Bi = Bi + rewards[i]
        target_values[i] = (1.0 - slam) * Ai + lam * Bi
    return target_values
        
def compute_critic_loss(observations, target_values, critic):
    return ((critic(observations) - target_values) ** 2).mean()

def train(steps=64, envs=64, agents=9, batch_size=64, epochs=10000, episodes=4, actor_lr=.0001, critic_lr=.0001, alpha=.4, lam=.95, gamma=.99, device="cuda:0"):

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
        observations, values, rewards, actor_loss = compute_actor_loss(steps=steps, envs=envs, agents=agents, environment=environment, agent=agent, sgamma=gamma)
        actor_loss.backward()
        actor_optimizer.step()

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                observations,
                compute_target_values(steps=steps, envs=envs, values=values, rewards=rewards, slam=lam, gamma=gamma, device=device),
            ), 
            batch_size=batch_size, 
            drop_last=False,
            shuffle=True
        )

        for episode in range(episodes):
            for i, (obs,tgt) in enumerate(dataloader):
                critic_optimizer.zero_grad()
                critic_loss = compute_critic_loss(obs, tgt, agent.critic)
                critic_loss.backward()
                critic_optimizer.step()

                tbar.set_description(f"{epoch}-{episode}-{i}, cls:{critic_loss.item():7.5f}, als:{actor_loss.item():7.5f}")

        # update target critic
        with torch.no_grad():
            for critic_param, target_critic_param in zip(agent.critic.parameters(), target_agent.critic.parameters()):
                target_critic_param.data.mul_(alpha)
                target_critic_param.data.add_((1. - alpha) * critic_param.data)

        
if __name__=="__main__":
    torch.autograd.set_detect_anomaly(True)
    train()
