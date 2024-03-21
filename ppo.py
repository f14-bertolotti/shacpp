import random, numpy, tqdm, torch

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
    def __init__(self, agents, device="cuda:0"):
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

                

class Storage(torch.utils.data.Dataset):

    def __init__(self, steps, envs, agents, device):
        self.steps, self.envs, self.agents, self.device = steps, envs, agents, device
        self.reset()

    def reset(self):
        self.observations = torch.zeros(self.steps, self.envs, self.agents, 2).to(self.device)
        self.actions      = torch.zeros(self.steps, self.envs, self.agents, 2).to(self.device)
        self.logprobs     = torch.zeros(self.steps, self.envs).to(self.device)
        self.rewards      = torch.zeros(self.steps, self.envs).to(self.device)
        self.values       = torch.zeros(self.steps, self.envs).to(self.device)
        self.returns      = torch.zeros(self.steps, self.envs).to(self.device)
        self.advantages   = torch.zeros(self.steps, self.envs).to(self.device)

    def compute_returns_and_advantages(self, observation, agent, gamma, gaelambda):
        with torch.no_grad():
            lastgaelam = 0
            next_value = agent.get_value(observation).reshape(1, -1)
            for t in reversed(range(self.steps)):
                nextvalues = next_value if t == self.steps - 1 else self.values[t + 1]
                delta = self.rewards[t] + gamma * nextvalues - self.values[t]
                self.advantages[t] = lastgaelam = delta + gamma * gaelambda  * lastgaelam
            self.returns = self.advantages + self.values

def policy_loss(advantages, ratio, clipcoef):
    loss1 = -advantages * ratio
    loss2 = -advantages * torch.clamp(ratio, 1 - clipcoef, 1 + clipcoef)
    loss = torch.max(loss1, loss2).mean()
    return loss

def value_loss(newvalue, oldvalues, returns, clipcoef):
    v_loss_unclipped = (newvalue - returns) ** 2
    v_clipped = oldvalues + torch.clamp(newvalue - oldvalues, -clipcoef, clipcoef)
    v_loss_clipped = (v_clipped - returns) ** 2
    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    vloss = 0.5 * v_loss_max.mean()
    return vloss

def main(
        seed          = 42,
        device        = "cuda:0",
        envs          = 16,
        steps         = 128,
        lr            = 1e-4,
        updates       = 1000,
        gamma         = .99,
        gaelambda    = .95,
        batch_size    = 32,
        epochs        = 4,
        clipcoef      = .2,
        entcoef       = .1,
        vfcoef        = .5,
        max_grad_norm = .5,
        tmax          = 400,
        lrmin         = 1e-5,
        etc           = 10,
        agents        = 9,
    ):

    seed_everything(seed)

    environment = Environment(envs=envs, agents=agents, device=device)
    agent       = Agent(agents=agents, device=device)
    optimizer   = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    storage     = Storage(envs=envs, steps=steps, agents=agents, device=device)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=lrmin)
    observation = environment.reset()

    for update in (tbar:=tqdm.tqdm(range(1, updates + 1))):

        storage.reset()
        for step in range(0, steps):
            storage.observations[step] = observation

            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(observation)
                storage.values[step] = value
            
            observation, reward = environment.step(action)
            storage.actions  [step] = action
            storage.logprobs [step] = logprob
            storage.rewards  [step] = reward

        storage.compute_returns_and_advantages(observation, agent, gamma, gaelambda) 

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                storage.observations.view(-1,agents,2),
                storage.actions     .view(-1,agents,2),
                storage.logprobs    .flatten(),
                storage.rewards     .flatten(),
                storage.values      .flatten(),
                storage.returns     .flatten(),
                storage.advantages  .flatten()
            ), 
            batch_size=batch_size, 
            shuffle=True
        )

        for epoch in range(epochs):
            for i, (observations, actions, logprobs, rewards, oldvalues, returns, advantages) in enumerate(dataloader):

                _, newlogprob, entropy, newvalues = agent.get_action_and_value(observations, actions)

                # Normalize
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                ploss = policy_loss(advantages, (newlogprob - logprobs).exp(), clipcoef)

                # Value loss
                vloss = value_loss(newvalues, oldvalues, returns, clipcoef)
                
                # Entropy loss
                eloss = entropy.mean()

                loss = ploss + vloss * vfcoef - entcoef * eloss 

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                tbar.set_description(f"{update}-{epoch}-{i}, lr:{scheduler.get_last_lr()[0]:7.6f}, vl:{vloss.item():7.4f}, el:{eloss.item():7.4f}, pl:{ploss.item():7.4f}, r:{rewards.mean():7.4f}")

            scheduler.step()

        if update % etc == 0: torch.save({"agentsd" : agent.state_dict()}, "agent.pkl")

if __name__ == "__main__":
    main(lrmin=1e-5,lr=1e-4,steps=64,epochs=4,envs=64,batch_size=256,updates=100000)
