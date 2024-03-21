import random, numpy, tqdm, torch

def layer_init(layer, std=1.141, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FeedForward(torch.nn.Module):
    def __init__(self, input_size=64, hidden_size=64, activation=torch.nn.ReLU(), device="cuda:0"):
        super().__init__()

        self.lin0 = layer_init(torch.nn.Linear(input_size, hidden_size, device=device))
        self.lin1 = layer_init(torch.nn.Linear(hidden_size, input_size, device=device))
        self.ln   = torch.nn.LayerNorm(input_size, device=device)
        self.activation = activation

    def forward(self, x):
        return x + self.ln(self.lin1(self.activation(self.lin0(x)))) 


class Agent(torch.nn.Module):
    def __init__(self, envs, device="cuda:0"):
        super().__init__()
        self.critic = torch.nn.Sequential(
            layer_init(torch.nn.Linear(2, 64, device=device)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64, device=device)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 1, device=device), std=1),
        )
        self.actor = torch.nn.Sequential(
            layer_init(torch.nn.Linear(2, 64, device=device)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64, device=device)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 8, device=device),std=.01),
        )

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs  = torch.distributions.Categorical(logits=logits)
        action = probs.sample() if action is None else action
        return action, probs.log_prob(action), probs.entropy()

    def get_action_and_value(self, x, action=None):
        return *self.get_action(x, action=action), self.critic(x)

def seed_everything(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

class Environment:
    def __init__(self, envs, device="cuda:0"):
        self.envs, self.device = envs, device
        self.reset()
        self.act2mv = torch.tensor([[+1,+1],[-1,+1],[+1,-1],[-1,-1],[+0,1],[+0,-1],[+1,+0],[-1,+0]],dtype=torch.float32,device=device)

    def reset(self):
        self.observation = torch.rand((self.envs,2), device=self.device)*20-10
        return self.observation

    def step(self, action):
        self.observation += self.act2mv[action]
        return self.observation, self.reward()

    def reward(self):
        return (self.observation[:,0] >= 5).logical_and\
               (self.observation[:,1] >= 5).logical_and\
               (self.observation[:,0] <= 15).logical_and\
               (self.observation[:,1] <= 15).float()

class Storage(torch.utils.data.Dataset):

    def __init__(self, steps, envs, device):
        self.steps, self.envs, self.device = steps, envs, device
        self.reset()

    def reset(self):
        self.observations = torch.zeros(self.steps, self.envs, 2).to(self.device)
        self.actions      = torch.zeros(self.steps, self.envs).to(self.device)
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
        entcoef       = .01,
        vfcoef        = .5,
        max_grad_norm = .5,
        tmax          = 400,
        lrmin         = 1e-5,
        etc           = 10,
    ):

    seed_everything(seed)

    environment = Environment(envs=envs, device=device)
    agent       = Agent(envs, device=device)
    optimizer   = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    storage     = Storage(envs=envs, steps=steps, device=device)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=lrmin)
    observation = environment.reset()

    for update in (tbar:=tqdm.tqdm(range(1, updates + 1))):

        storage.reset()
        for step in range(0, steps):
            storage.observations[step] = observation

            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(observation)
                storage.values[step] = value.flatten()
            
            observation, reward = environment.step(action)
            storage.actions  [step] = action
            storage.logprobs [step] = logprob
            storage.rewards  [step] = reward

        storage.compute_returns_and_advantages(observation, agent, gamma, gaelambda) 

        for epoch in range(epochs):

            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    storage.observations.view(-1,2),
                    storage.actions     .flatten(),
                    storage.logprobs    .flatten(),
                    storage.rewards     .flatten(),
                    storage.values      .flatten(),
                    storage.returns     .flatten(),
                    storage.advantages  .flatten()
                ), 
                batch_size=batch_size, 
                shuffle=True
            )
            for observations, actions, logprobs, rewards, oldvalues, returns, advantages in dataloader:

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

                tbar.set_description(f"{update}-{epoch}, lr:{scheduler.get_last_lr()[0]:7.6f}, vl:{vloss.item():7.4f}, el:{eloss.item():7.4f}, pl:{ploss.item():7.4f}, r:{storage.rewards.mean():7.4f}")

            scheduler.step()

        if update % etc == 0: torch.save({"agentsd" : agent.state_dict()}, "agent.pkl")

if __name__ == "__main__":
    main(lrmin=1e-4,lr=1e-4)
