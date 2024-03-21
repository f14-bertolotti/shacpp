import tqdm, math, torch

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class Critic(torch.nn.Module):
    def __init__(self, layers=3, hidden_size=128, device="cuda:0"):
        super().__init__()
        self.embedding = torch.nn.Linear(2, hidden_size, device=device)
        self.layernorm = torch.nn.LayerNorm(hidden_size, device=device)
        self.critic = torch.nn.Linear(hidden_size, 1, device=device)


        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model         = hidden_size,
                nhead           = 2,
                dim_feedforward = hidden_size*2,
                activation      = "relu",
                device          = device,
                batch_first     = True
            ), 
            num_layers=layers
        )

    def forward(self, state):
        embedding = self.layernorm(self.embedding(state))
        embedding = self.encoder(embedding)
        critic    = self.critic(embedding)
        return critic


class Actors(torch.nn.Module):
    def __init__(self, agents=25, layers=3, hidden_size=128, device="cuda:0"):
        super().__init__()
        self.embedding = torch.nn.Linear(2, hidden_size, device=device)
        self.layernorm = torch.nn.LayerNorm(hidden_size, device=device)
        self.logits = torch.nn.Linear(hidden_size, 8, device=device)

        self.pos_embedding = torch.nn.Parameter(
            torch.normal(
                mean   = 0,
                std    = 1,
                size   = (agents, hidden_size),
                dtype  = torch.float,
                requires_grad=True,
                device = device
            )
        )

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model         = hidden_size,
                nhead           = 2,
                dim_feedforward = hidden_size*2,
                activation      = "relu",
                device          = device,
                batch_first     = True
            ), 
            num_layers=layers
        )

    def forward(self, state):
        embedding = self.layernorm(self.pos_embedding + self.embedding(state))
        embedding = self.encoder(embedding)
        logits    = self.logits(embedding)
        return torch.nn.functional.softmax(logits, dim=-1)

class Environment:

    def __init__(self, actor, critic, agents, batch=128, device="cuda:0"):
       self.actor, self.critic, self.batch, self.agents, self.device = actor, critic, batch, agents, device
       self.actions = torch.tensor([[1,0],[0,1],[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]], device=device, dtype=torch.float)
       self.points = torch.tensor([[(i/2)*20-10,(j/2)*20-10] for i in range(3) for j in range(3)],dtype=torch.float, device=device)
       self.reset()

    def reset(self):
        self.state = torch.zeros(
            size   = (self.batch, self.agents, 2),
            device = self.device,
            dtype  = torch.float
        )

    def step(self):
        action_probs = self.policy(self.state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
        self.state = (self.state + self.actions[action])
        state_val = self.critic(self.state)
        reward = self.reward()

        return {
            "state"           : self.state.detach(),
            "action_logprobs" : action_logprobs.detach(),
            "action_probs"    : action_probs.detach(),
            "state_val"       : state_val.detach(),
            "reward"          : reward.detach()
        }

    def reward(self):
        inside  = (torch.cdist(self.points, self.state) <  3).sum(1)
        outside = (torch.cdist(self.points, self.state) >= 3).sum(1)
        #negative = (torch.cdist(self.points, self.state).min(-1).values >= 5).sum(-1).unsqueeze(-1)
        return inside - outside

def discounted_reward(episode, gamma, device):
    R, rewards = torch.zeros_like(episode[0]["reward"], dtype=torch.float, device=device, requires_grad=True), []
    for step in episode[::-1]:
        R = step["reward"] + gamma * R
        rewards.insert(0,R)
    rewards = torch.stack(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    return rewards

if __name__ == "__main__":
    actor = Actors(agents=8)
    critic = Critic()
    environment = Environment(actor=actor, critic=critic, agents=8, batch=2048)
    optimizer = torch.optim.Adam([
        {'params': actor .parameters(), 'lr': 0.0001},
        {'params': critic.parameters(), 'lr': 0.0001}
    ])
    lossfn = torch.nn.MSELoss()
    epochs = 100000

    for epoch in (tbar:=tqdm.tqdm(range(1,epochs))):
        optimizer.zero_grad()
        
        episode = [environment.step() for _ in range(30)]
        rewards = discounted_reward(episode, .99, "cuda:0")
        logprbs = [step["logprb"] for step in episode]

        loss = sum([-logprb*reward for logprb,reward in zip(logprbs,rewards)]).mean()

        loss.backward()
        optimizer.step()

        environment.reset()#torch.rand((3098,)) < 0.1)

        tbar.set_description(f"{loss.item():2.3f}")

        if epoch % 100 == 0: torch.save({"modelsd":policy.state_dict()}, "engine.pkl")




    

