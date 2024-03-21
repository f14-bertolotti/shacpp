import tqdm, torch


class Policy(torch.nn.Module):

    def __init__(self, layers=3, hidden_size=128, device="cuda:0"):
        super().__init__()
        self.embedding = torch.nn.Linear(2, hidden_size, device=device)
        self.lins0 = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) for _ in range(layers)])
        self.lins1 = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size, device=device) for _ in range(layers)])
        self.lns  = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_size, device=device) for _ in range(layers)])
        self.logits = torch.nn.Linear(hidden_size, 4, device=device)

    def forward(self, state):
        embedding = self.embedding(state)
        for lin0, lin1, ln in zip(self.lins0, self.lins1, self.lns):
            embedding = ln(embedding + lin1(torch.nn.functional.relu(lin0(embedding))))
        return torch.nn.functional.softmax(self.logits(embedding), dim=-1)
        

class Environment:

    def __init__(self, policy, batch=128, device="cuda:0"):
       self.policy, self.batch, self.device = policy, batch, device
       self.actions = torch.tensor([[.5,0],[0,.5],[-.5,0],[0,-.5]], device=device, dtype=torch.float)
       self.reset()

    def reset(self):
        self.state  = torch.zeros(
            size   = (self.batch, 2),
            device = self.device,
            dtype  = torch.float
        ) 

    def step(self):
        policy = self.policy(self.state)
        policy = torch.distributions.Categorical(policy)
        action = policy.sample()
        logprb = policy.log_prob(action)
        self.state = self.state + self.actions[action]
        reward = self.reward()

        return {"state" : self.state, "logprb" : logprb, "reward" : reward}

    def reward(self):
        return (10 < self.state).logical_and(self.state < 15).prod(-1)

def discounted_reward(episode, gamma, device):
    R, rewards = torch.zeros_like(episode[0]["reward"], dtype=torch.float, device=device, requires_grad=True), []
    for step in episode[::-1]:
        R = step["reward"] + gamma * R
        rewards.insert(0,R)
    rewards = torch.stack(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    return rewards

if __name__ == "__main__":
    policy = Policy()
    environment = Environment(policy, batch=1024)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)

    for epoch in (tbar:=tqdm.tqdm(range(1000))):
        environment.reset()
        optimizer.zero_grad()

        episode = [environment.step() for _ in range(500)]
        rewards = discounted_reward(episode, .99, "cuda:0")
        logprbs = [step["logprb"] for step in episode]

        loss = sum([-logprb*reward for logprb,reward in zip(logprbs,rewards)]).mean()

        loss.backward()
        optimizer.step()

        tbar.set_description(f"{loss.item()}")

        if epoch % 100 == 0: torch.save({"modelsd":policy.state_dict()}, "engine.pkl")




    

