import random, numpy, tqdm, torch

def seed_everything(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def layer_init(layer, std=numpy.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Environment:
    def __init__(self, envs, device="cuda:0"):
        self.envs, self.device = envs, device
        self.reset()
        self.act2mv = torch.tensor([[+1,+1],[-1,+1],[+1,-1],[-1,-1],[+0,1],[+0,-1],[+1,+0],[-1,+0]],dtype=torch.float32,device=device)

    def reset(self):
        self.observation = torch.rand((self.envs,2), device=self.device)*20-10

    def step(self, action):
        self.observation += self.act2mv[action]

    def reward(self):
        return (self.observation[:,0] >= 5).logical_and\
               (self.observation[:,1] >= 5).logical_and\
               (self.observation[:,0] <= 8).logical_and\
               (self.observation[:,1] <= 8).float()

class Storage:
    def __init__(self, envs, steps, device):
        self.envs, self.steps, self.device = envs, steps, device
        self.reset()

    def reset(self):
        self.observations = torch.zeros((self.envs, self.steps, 2), dtype=torch.float32).to(self.device)
        self.actions      = torch.zeros((self.envs, self.steps)   , dtype=torch.uint8  ).to(self.device)
        self.logprobs     = torch.zeros((self.envs, self.steps)   , dtype=torch.float32).to(self.device)
        self.entropies    = torch.zeros((self.envs, self.steps)   , dtype=torch.float32).to(self.device)
        self.rewards      = torch.zeros((self.envs, self.steps)   , dtype=torch.float32).to(self.device)
        self.values       = torch.zeros((self.envs, self.steps)   , dtype=torch.float32).to(self.device)


    def __setitem__(self, index, value):
        if "observation" in value: self.observations[:, index] = value["observation"]
        if "action"      in value: self.actions     [:, index] = value["action"]
        if "logprob"     in value: self.logprobs    [:, index] = value["logprob"]
        if "entropy"     in value: self.entropies   [:, index] = value["entropy"]
        if "reward"      in value: self.rewards     [:, index] = value["reward"]
        if "value"       in value: self.values      [:, index] = value["value"]

    def __getitem__(self, index):
        return {
            "observation" : self.observations[index],
            "action"      : self.actions     [index],
            "logprob"     : self.logprobs    [index],
            "entropy"     : self.entropies   [index],
            "reward"      : self.rewards     [index],
            "value"       : self.values      [index]
        }


class Agent(torch.nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = torch.nn.Sequential(
            layer_init(torch.nn.Linear(2, 64)),
            torch.nn.ReLU(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.ReLU(),
            layer_init(torch.nn.Linear(64, 1)),
        )
        self.actor = torch.nn.Sequential(
            layer_init(torch.nn.Linear(2, 64)),
            torch.nn.ReLU(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.ReLU(),
            layer_init(torch.nn.Linear(64, 8)),
        )

    def get_value(self, env):
        return {
            "value" : self.critic(env).flatten()
        }

    def get_action(self, env, action=None):
        logits = self.actor(env)
        probs  = torch.distributions.Categorical(logits=logits)
        action = probs.sample() if action is None else action
        return {
            "action" : action,
            "logits"  : logits,
            "logprob" : probs.log_prob(action), 
            "entropy" : probs.entropy(),
            "probs"   : probs
        }

    def get_action_and_value(self, env, action=None):
        return self.get_action(env, action) | self.get_value(env)


def main(
            seed          = 42,
            device        = "cuda:0",
            epochs        = 5,
            batchsize     = 5,
            updates       = 10000,
            clipcoef      = .2,
            entcoef       = .01,
            vfcoef        = .5,
            max_grad_norm = .5,
            envs          = 32,
            steps         = 100,
            gaelambda     = .95,
            gamma         = .99,
            lr            = 1e-4,
            tmax          = 1000,
            lrmin         = 1e-5,
            etc           = 100
        ):

    seed_everything(seed)
    agent       = Agent().to(device)
    optimizer   = torch.optim.Adam(agent.parameters(), lr=lr)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tmax, eta_min=lrmin, last_epoch=-1)
    storage     = Storage(envs=envs, steps=steps, device=device)
    environment = Environment(envs=envs, device=device)

    for update in (tbar:=tqdm.tqdm(range(1, updates + 1))):
        for step in range(0, steps):
        
           with torch.no_grad():
               result = agent.get_action_and_value(environment.observation)
        
           storage[step] = {
               "observation" : environment.observation,
               "action"      : result["action"],
               "logprob"     : result["logprob"],
               "value"       : result["value"],
               "entropy"     : result["entropy"]
           }
           environment.step(result["action"])
           storage[step] = {"reward" : environment.reward()}


        returns = None
        with torch.no_grad():
            value = agent.get_value(environment.observation)["value"]
            advantages = torch.zeros_like(storage.rewards)
            lastgaelam = 0
            for t in reversed(range(steps)):
                value = value if t == steps - 1 else storage.values[:,t + 1]
                delta = storage.rewards[:,t] + gamma * value - storage.values[:,t]
                advantages[:,t] = lastgaelam = delta + gamma * gaelambda * lastgaelam
            returns = advantages + storage.values


                    
        for epoch in range(epochs):
            permutation  = torch.randperm(envs * steps, device=device)
            observations = storage.observations.view(envs * steps, 2)[permutation].view(batchsize, -1, 2)
            actions      = storage.actions     .view(envs * steps   )[permutation].view(batchsize, -1)
            logprobs     = storage.logprobs    .view(envs * steps   )[permutation].view(batchsize, -1)
            values       = storage.values      .view(envs * steps   )[permutation].view(batchsize, -1)
            advantages   = advantages          .view(envs * steps   )[permutation].view(batchsize, -1)
            returns      = returns             .view(envs * steps   )[permutation].view(batchsize, -1)


            for bobs,bact,blpr,badv,bret,bval in zip(observations, actions, logprobs, advantages, returns, values):
                result = agent.get_action_and_value(bobs, bact)
                logratio = result["logprob"] - blpr
                ratio    = logratio.exp()
                badv = (badv - badv.mean()) / (badv.std() + 1e-8)

                # Policy loss
                ploss1 = -badv * ratio
                ploss2 = -badv * torch.clamp(ratio, 1 - clipcoef, 1 + clipcoef)
                ploss = torch.max(ploss1, ploss2).mean()
                
                # Value loss
                v_loss_unclipped = (result["value"] - bret) ** 2
                v_clipped = bval + torch.clamp(result["value"] - bval, clipcoef, clipcoef)
                v_loss_clipped = (v_clipped - bret) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                vloss = 0.5 * v_loss_max.mean()

                # Entropy loss
                eloss = result["entropy"].mean()
                
                # Aggregated loss
                loss =  (-entcoef * eloss) + (ploss) + (vloss * vfcoef)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                tbar.set_description(f"i:{update: <5}, lr:{scheduler.get_last_lr()[0]:.6f}, vls:{vloss:7.6f}, pls:{ploss:10.6f}, els:{eloss:7.6f}, rew:{storage.rewards.mean():7.6f}")

            scheduler.step()

        if update % etc == 0: torch.save({"agentsd" : agent.state_dict()}, "agent.pkl")

if __name__ == "__main__":

    main(
        device="cuda:0",
        envs = 256,
        steps = 50,
        batchsize=16,
        entcoef=.1,
        lrmin=1e-5,
        lr=1e-5
    )
