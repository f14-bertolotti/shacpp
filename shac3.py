
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy, torch, copy, time, os

def layer_init(layer, std=1.141, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x): return self.func(x)

class CriticMLP(torch.nn.Module):
    def __init__(self, obs_dim, device='cuda:0'):
        super(CriticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] +  [128, 64, 32] + [1]

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(layer_init(torch.nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(torch.nn.ELU())
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = torch.nn.Sequential(*modules).to(device)
    
        self.obs_dim = obs_dim

        print(self.critic)

    def forward(self, observations):
        return self.critic(observations.view(observations.size(0),-1))

class ActorStochasticMLP(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + [128,64,32] + [action_dim]

        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(torch.nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(torch.nn.ELU())
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))
            else:
                modules.append(torch.nn.Identity())
            
        self.mu_net = torch.nn.Sequential(*modules).to(device)

        logstd = -1.0

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.mu_net)
        print(self.logstd)
    
    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic = False):
        obs = obs.view(obs.size(0),-1)
        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp() # (num_actions)
            # eps = torch.randn((*obs.shape[:-1], std.shape[-1])).to(self.device)
            # sample = mu + eps * std
            dist = torch.distributions.Normal(mu, std)
            sample = dist.rsample()
            return sample.view(sample.size(0),9,2)
    
    def forward_with_dist(self, obs, deterministic = False):
        mu = self.mu_net(obs)
        std = self.logstd.exp() # (num_actions)

        if deterministic:
            return mu, mu, std
        else:
            dist = torch.distributions.Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std
        
    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = torch.distributions.Normal(mu, std)

        return dist.log_prob(actions)

class Environment:
    def __init__(self, envs, agents, device="cuda:0"):
        self.envs, self.agents, self.device = envs, agents, device
        self.reset()
        self.points = torch.tensor([[0,0]], dtype=torch.float32, device=device)

    def reset(self, percentage=None):
        if percentage is not None:
            mask = torch.randperm(self.envs) < int(self.envs * percentage)
            self.observation = self.observation.detach()
            self.observation[mask] = torch.rand((int(self.envs * percentage),self.agents,2), requires_grad=True, device=self.device)*20-10
        else: 
            self.observation = torch.rand((self.envs,self.agents,2), requires_grad=True, device=self.device)*20-10
        return self.observation

    def step(self, observation, action):
        return observation + action, self.reward(), torch.tensor([False]*self.envs,device=self.device)

    def reward(self):
        return -torch.cdist(self.observation, torch.zeros((-5,10),device=self.device)).mean(-1).mean(-1)
        dists = torch.cdist(self.points, self.observation)
        return (dists.min(-1).values  < 3.333).float().mean(-1) - \
               (dists.min(-1).values >= 3.333).float().mean(-1)

    def clear_grad(self):
       pass 


class SHAC:
    def __init__(
            self, 
            agents = 9,
            steps = 32,
            envs = 64,
            epochs = 100000,
            critic_lr = 2e-3,
            actor_lr = 2e-3,
            batch_size = 128,
            critic_epochs = 4,
            betas = [0.7, 0.95],
            device="cuda:0"
        ):
        self.env = Environment(envs, agents)


        self.num_envs = envs         
        self.num_obs = agents*2
        self.num_actions = agents*2
        self.max_episode_length = steps
        self.device = device

        self.gamma = .99
        
        self.critic_method = 'td-lambda'
        if self.critic_method == 'td-lambda':
            self.lam = .95

        self.steps_num = steps
        self.max_epochs = epochs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_schedule ='linear'
        
        self.target_critic_alpha = .4

        self.obs_rms = RunningMeanStd(shape = (self.num_obs), device = self.device)
        self.ret_rms = None

        self.rew_scale = 1.0

        self.critic_iterations = critic_epochs
        self.batch_size = batch_size

        self.truncate_grad = True
        self.grad_norm = 1.0
        

        # create actor critic network
        self.actor = ActorStochasticMLP(9*2,9*2) 
        self.critic = CriticMLP(9*2)
        self.target_critic = copy.deepcopy(self.critic)
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
    
        # initialize optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas = betas, lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), betas = betas, lr = self.critic_lr)

        # replay buffer
        self.obs_buf = torch.zeros((self.steps_num, self.num_envs, agents, 2), dtype = torch.float32, device = self.device)
        self.rew_buf = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.done_mask = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.next_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.target_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.ret = torch.zeros((self.num_envs), dtype = torch.float32, device = self.device)

        # for kl divergence computing
        self.old_mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.old_sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = torch.int)
        self.best_policy_loss = numpy.inf
        self.actor_loss = numpy.inf
        self.value_loss = numpy.inf
        
    def compute_actor_loss(self, deterministic = False):
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        
        actor_loss = torch.tensor(0., dtype = torch.float32, device = self.device)

        with torch.no_grad():
            obs_rms = copy.deepcopy(self.obs_rms)
                

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.reset(.1)
        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                self.obs_rms.update(obs)
            # normalize the current obs
            obs = obs_rms.normalize(obs)
        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            actions = self.actor(obs)

            obs, rew, done = self.env.step(obs, actions)
            
            with torch.no_grad():
                raw_rew = rew.clone()
            
            # scale the reward
            rew = rew * self.rew_scale
            
            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)


            self.episode_length += 1
        
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            next_values[i + 1] = self.target_critic(obs).squeeze(-1)

            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError
            
            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
            else:
                # terminate all envs at the end of optimization iteration
                actor_loss = actor_loss + (- rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()
        
            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.
            rew_acc[i + 1, done_env_ids] = 0.

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.
                self.next_values[i] = next_values[i + 1].clone()

        actor_loss /= self.steps_num * self.num_envs

            
        self.actor_loss = actor_loss.detach().cpu().item()
            
        self.step_count += self.steps_num * self.num_envs

        return actor_loss
    
    
    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            Bi = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            lam = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (1. - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError
            
    def compute_critic_loss(self, obs, val):
        predicted_values = self.critic(obs).squeeze(-1)
        target_values = val
        critic_loss = ((predicted_values - target_values) ** 2).mean()

        return critic_loss

    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    def train(self):
        self.start_time = time.time()


        # initializations
        self.initialize_env()
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = torch.int)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        
        def actor_closure():
            self.actor_optimizer.zero_grad()

            actor_loss = self.compute_actor_loss()
            actor_loss.backward()

            with torch.no_grad():
                self.grad_norm_before_clip = grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = grad_norm(self.actor.parameters()) 
                
                # sanity check
                if torch.isnan(self.grad_norm_before_clip) or self.grad_norm_before_clip > 1000000.:
                    print('NaN gradient')
                    raise ValueError

            return actor_loss

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
            else:
                lr = self.actor_lr

            # train actor
            self.actor_optimizer.step(actor_closure).detach().item()

            # train critic
            # prepare dataset
            with torch.no_grad():
                self.compute_target_values()
                dataset = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        self.obs_buf.view(self.num_envs*self.steps_num,self.obs_buf.size(-2),self.obs_buf.size(-1)),
                        self.target_values.view(-1),
                    ), 
                    batch_size = self.batch_size,
                    drop_last  = False,
                    shuffle    = True
                )

            self.value_loss = 0.
            for j in range(self.critic_iterations):
                total_critic_loss = 0.
                batch_cnt = 0
                for i,(obs,val) in enumerate(dataset):
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(obs,val)
                    training_critic_loss.backward()
                    
                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1
                
                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, self.value_loss), end='\r')

            self.iter_count += 1
            
            print('iter {}, rew:{}, als:{}, vls {}, grad norm before clip {:.2f}, grad norm after clip {:.2f}'.format(\
                    self.iter_count, self.rew_buf.mean(), self.actor_loss, self.value_loss, self.grad_norm_before_clip.item(), self.grad_norm_after_clip.item()))

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)


        self.run(self.num_envs)

if __name__ == "__main__":
    SHAC().train()
