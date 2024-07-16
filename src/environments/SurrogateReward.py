from environments.rewards import Dummy
from environments import Environment
import torch



class SurrogateReward(Environment):

    def __init__(self, environment):

        super().__init__(
            rms    = environment.rms    ,
            envirs = environment.envirs ,
            agents = environment.agents ,
            device = environment.device
        )
        self.world = environment.world
        self.rewardnn = Dummy()
        self.dataset_size = 10000
        self.epochs = 50
        self.batch_size = 500
        self.data = torch.zeros(self.dataset_size , environment.get_observation_size() + environment.get_action_size() , dtype=torch.float32  , requires_grad=False, device=environment.device)
        self.rewards      = torch.zeros(self.dataset_size , dtype=torch.float32 , requires_grad=False, device=environment.device)
        self.mask         = torch.zeros(self.dataset_size , dtype=torch.bool    , requires_grad=False, device=environment.device)

    def set_optimizer(self, value): self.optimizer = value
    def set_scheduler(self, value): self.scheduler = value
    def set_reward_nn(self, value): self.rewardnn  = value

    def update_statistics(self, observations):
        if self.rms: self.rms.update(observations.view(-1,observations.size(-1)))

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.data   [self.mask],
                self.rewards[self.mask],
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = self.batch_size,
            shuffle    = True,
            drop_last  = True
        )

        print((self.rewards[self.mask] == 0).sum(),(self.rewards[self.mask] == 1).sum())
        #for epoch in range(self.epochs):
        while True:
            for obs, rew in dataloader:
                self.optimizer.zero_grad()
                res = self.rewardnn(obs)
                lss = ((res - rew)**2).mean()
                print((res-rew).abs().mean(), rew.isclose(res,atol=0.1).float().mean())
                lss.backward()
                self.optimizer.step()
            self.scheduler.step()
            if rew.isclose(res,atol=0.1).float().mean().item() > .9: break

    def pert(self, low:torch.Tensor, peak:torch.Tensor, high:torch.Tensor, lamb=8):
        r = high - low
        alpha = 1 + lamb * (peak - low) / r
        beta  = 1 + lamb * (high - peak) / r
        return low + torch.distributions.Beta(alpha, beta).sample() * r


    def step(self, oldobs, action):
        next_observation, reward, done, info = self.world.step(action.transpose(0,1))
        observation = torch.stack(next_observation).transpose(0,1)
        real_reward = torch.stack(reward          ).transpose(0,1)

        #mask = torch.randint(low=0,high=self.dataset_size,size=(observation.size(0)*observation.size(1),),device=self.device)
        mask = torch.clamp(self.pert(
            low  = torch.zeros(observation.size(0)*observation.size(1), dtype = torch.float32, device=self.device),
            peak = real_reward.flatten(0,1) * (self.dataset_size-1),
            high = torch.ones(observation.size(0)*observation.size(1), dtype  = torch.float32, device=self.device) * self.dataset_size
        ).round().to(torch.long),0,self.dataset_size-1)

        self.mask[mask] = True
        self.data[mask,:self.get_observation_size()] = oldobs.flatten(0,1).detach().clone()
        self.data[mask,-self.get_action_size():]     = action.flatten(0,1).detach().clone()
        self.rewards[mask] = real_reward.flatten(0,1).detach().clone()
        
        surrogate_reward = self.rewardnn(torch.cat([observation , action] , dim=2))

        return {
        "observation"      : observation      ,
        "real_reward"      : real_reward      ,
        "surrogate_reward" : surrogate_reward ,
        "reward"           : surrogate_reward ,
        "done"             : done             ,
        "info"             : info             ,
        }



