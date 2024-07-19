from environments import environment, Environment
from nn import Dummy
import itertools, torch

@environment.group()
def proxied(): pass

class Proxify(Environment):
    """ This class takes an enviroment and replaces the normal reward with a trained one """

    def __init__(
            self, 
            trainer    ,
            environment, 
            dataset_size = 10000  ,
            batch_size   = 500    ,
            lamb         = 8      ,
            atol         = .1     ,
            threshold    = None   ,
            shuffle      = True   ,
            drop_last    = True   ,
            epochs       = None   ,
        ):
        self.trainer = trainer

        # setup environment variables
        super().__init__(
            rms    = environment.rms    ,
            envirs = environment.envirs ,
            agents = environment.agents ,
            device = environment.device ,
        )

        # setup world
        self.world = environment.world

        # setup proxify variables
        self.dataset_size = dataset_size
        self.batch_size   = batch_size
        self.threshold    = threshold
        self.lamb         = lamb
        self.atol         = atol
        self.epochs       = epochs
        self.shuffle      = shuffle
        self.drop_last    = drop_last

        # setup dummy network (for the LSP)
        self.rewardnn = Dummy()

        # setup dataset space
        self.data     = torch.zeros(self.dataset_size , environment.get_observation_size() + environment.get_action_size() , dtype=torch.float32, requires_grad=False, device=environment.device)
        self.rewards  = torch.zeros(self.dataset_size , dtype=torch.float32 , requires_grad=False, device=environment.device)
        self.mask     = torch.zeros(self.dataset_size , dtype=torch.bool    , requires_grad=False, device=environment.device)

        self.updates = 0

    def set_optimizer(self, value): self.optimizer = value
    def set_scheduler(self, value): self.scheduler = value
    def set_reward_nn(self, value): self.rewardnn  = value

    def update_statistics(self, observations):
        self.updates += 1

        """ this method performs also the training of the reward, other than computing running statistics """
        if self.rms: self.rms.update(observations.view(-1,observations.size(-1)))

        for callback in self.trainer.callbacks: callback.start_update_proxy(locals())

        # create the dataset for the reward network
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.data   [self.mask],
                self.rewards[self.mask],
            ),
            collate_fn = torch.utils.data.default_collate ,
            batch_size = self.batch_size                  ,
            shuffle    = self.shuffle                     ,
            drop_last  = self.drop_last
        )

        # train the reward network
        for epoch in itertools.count(0):
            
            for callback in self.trainer.callbacks: callback.start_epoch_proxy(locals())

            # max epoch is set, terminate when reached
            if self.epochs is not None and epoch >= self.epochs: break

            accuracies = []

            # training epoch
            for step, (obs, rew) in enumerate(dataloader,1):
                for callback in self.trainer.callbacks: callback.start_step_proxy(locals())

                self.optimizer.zero_grad()
                res  = self.rewardnn(obs)
                loss = ((res - rew)**2).mean()
                loss.backward()
                self.optimizer.step()
                
                # compute accuracy of the reward
                accuracies.append(rew.isclose(res,atol=self.atol))
                for callback in self.trainer.callbacks: callback.end_step_proxy(locals())

            self.scheduler.step()
            
            accuracy = torch.cat(accuracies, dim=0).float().mean().item()

            for callback in self.trainer.callbacks: callback.end_epoch_proxy(locals())
            
            # terminate if threshold is set and is reached
            if self.threshold is not None and accuracy > self.threshold: break

        for callback in self.trainer.callbacks: callback.end_update_proxy(locals())


    def pert(self, low:torch.Tensor, peak:torch.Tensor, high:torch.Tensor):
        """ pert distribution   : https://en.wikipedia.org/wiki/PERT_distribution 
            implementation from : https://stackoverflow.com/questions/68476485/random-values-from-a-pert-distribution-in-python """

        r = high - low
        alpha = 1 + self.lamb * (peak - low) / r
        beta  = 1 + self.lamb * (high - peak) / r
        return low + torch.distributions.Beta(alpha, beta).sample() * r

    def collect_data(self, newobs, oldobs, real_reward, action):
        """ collects data for training the reward function """

        # distribute samples according their reward
        # so that the dataset eventually become balanced
        updated = self.pert(
            low  = torch.zeros(newobs.size(0)*newobs.size(1) , dtype = torch.float32 , device=self.device)                         ,
            high = torch.ones (newobs.size(0)*newobs.size(1) , dtype = torch.float32 , device=self.device) * (self.dataset_size-1) ,
            peak = real_reward.flatten(0,1) / (real_reward.max()+1e-5) * (self.dataset_size-1),
        ).round().to(torch.long)

        # update the dataset
        self.mask    [updated                              ] = True
        self.data    [updated,:self.get_observation_size() ] = oldobs      .flatten (0,1).detach().clone()
        self.data    [updated,-self.get_action_size():     ] = action      .flatten (0,1).detach().clone()
        self.rewards [updated                              ] = real_reward .flatten (0,1).detach().clone()

    def step(self, action, oldobs=None):
        next_observation, reward, done, info = self.world.step(action.transpose(0,1))
        observation = torch.stack(next_observation).transpose(0,1)
        real_reward = torch.stack(reward          ).transpose(0,1)
        
        if oldobs is not None: self.collect_data(observation, oldobs, real_reward, action)

        proxy_reward = self.rewardnn(torch.cat([observation , action] , dim=2))

        return {
            "observation"  : observation  ,
            "real_reward"  : real_reward  ,
            "proxy_reward" : proxy_reward ,
            "reward"       : proxy_reward ,
            "done"         : done         ,
            "info"         : info         ,
        }



