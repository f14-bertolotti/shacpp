from algorithms import Algorithm, algorithm
from optimizers import add_adam_command, add_sgd_command
from schedulers import add_constant_command, add_cosine_command
from algorithms.shac import loss
import copy, time, click, torch

class SHAC(Algorithm):
    def __init__(self, 
        trainer, 
        max_grad_norm = .5  ,
        epochs        = 64  ,
        batch_size    = 256 ,
        target_alpha  = .5  ,
    ): 
        self.trainer       = trainer
        self.max_grad_norm = max_grad_norm
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.target_alpha  = target_alpha
        self.start_time    = time.time()

    def set_critic_optimizer (self, value): self.critic_optimizer  = value
    def set_critic_scheduler (self, value): self.critic_scheduler  = value
    def set_actor_optimizer  (self, value): self.actor_optimizer   = value
    def set_actor_scheduler  (self, value): self.actor_scheduler   = value
    def set_trajectory(self, value): self.trajectory = value

    def start(self): 
        self.target_agent = copy.deepcopy(self.trainer.agent)
        self.agent        = self.trainer.agent

    def end  (self): pass

    def step (self, episode):

        for callback in self.trainer.callbacks: callback.start_episode(locals())

        trajectories = self.trajectory(
            agent        = self.agent,
            target_agent = self.target_agent,
            environment  = self.trainer.environment,
        )

        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                trajectories["observations" ].flatten(0,1).detach().clone(),
                trajectories["target_values"].flatten(0,1).detach().clone(),
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = self.batch_size,
            shuffle    = True
        )
        
        result = []
        for epoch in range(0, self.epochs+1):

            for callback in self.trainer.callbacks: callback.start_epoch(locals())
            for step, (observations, target_values) in enumerate(dataloader, 1):

                for callback in self.trainer.callbacks: callback.start_step(locals())
                self.critic_optimizer.zero_grad()

                values = self.agent.get_value(observations)["values"]
                lossval = loss(values = values, target_values = target_values)
                lossval.backward()

                for callback in self.trainer.callbacks: callback.before_update(locals())

                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            
                self.critic_optimizer.step()

                for callback in self.trainer.callbacks: callback.end_step(locals())
        
            self.critic_scheduler.step()
            for callback in self.trainer.callbacks: callback.end_epoch(locals())

        with torch.no_grad():
            for param, param_targ in zip(self.agent.critic.parameters(), self.target_agent.critic.parameters()):
                param_targ.data.mul_(self.target_alpha)
                param_targ.data.add_((1. - self.target_alpha) * param.data)

        for callback in self.trainer.callbacks: callback.end_episode(locals())


@algorithm.group(invoke_without_command=True)
@click.option("--batch-size"    , "batch_size"    , type=int   , default=256 )
@click.option("--epochs"        , "epochs"        , type=int   , default=4   )
@click.option("--max-grad-norm" , "max_grad_norm" , type=float , default=.5  )
@click.option("--target-alpha"  , "target_alpha"  , type=float , default=0   )
@click.pass_obj
def shac(trainer, batch_size, epochs, max_grad_norm, target_alpha):
    if not hasattr(trainer, "algorithm"):
        trainer.set_algorithm(
            SHAC(
                trainer       = trainer       ,
                batch_size    = batch_size    ,
                epochs        = epochs        ,
                max_grad_norm = max_grad_norm ,
                target_alpha  = target_alpha
            )
        )

@shac.group()
def critic_optimizer(): pass
add_adam_command(critic_optimizer, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.agent.critic, attrname="set_critic_optimizer")
add_sgd_command (critic_optimizer, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.agent.critic, attrname="set_critic_optimizer")

@shac.group()
def critic_scheduler(): pass
add_cosine_command  (critic_scheduler, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.algorithm, attrname="set_critic_scheduler")
add_constant_command(critic_scheduler, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.algorithm, attrname="set_critic_scheduler")

@shac.group()
def actor_optimizer(): pass
add_adam_command(actor_optimizer, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.agent.actor, attrname="set_actor_optimizer")
add_sgd_command (actor_optimizer, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.agent.actor, attrname="set_actor_optimizer")

@shac.group()
def actor_scheduler(): pass
add_cosine_command  (actor_scheduler, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.algorithm, attrname="set_actor_scheduler")
add_constant_command(actor_scheduler, srcnav=lambda x:x.algorithm, tgtnav=lambda x:x.algorithm, attrname="set_actor_scheduler")


