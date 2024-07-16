from agents import Options, Agent, agent
import utils, click, torch, copy

class MLPOFA(Agent, torch.nn.Module):
    """ 
        Critic: a NN that outputs one value per agent.
        Actor : a NN that output actions per agent.

        Both Actor and Critic see all agent's observation as single flat tensor.
        All agents have the same set of parameters.
    """

    def __init__(self, trainer, hidden_size, layers, shared, actor_init_gain, critic_init_gain):
        torch.nn.Module.__init__(self)

        agents           = trainer.environment.agents
        observation_size = trainer.environment.get_observation_size()
        action_size      = trainer.environment.get_action_size()
        device           = trainer.environment.device

        # actor layers
        self.actor_obs2emb = utils.layer_init(torch.nn.Linear(observation_size*agents, hidden_size, device=device))
        self.actor_hidden  = torch.nn.ModuleList([l for _ in range(layers) for l in [utils.layer_init(torch.nn.Linear(hidden_size, hidden_size, device=device)), torch.nn.Tanh()]])
        self.actor_hid2act = utils.layer_init(torch.nn.Linear(hidden_size, action_size, device=device), std=actor_init_gain)

        # critic layers
        self.critic_obs2emb = self.actor_obs2emb if shared else copy.deepcopy(self.actor_obs2emb)
        self.critic_hidden  = self.actor_hidden  if shared else copy.deepcopy(self.actor_hidden)
        self.critic_hid2val = utils.layer_init(torch.nn.Linear(hidden_size, 1, bias=False, device=device), std=critic_init_gain)

        Agent.__init__(
            self,
            critic = torch.nn.Sequential(
                torch.nn.Flatten(1,2),
                utils.Lambda(lambda x:x.unsqueeze(1).repeat(1,agents,1)),

                self.critic_obs2emb,
                torch.nn.Tanh(),
                *self.critic_hidden,
                
                self.critic_hid2val,
                utils.Lambda(lambda x:x.squeeze(-1)),
            ),
            actor = torch.nn.Sequential(
                torch.nn.Flatten(1,2),
                utils.Lambda(lambda x:x.unsqueeze(1).repeat(1,agents,1)),

                self.actor_obs2emb,
                torch.nn.Tanh(),
                *self.actor_hidden,
                
                self.actor_hid2act,
                torch.nn.Tanh(),
            )
        )

        self.actor.logstd = torch.nn.Parameter(torch.zeros(1, action_size).to(device))


@agent.group(invoke_without_command=True)
@Options.mlp
@click.pass_obj
def mlp_ofa(trainer, hidden_size, layers, shared, state_dict_path, compile, actor_init_gain, critic_init_gain):
    compiler = torch.compile if compile else lambda x:x
    trainer.set_agent(
        compiler(MLPOFA(
            trainer          = trainer          ,
            shared           = shared           ,
            layers           = layers           ,
            hidden_size      = hidden_size      ,
            actor_init_gain  = actor_init_gain  ,
            critic_init_gain = critic_init_gain
        ))
    )


    if state_dict_path: trainer.agent.load_state_dict(torch.load(state_dict_path)["agentsd"])
