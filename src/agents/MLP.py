from agents import Options, Agent, agent
import utils, click, torch, copy

class MLP(Agent, torch.nn.Module):
    """ 
        Critic: a NN that outputs one value per agent.
        Actor : a NN that output actions per agent.

        Both Actor and Critic see all agent's observation as single flat tensor.
        Different agents have different set of parameters.
    """

    def __init__(self, trainer, hidden_size, layers, shared, actor_init_gain, critic_init_gain):
        torch.nn.Module.__init__(self)

        agents           = trainer.environment.agents
        observation_size = trainer.environment.get_observation_size()
        action_size      = trainer.environment.get_action_size()
        device           = trainer.environment.device

        # actor layers
        self.actor_first  = utils.layer_init(utils.MultiLinear(agents, observation_size*agents, hidden_size, bias=True, requires_grad=True, device=device))
        self.actor_hidden = torch.nn.ModuleList([l for _ in range(layers) for l in [utils.layer_init(utils.MultiLinear(agents, hidden_size, hidden_size, bias=True, requires_grad=True, device=device)),torch.nn.Tanh()]])
        self.actor_last   = utils.layer_init(utils.MultiLinear(agents , hidden_size , action_size , bias=False , requires_grad=True , device=device), std=actor_init_gain)

        # critic layers
        self.critic_first  = self.actor_first  if shared else copy.deepcopy(self.actor_first )
        self.critic_hidden = self.actor_hidden if shared else copy.deepcopy(self.actor_hidden)
        self.critic_last   = utils.layer_init(utils.MultiLinear(agents, hidden_size, 1, bias=False, requires_grad=True, device=device), std=critic_init_gain)

        Agent.__init__(
            self,
            critic = torch.nn.Sequential(
                torch.nn.Flatten(1,2),
                utils.Lambda(lambda x:x.unsqueeze(1).repeat(1,agents,1)),
                utils.Lambda(lambda x:x.unsqueeze(-2)),
                self.critic_first,
                torch.nn.Tanh(),
                *self.critic_hidden,
                self.critic_last,
                utils.Lambda(lambda x:x.squeeze(-2).squeeze(-1)),
            ),
            actor = torch.nn.Sequential(
                torch.nn.Flatten(1,2),
                utils.Lambda(lambda x:x.unsqueeze(1).repeat(1,agents,1)),
                utils.Lambda(lambda x:x.unsqueeze(-2)),
                self.actor_first,
                torch.nn.Tanh(),
                *self.actor_hidden,
                self.actor_last,
                utils.Lambda(lambda x:x.squeeze(-2)),
                torch.nn.Tanh(),
            )
        )


        self.actor.logstd = torch.nn.Parameter(torch.zeros(1, action_size).to(device))


@agent.group(invoke_without_command=True)
@Options.mlp
@click.pass_obj
def mlp(trainer, hidden_size, layers, shared, state_dict_path, compile, actor_init_gain, critic_init_gain):
    compiler = torch.compile if compile else lambda x:x
    trainer.set_agent(
        compiler(MLP(
            trainer          = trainer          ,
            shared           = shared           ,
            layers           = layers           ,
            hidden_size      = hidden_size      ,
            actor_init_gain  = actor_init_gain  ,
            critic_init_gain = critic_init_gain
        ))
    )


    if state_dict_path: trainer.agent.load_state_dict(torch.load(state_dict_path)["agentsd"])
