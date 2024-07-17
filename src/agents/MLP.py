from agents import Options, Agent, agent
import utils, click, torch, copy, nn

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

        self.critic = nn.MLP(
            activation = "Tanh",
            input_size = observation_size*agents,
            hidden_size = hidden_size,
            output_size = hidden_size,
            layers      = layers,
            dropout     = 0
        )

        self.actor = self.critic if shared else copy.deepcopy(self.critic)

        Agent.__init__(
            self,
            critic = torch.nn.Sequential(
                torch.nn.Flatten(1,2),
                utils.Lambda(lambda x:x.unsqueeze(1).repeat(1,agents,1)),
                utils.Lambda(lambda x:x.unsqueeze(-2)),
                self.critic,
                torch.nn.Tanh(),
                torch.nn.Dropout(0),
                utils.layer_init(utils.MultiLinear(agents , hidden_size , 1 , bias=False , requires_grad=True , device=device), std=actor_init_gain),
                utils.Lambda(lambda x:x.squeeze(-2).squeeze(-1)),
            ),
            actor = torch.nn.Sequential(
                torch.nn.Flatten(1,2),
                utils.Lambda(lambda x:x.unsqueeze(1).repeat(1,agents,1)),
                utils.Lambda(lambda x:x.unsqueeze(-2)),
                self.actor,
                torch.nn.Tanh(),
                torch.nn.Dropout(0),
                utils.layer_init(utils.MultiLinear(agents, hidden_size, action_size, bias=False, requires_grad=True, device=device), std=critic_init_gain),
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
