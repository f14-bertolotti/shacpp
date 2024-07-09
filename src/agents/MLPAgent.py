from agents import agent
import utils, click, torch, copy

class MLPAgent(torch.nn.Module):
    """ 
        Critic: a NN that outputs one value per agent.
        Actor : a NN that output actions per agent.

        Both Actor and Critic see all agent's observation as single flat tensor.
        Different agents have different set of parameters.
    """
    def __init__(self, trainer, hidden_size, layers, shared):
        super().__init__()
        self.agents      = trainer.environment.agents
        observation_size = trainer.environment.get_observation_size()
        action_size      = trainer.environment.get_action_size()
        device           = trainer.environment.device

        # base architecture
        self.base = torch.nn.Sequential(
            torch.nn.Flatten(1,2),
            utils.layer_init(torch.nn.Linear(observation_size * self.agents, hidden_size, device=device)),
            torch.nn.Tanh(),
            *[utils.layer_init(torch.nn.Linear(hidden_size, hidden_size, device=device)) for _ in range(layers)],
            torch.nn.Tanh(),
        )
        
        # share the base architecture in case shared=True
        self.critic_base = self.base
        self.actor_base  = self.base if shared else copy.deepcopy(self.base)

        # critic architecture
        self.critic = torch.nn.Sequential(
            self.critic_base,
            utils.layer_init(torch.nn.Linear(hidden_size, self.agents, device=device), std=1.0),
        )

        # actor architecture
        self.actor = torch.nn.Sequential(
            self.actor_base,
            utils.layer_init(torch.nn.Linear(hidden_size, action_size * self.agents, device=device), std=0.01),
            torch.nn.Tanh(),
            utils.Lambda(lambda x:x.view(x.size(0),self.agents,action_size)),
        )
        self.actor.logstd = torch.nn.Parameter(torch.zeros(1, action_size)).to(device)

    def get_value(self, x):
        return {"values" : self.critic(x)}

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std    = torch.exp(action_logstd)
        probs         = torch.distributions.normal.Normal(action_mean, action_std)

        if action is None: action = torch.clamp(probs.rsample(),-1,1)
        
        return {
            "logits"   : action_mean,
            "actions"  : action,
            "logprobs" : probs.log_prob(action).sum(-1),
            "entropy"  : probs.entropy().sum(-1)
        }

    def get_action_and_value(self, observation, action=None):
        return self.get_action(observation, action=action) | self.get_value(observation)


@agent.group(invoke_without_command=True)
@click.option("--hidden-size"     , "hidden_size"     , type=int          , default=64    , help="Hidden number of neurons.")
@click.option("--shared"          , "shared"          , type=bool         , default=False , help="True if the first layers should be shared between critic and actor.")
@click.option("--layers"          , "layers"          , type=int          , default=1     , help="Number of layer of the network.")
@click.option("--state-dict-path" , "state_dict_path" , type=click.Path() , default=None  , help="Path to the agent to be loaded.")
@click.option("--compile"         , "compile"         , type=bool         , default=True  , help="True if the agent should be compiled")
@click.pass_obj
def mlp_agent(trainer, hidden_size, layers, shared, state_dict_path, compile):
    compiler = torch.compile if compile else lambda x:x
    trainer.set_agent(
        compiler(MLPAgent(
            trainer     = trainer     ,
            shared      = shared      ,
            layers      = layers      ,
            hidden_size = hidden_size ,
        ))
    )


    if state_dict_path: trainer.agent.load_state_dict(torch.load(state_dict_path)["agentsd"])
