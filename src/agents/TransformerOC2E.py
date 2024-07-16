from agents import agent, Agent, Options 
import utils, click, torch, copy

class TransformerOC2E(Agent, torch.nn.Module):
    """
    Actor-Critic Architecture based on the transformer. 

    Architecture:
        - Each individual observation component is mapped to an embedding.
        - Each embedding is added to a positional embedding.
        - A LayerNorm layer is applied.
        - A Transformer Encoder is applied.
        - Actor maps to actions, Critic maps to value.

    """

    def __init__(
            self, 
            trainer,
            observation_size = 2      ,
            action_size      = 2      ,
            layers           = 3      ,
            embedding_size   = 64     ,
            feedforward_size = 256    ,
            heads            = 2      ,
            shared           = True   ,
            actor_init_gain  = 1.41   ,
            critic_init_gain = 1.41   ,
            activation       = "gelu"
        ):

        torch.nn.Module.__init__(self)

        # get environment settings 
        agents           = trainer.environment.agents
        observation_size = trainer.environment.get_observation_size()
        action_size      = trainer.environment.get_action_size()
        device           = trainer.environment.device

        # setup embedding parameters (obs embedding and obs positions) 
        self.actor_embedding     = torch.nn.Parameter(torch.normal(torch.zeros(observation_size, 1, embedding_size)         , torch.ones(observation_size, 1 , embedding_size)         * 0.2).to(device))
        self.actor_positions     = torch.nn.Parameter(torch.normal(torch.zeros(agents, observation_size, 1, embedding_size) , torch.ones(agents, observation_size, 1 , embedding_size) * 0.2).to(device))

        # setup transformer encoder
        self.actor_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                dim_feedforward = feedforward_size ,
                d_model         = embedding_size   ,
                activation      = activation       ,
                device          = device           ,
                nhead           = heads            ,
                batch_first     = True
            ), 
            num_layers = layers
        )

        # if shared is True, shares the parameters, otherwise share the architecture.
        self.critic_embedding = self.actor_embedding if shared else copy.deepcopy(self.actor_embedding)
        self.critic_positions = self.actor_positions if shared else copy.deepcopy(self.actor_positions)
        self.critic_encoder   = self.actor_encoder   if shared else copy.deepcopy(self.actor_encoder)

        # set critic and actor
        Agent.__init__(
            self,
            critic = torch.nn.Sequential(
                utils.Lambda(lambda x: x.unsqueeze(-1).unsqueeze(-1)),
                utils.Lambda(lambda x: torch.matmul(x,self.critic_embedding) + self.critic_positions),
                utils.Lambda(lambda x: x.flatten(1,3)),
                torch.nn.LayerNorm(embedding_size, device=device),
                self.critic_encoder,
                utils.Lambda(lambda x: x[:,::observation_size,:]),
                utils.layer_init(torch.nn.Linear(embedding_size, 1, device=device), std=actor_critic_gain),
                utils.Lambda(lambda x:x.squeeze(-1)),
            ),
            actor = torch.nn.Sequential(
                utils.Lambda(lambda x: x.unsqueeze(-1).unsqueeze(-1)),
                utils.Lambda(lambda x: torch.matmul(x,self.actor_embedding) + self.actor_positions),
                utils.Lambda(lambda x: x.flatten(1,3)),
                torch.nn.LayerNorm(embedding_size, device=device),
                self.actor_encoder,
                utils.Lambda(lambda x: x[:,::observation_size,:]),
                utils.layer_init(torch.nn.Linear(embedding_size, action_size, device=device),std=actor_init_gain),
                torch.nn.Tanh(),
            )
        )

        # set logstd
        self.actor.logstd = torch.nn.Parameter(torch.zeros(1, action_size).to(device))

@agent.group(invoke_without_command=True)
@Options.transformer
@click.pass_obj
def transformer_oc2e(trainer, layers, embedding_size, feedforward_size, heads, activation, shared, compile, state_dict_path, actor_init_gain, critic_init_gain):
    compiler = torch.compile if compile else lambda x:x

    trainer.set_agent(
        compiler(TransformerOC2E(
            trainer          = trainer          ,
            layers           = layers           ,
            embedding_size   = embedding_size   ,
            feedforward_size = feedforward_size ,
            heads            = heads            ,
            activation       = activation       ,
            shared           = shared           ,
            actor_init_gain  = actor_init_gain  ,
            critic_init_gain = critic_init_gain
 
        ))
    )

    if state_dict_path: trainer.agent.load_state_dict(torch.load(state_dict_path)["agentsd"])
