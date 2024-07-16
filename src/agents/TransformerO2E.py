from agents import Options, Agent, agent
import utils, click, torch, copy

class TransformerO2E(Agent, torch.nn.Module):
    def __init__(
            self, 
            trainer,
            observation_size = 2      ,
            action_size      = 2      ,
            layers           = 3      ,
            embedding_size   = 64     ,
            feedforward_size = 256    ,
            heads            = 2      ,
            memory_tokens    = 0      ,
            shared           = True   ,
            actor_init_gain   = 1.41  , 
            critic_init_gain = 1.41   ,
            activation       = "gelu"
        ):
        torch.nn.Module.__init__(self)

        # get environment settings 
        agents           = trainer.environment.agents
        observation_size = trainer.environment.get_observation_size()
        action_size      = trainer.environment.get_action_size()
        device           = trainer.environment.device

        # setup embedding parameters (obs embedding and obs embeddings) 
        self.actor_embedding = utils.layer_init(torch.nn.Linear(observation_size, embedding_size, device=device))
        self.actor_position  = torch.nn.Embedding(agents + memory_tokens, embedding_size, device=device)
        self.actor_memorytkn = torch.nn.Parameter(torch.normal(torch.zeros(1,memory_tokens, embedding_size), torch.ones(1,memory_tokens, embedding_size) * 0.2).to(device))

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
        self.critic_position  = self.actor_position  if shared else copy.deepcopy(self.actor_position) 
        self.critic_embedding = self.actor_embedding if shared else copy.deepcopy(self.actor_embedding)
        self.critic_encoder   = self.actor_encoder   if shared else copy.deepcopy(self.actor_encoder)
        self.critic_memorytkn = self.actor_memorytkn if shared else copy.deepcopy(self.actor_memorytkn)

        Agent.__init__(
            self,
            critic = torch.nn.Sequential(
                self.critic_embedding,
                utils.Lambda(lambda x:torch.cat([x,self.critic_memorytkn.repeat(x.size(0),1,1)], dim=1)),
                utils.Lambda(lambda x:x + self.critic_position(torch.arange(x.size(1), device=device, dtype=torch.long))),
                torch.nn.LayerNorm(embedding_size, device=device),
                self.critic_encoder,
                utils.layer_init(torch.nn.Linear(embedding_size, 1, device=device), std=critic_init_gain),
                utils.Lambda(lambda x:x[:,:agents,:]),
                utils.Lambda(lambda x:x.squeeze(-1)),
            ),
            actor = torch.nn.Sequential(
                self.actor_embedding,
                utils.Lambda(lambda x:torch.cat([x, self.actor_memorytkn.repeat(x.size(0),1,1)], dim=1)),
                utils.Lambda(lambda x:x + self.actor_position(torch.arange(x.size(1), device=device, dtype=torch.long))),
                torch.nn.LayerNorm(embedding_size, device=device),
                self.actor_encoder,
                utils.layer_init(torch.nn.Linear(embedding_size, action_size, device=device),std=actor_init_gain),
                utils.Lambda(lambda x:x[:,:agents,:]),
                torch.nn.Tanh(),
            )
        )

        self.actor.logstd = torch.nn.Parameter(torch.zeros(1, action_size).to(device))


@agent.group(invoke_without_command=True)
@Options.transformer
@click.option("--memory-tokens", "memory_tokens", type=int, default=0, help="memory tokens to be added")
@click.pass_obj
def transformer_o2e(trainer, layers, embedding_size, feedforward_size, heads, activation, shared, compile, state_dict_path, memory_tokens, actor_init_gain, critic_init_gain):
    compiler = torch.compile if compile else lambda x:x
    trainer.set_agent(
        compiler(TransformerO2E(
            trainer          = trainer          ,
            layers           = layers           ,
            embedding_size   = embedding_size   ,
            feedforward_size = feedforward_size ,
            heads            = heads            ,
            activation       = activation       ,
            memory_tokens    = memory_tokens    ,
            shared           = shared           ,
            actor_init_gain  = actor_init_gain  ,
            critic_init_gain = critic_init_gain
 
        ))
    )

    if state_dict_path: trainer.agent.load_state_dict(torch.load(state_dict_path)["agentsd"])
