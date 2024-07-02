from agents import agent
import utils, click, torch, copy

class TransformerAgent(torch.nn.Module):
    def __init__(
            self, 
            observation_size = 2,
            action_size      = 2,
            layers           = 3,
            embedding_size   = 64,
            feedforward_size = 256,
            heads            = 2,
            shared           = True,
            activation       = "gelu",
            device           = "cuda:0"
        ):
        super().__init__()
        self.device = device


        self.actor_position = utils.layer_init(torch.nn.Linear(observation_size, embedding_size, device=device))
        self.actor_pos_embedding = torch.nn.Embedding(256, 64).to(device)
        self.actor_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model         = embedding_size,
                nhead           = heads,
                dim_feedforward = feedforward_size,
                activation      = activation,
                device          = device,
                batch_first     = True
            ), 
            num_layers = layers
        )
        self.critic_position      = self.actor_position      if shared else copy.deepcopy(self.actor_position)
        self.critic_encoder       = self.actor_encoder       if shared else copy.deepcopy(self.actor_encoder)
        self.critic_pos_embedding = self.actor_pos_embedding if shared else copy.deepcopy(self.actor_pos_embedding)

        self.critic = torch.nn.Sequential(
            self.actor_position,
            utils.Lambda(lambda x:x + self.critic_pos_embedding(torch.arange(x.size(1), device=device, dtype=torch.long))),
            self.actor_encoder,
            utils.layer_init(torch.nn.Linear(embedding_size, 1, device=device), std=1),
        )
        
        self.actor_mean = torch.nn.Sequential(
            self.critic_position,
            utils.Lambda(lambda x:x + self.actor_pos_embedding(torch.arange(x.size(1), device=device, dtype=torch.long))),
            self.critic_encoder,
            utils.layer_init(torch.nn.Linear(embedding_size, action_size, device=device),std=.01),
            torch.nn.Tanh(),
        )

        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, action_size)).to(device)

    def get_value(self, x):
        flag = type(x) == list
        x = torch.stack(x).transpose(0,1) if type(x) == list else x
        return {"values" : self.critic(x).transpose(0,1) if flag else self.critic(x)}

    def get_action(self, x, action=None):
        flag = type(x) == list
        x = torch.stack(x).transpose(0,1) if flag else x
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = torch.clamp(probs.rsample(),-1,1)
        
        return {
            "logits"   : action_mean,
            "actions"  : action.transpose(0,1) if flag else action,
            "logprobs" : probs.log_prob(action).sum(-1).transpose(0,1) if flag else probs.log_prob(action).sum(-1),
            "entropy"  : probs.entropy().sum(-1).transpose(0,1) if flag else probs.entropy().sum(-1)
        }

    def get_action_and_value(self, observation, action=None):
        result = self.get_action(observation, action=action) | self.get_value(observation)
        return result["actions"], result["logprobs"], result["entropy"], result["values"]




@agent.group(invoke_without_command=True)
@click.option("--observation-size" , "observation_size" , type=int , default=2)
@click.option("--action-size"      , "action_size"      , type=int , default=2)
@click.option("--layers"           , "layers"           , type=int , default=3)
@click.option("--embedding-size"   , "embedding_size"   , type=int , default=64)
@click.option("--feedforward_size" , "feedforward_size" , type=int , default=256)
@click.option("--heads"            , "heads"            , type=int , default=2)
@click.option("--activation"       , "activation"       , type=str , default="gelu")
@click.option("--shared"           , "shared"           , type=bool, default=True)
@click.option("--device"           , "device"           , type=str , default="cuda:0")
@click.option("--state-dict-path"  , "state_dict_path"  , type=click.Path(), default=None)
@click.pass_obj
def transformer_agent(trainer, observation_size, action_size, layers, embedding_size, feedforward_size, heads, activation, shared, device, state_dict_path):
    trainer.set_agent(
        TransformerAgent(
            observation_size = observation_size,
            action_size      = action_size,
            layers           = layers,
            embedding_size   = embedding_size,
            feedforward_size = feedforward_size,
            heads            = heads,
            activation       = activation,
            shared           = shared,
            device           = device
        )
    )

    if state_dict_path: trainer.agent.load_state_dict(torch.load(state_dict_path)["agentsd"])
