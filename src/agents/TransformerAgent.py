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
        self.cov_var = torch.full(size=(observation_size,), fill_value=0.005, device=device)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_position = utils.layer_init(torch.nn.Linear(observation_size, embedding_size, device=device))
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
        self.critic_position = self.actor_position if shared else copy.deepcopy(self.actor_position)
        self.critic_encoder  = self.actor_encoder  if shared else copy.deepcopy(self.actor_encoder)

        self.critic = torch.nn.Sequential(
            self.actor_position,
            self.actor_encoder,
            utils.Lambda(lambda x:x.sum(-2)),
            utils.layer_init(torch.nn.Linear(embedding_size, 1, device=device), std=1),
        )
        
        self.actor = torch.nn.Sequential(
            self.critic_position,
            self.critic_encoder,
            utils.layer_init(torch.nn.Linear(embedding_size, action_size, device=device),std=.01),
            torch.nn.Tanh(),
            utils.Lambda(lambda x: x/30)
        )

    def get_value(self, x):
        return {"values" : self.critic(x).squeeze(-1).squeeze(-1)}

    def get_action(self, observation, action=None):
        logits = self.actor(observation)
        probs  = torch.distributions.MultivariateNormal(logits, self.cov_mat)
        action = probs.rsample() if action is None else action
        return {
            "logits"   : logits,
            "actions"  : action,
            "logprobs" : probs.log_prob(action).sum(-1),
            "entropy"  : probs.entropy()
        }

    def get_action_and_value(self, observation, action=None):
        return self.get_action(observation, action=action) | self.get_value(observation)

@agent.group(invoke_without_command=True)
@click.option("--observation_size" , "observation_size" , type=int , default=2)
@click.option("--action_size"      , "action_size"      , type=int , default=2)
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

    trainer.agent.load_state_dict(torch.load(state_dict_path)["agentsd"])
