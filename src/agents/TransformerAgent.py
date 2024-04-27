from agents import agent
import utils, click, torch

class TransformerAgent(torch.nn.Module):
    def __init__(
            self, 
            observation_size = 2,
            action_size      = 4,
            layers           = 3,
            embedding_size   = 64,
            feedforward_size = 256,
            heads            = 2,
            activation       = "gelu",
            device           = "cuda:0"
        ):
        super().__init__()
        self.device = device
        self.cov_var = torch.full(size=(observation_size,), fill_value=0.5, device=device)
        self.cov_mat = torch.diag(self.cov_var)

        self.agent_position = utils.layer_init(torch.nn.Linear(observation_size, embedding_size, device=device))
        #self.agent_type     = torch.nn.Embedding(observation_size, embedding_size, device=device)

        self.encoder = torch.nn.TransformerEncoder(
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

        self.critic = torch.nn.Sequential(
            self.agent_position,
            self.encoder,
            utils.Lambda(lambda x:x.sum(-2)),
            utils.layer_init(torch.nn.Linear(embedding_size, 1, device=device), std=1),
        )
        
        self.actor = torch.nn.Sequential(
            utils.Lambda(lambda x: self.agent_position(x[0])),
            self.encoder,
            utils.layer_init(torch.nn.Linear(embedding_size, action_size, device=device),std=.01),
            torch.nn.Tanh()
        )

    def get_value(self, x):
        return {"value" : self.critic(x).squeeze(-1).squeeze(-1)}

    def get_action(self, observation, agent_type=None, action=None):
        if agent_type is None: 
            agent_type = torch.zeros(observation.shape[:-1], device=self.device, dtype=torch.int)
            agent_type[:,0] = 1

        logits = self.actor((observation, agent_type))
        probs  = torch.distributions.MultivariateNormal(logits[:,:,:2], self.cov_mat)
        action = probs.sample() if action is None else action
        return {
            "action"   : action,
            "logprobs" : probs.log_prob(action).sum(-1),
            "entropy"  : probs.entropy()
        }

    def get_action_and_value(self, observation, agent_type=None, action=None):
        return self.get_action(observation, agent_type, action=action) | self.get_value(observation)

@agent.group(invoke_without_command=True)
@click.option("--observation_size" , "observation_size" , type=int , default=2)
@click.option("--action_size"      , "action_size"      , type=int , default=4)
@click.option("--layers"           , "layers"           , type=int , default=3)
@click.option("--embedding-size"   , "embedding_size"   , type=int , default=64)
@click.option("--feedforward_size" , "feedforward_size" , type=int , default=256)
@click.option("--heads"            , "heads"            , type=int , default=2)
@click.option("--activation"       , "activation"       , type=str , default="gelu")
@click.option("--device"           , "device"           , type=str , default="cuda:0")
@click.pass_obj
def transformer_agent(trainer, observation_size, action_size, layers, embedding_size, feedforward_size, heads, activation, device):
    trainer.set_agent(
        TransformerAgent(
            observation_size = observation_size,
            action_size      = action_size,
            layers           = layers,
            embedding_size   = embedding_size,
            feedforward_size = feedforward_size,
            heads            = heads,
            activation       = activation,
            device           = device
        )
    )
