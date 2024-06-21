from environments.rewards.Reward import reward
import torch, click

class TransformerReward(torch.nn.Module):
    def __init__(self, layers = 3, embedding_size = 64, heads=2, feedforward_size=256, activation="gelu", device="cuda:0"):
        super().__init__()
        self.embedding = torch.nn.Linear(2, embedding_size, device=device)
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
        self.fin = torch.nn.Linear(embedding_size, 1, device=device)

    def forward(self, observation):
        embeddings = self.embedding(observation)
        encoded = self.encoder(embeddings).mean(1)
        reward = self.fin(encoded)
        return reward.squeeze(-1)
        
@reward.group(invoke_without_command=True)
@click.option("--layers"           , "layers"           , type=int , default=3        , help="number of layers")
@click.option("--embedding-size"   , "embedding_size"   , type=int , default=64       , help="embedding size")
@click.option("--heads"            , "heads"            , type=int , default=2        , help="number of attention heads")
@click.option("--feedforward-size" , "feedforward_size" , type=int , default=256      , help="feedforward size")
@click.option("--activation"       , "activation"       , type=str , default="gelu"   , help="activation function")
@click.option("--device"           , "device"           , type=str , default="cuda:0" , help="device")
@click.pass_obj
def transformer_reward(trainer, layers, embedding_size, heads, feedforward_size, activation, device):
    trainer.environment.set_reward_nn(
        TransformerReward(
            layers           = layers,
            embedding_size   = embedding_size,
            heads            = heads,
            feedforward_size = feedforward_size,
            activation       = activation,
            device           = device
        )
    )

