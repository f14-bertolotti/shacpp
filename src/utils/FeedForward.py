import torch 

class FeedForward(torch.nn.Module):
    """ FeedForward Module with Skip Connection """
    def __init__(
        self, 
        input_size       :int            ,
        feedforward_size :int            ,
        dropout          :float = 0.0    ,
        activation       :str   = "ReLU"
    ):
        super().__init__()
        self.actv = getattr(torch.nn,activation)()
        self.lin1 = torch.nn.Linear(input_size, feedforward_size)
        self.lin2 = torch.nn.Linear(feedforward_size, input_size)
        self.lnrm = torch.nn.LayerNorm(input_size)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.lnrm(x + self.lin2(self.actv(self.lin1(self.drop(x)))))


