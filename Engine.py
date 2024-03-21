import math, numpy, torch

def p(x): print(x); return x

class Policy(torch.nn.Module):

    def __init__(
            self, 
            agents           = 10,
            layers           = 3,
            input_size       = 2,
            embedding_size   = 128,
            feedforward_size = 256,
            heads            = 4,
            max_com_dist     = 3,
            activation       = "relu",
            device           = "cuda:0"
        ):
        super().__init__()
        self.activation   = activation
        self.agents       = agents
        self.heads        = heads
        self.max_com_dist = max_com_dist
        self.device       = device

        self.first = torch.nn.Linear(input_size , embedding_size, device=device)
        self.posis = torch.nn.Parameter(
            torch.normal(
                mean   = 0,
                std    = 1,
                size   = (agents, embedding_size),
                dtype  = torch.float,
                device = device
            )
        )

        #self.encoder = torch.nn.TransformerEncoder(
        #    torch.nn.TransformerEncoderLayer(
        #        d_model         = embedding_size,
        #        nhead           = heads,
        #        dim_feedforward = feedforward_size,
        #        activation      = activation,
        #        device          = device,
        #        batch_first     = True
        #    ), 
        #    num_layers=layers
        #)

        self.lins1 = torch.nn.ModuleList([torch.nn.Linear(embedding_size, feedforward_size, device=device) for _ in range(layers)])
        self.lins2 = torch.nn.ModuleList([torch.nn.Linear(feedforward_size, embedding_size, device=device) for _ in range(layers)])
        self.lns   = torch.nn.ModuleList([torch.nn.LayerNorm(embedding_size, device = device) for _ in range(layers)])

        self.output_a = torch.nn.Linear(embedding_size, 18, device=device)
        self.output_i = torch.nn.Linear(embedding_size, 5, device=device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def distance_mask(self, positions):
        return (torch.cdist(positions, positions) > self.max_com_dist).repeat(self.heads,1,1)

    def forward(self, positions):
        embeddings = self.first(positions) + self.posis
        #embeddings = self.encoder(embeddings)#, mask=self.distance_mask(positions))

        for lin0, lin1, ln in zip(self.lins1, self.lins2, self.lns):
            embeddings = embeddings + ln(lin1(torch.nn.functional.relu(lin0(embeddings))))

        prba = torch.nn.functional.softmax(self.output_a(embeddings), dim=-1)
        prbi = torch.nn.functional.softmax(self.output_i(embeddings), dim=-1)

        smpa = torch.distributions.Categorical(probs=prba).sample()
        smpi = torch.distributions.Categorical(probs=prbi).sample()

        lpa = torch.distributions.Categorical(probs=prba).log_prob(smpa)
        lpi = torch.distributions.Categorical(probs=prba).log_prob(smpi)

        a = (math.pi * 4 / 18) * smpa.unsqueeze(-1)
        i = (1 / 5) * smpi.unsqueeze(-1)
    
        actions = torch.cat([i * torch.cos(a), i * torch.sin(a)], -1)

        return actions, lpa, lpi
            
class Engine(torch.nn.Module):

    def __init__(self, batch_size=100, agents=10, trace=True, device="cuda:0"):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.agents = Policy(agents, device=device)
        self.trace = trace
        self.points = torch.tensor([[(i/5)*20-10,(j/5)*20-10] for i in range(5) for j in range(5)],dtype=torch.float, device=device)
        
        self.gamma=0.99

    def reward(self, positions):
        return (torch.cdist(self.points, positions).min(-1).values < 1).sum(-1)

    def forward(self, pos, iterations=20):
        if self.trace: trace = torch.empty((iterations+1, self.batch_size, self.agents.agents, 2), device=self.device, requires_grad=False, dtype=torch.float)
        if self.trace: trace[0] = pos
        
        trace_movements = []
        trace_log_probs = []
        rewards = []

        for i in range(iterations):
            action, lpa, lpi = self.agents(pos)
            pos = pos + action
            rewards.append(self.reward(pos))
            trace_movements.append(action)
            trace_log_probs.append((lpa, lpi))
            
            
            if self.trace: trace[i+1] = pos

        return {"pos" : pos, 
                "trace" : trace if self.trace else None, 
                "rewards" : torch.stack(rewards),
                "trace_log_probs" : trace_log_probs,
                "trace_movements" : trace_movements}

    def lossfn(self, result):

        loss = sum([(-r.unsqueeze(-1)*lpi.squeeze(-1)) + (-r.unsqueeze(-1)*lpa.squeeze(-1)) for r,(lpa,lpi) in zip(result["rewards"],result["trace_log_probs"])])

        return (loss.mean(), )


