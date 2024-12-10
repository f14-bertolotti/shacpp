import models
import torch

class AxisTransformerWorld(models.Model):
    """ 
        Transformer World Model with alternate attention patterns:
            1) autoregressive attention patterns on actions and agents cannot attend to each other.
            2) each action at step s can attend to all actions of all other agents at step s.
        A linear layer is used to map the encoded observations to the rewards and values.
    """
    def __init__(
        self, 
        observation_size : int,
        action_size      : int,
        agents           : int,
        steps            : int,
        heads            : int   = 1        ,
        layers           : int   = 3        ,
        dropout          : float = 0.0      ,
        hidden_size      : int   = 128      ,
        feedforward_size : int   = 512      ,
        activation       : str   = "ReLU"   ,
        device           : str   = "cuda:0" ,
        compute_reward   : bool  = True     ,
        compute_value    : bool  = True     ,
    ):
        super().__init__(observation_size, action_size, agents, steps)
        activation = {"ReLU":"relu", "GELU":"gelu"}[activation]
        self.compute_reward = compute_reward
        self.compute_value  = compute_value

        self.obs2hid = torch.nn.Linear(observation_size, hidden_size, device = device)
        self.act2hid = torch.nn.Linear(action_size     , hidden_size, device = device)
        self.posemb  = torch.nn.Parameter(torch.empty(1, steps+1, 1, hidden_size, device = device).normal_(0,0.02))

        self.ln = torch.nn.LayerNorm(hidden_size, device=device)

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                dim_feedforward = feedforward_size ,
                d_model         = hidden_size      ,
                activation      = activation       ,
                device          = device           ,
                nhead           = heads            ,
                dropout         = dropout          ,
                batch_first     = True
            ) for _ in range(layers)
        ])

        if self.compute_reward: self.hid2rew = torch.nn.Linear(hidden_size, 1, device = device)
        if self.compute_value : self.hid2val = torch.nn.Linear(hidden_size, 1, device = device)
        self.hid2obs = torch.nn.Linear(hidden_size, observation_size, device = device)

        self.steps_mask = AxisTransformerWorld.generate_steps_mask(agents, steps+1, device)
        self.agent_mask = AxisTransformerWorld.generate_agent_mask(agents, steps+1, device)
        self.merge_mask = (self.steps_mask == 0).logical_or(self.agent_mask == 0)
        self.merge_mask = torch.where(self.merge_mask, torch.tensor(0.0), torch.tensor(float('-inf')))

    @staticmethod
    def generate_agent_mask(agents, steps, device="cpu"):
        """ Generate autoregressive mask that prevents agents from attending to each other. """
        agent_mask = torch.full((agents*steps, agents*steps), float("-inf"), dtype=torch.float, device=device)
        for i in range(steps):
            for j in range(agents):
                for k in range(agents):
                    agent_mask[j+i*agents][k+i*agents] = 0
        return agent_mask

    @staticmethod
    def generate_steps_mask(agents, steps, device="cpu"):
        """ generate mask that prevents actions to attend to future and previous actions """
        steps_mask = torch.full((agents*steps, agents*steps), float("-inf"), dtype=torch.float, device=device)
        for i in range(steps):
            for j in range(agents):
                for k in range(steps):
                    if k <= k*agents <= i*agents: steps_mask[j + i*agents][k*agents + j] = 0
        return steps_mask

    def forward(self, obs, act):
        hidobs = self.obs2hid(obs)[:,[0]]
        hidact = self.act2hid(act)
        hidden = self.ln(torch.cat([hidobs, hidact], dim=1) + self.posemb).flatten(1,2)

        for i,layer in enumerate(self.layers):
            hidden = layer(hidden, src_mask=self.merge_mask)

        hidden = hidden.view(hidden.size(0), self.steps+1, self.agents, hidden.size(2))

        return {
            "observations" : self.hid2obs(hidden),
            "rewards"      : self.hid2rew(hidden)[:,1:].squeeze(-1) if self.compute_reward else None,
            "values"       : self.hid2val(hidden)[:,1:].squeeze(-1) if self.compute_value  else None
        } 

