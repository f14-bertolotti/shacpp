class Shacwm:
    def __init__(self):
        self.seed                   = 42
        self.episodes               = 10000

        self.train_envs             = 512
        self.train_steps            = 32

        self.eval_envs              = 512
        self.eval_steps             = 64

        self.policy_layers          = 1
        self.policy_hidden_size     = 2048
        self.policy_dropout         = 0.0
        self.policy_activation      = "Tanh"

        self.world_layers           = 1
        self.world_hidden_size      = 64
        self.world_feedforward_size = 256
        self.world_dropout          = 0.0
        self.world_activation       = "ReLU"

        self.world_learning_rate    = 0.001
        self.policy_learning_rate   = 0.001

        self.cache_size             = 10000
        self.world_batch_size       = 2000
        self.world_epochs           = 4

        self.gamma_factor           = 0.99
        self.lambda_factor          = 0.95
        
        self.etr                    = 5
        self.etv                    = 10

        self.compile                = True
        self.restore_path           = None
        self.device                 = "cuda:0"

shacwm = Shacwm()
