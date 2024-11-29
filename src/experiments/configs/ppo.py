class Ppo():
    def __init__(self):
        self.seed                   = 42
        self.episodes               = 10000

        self.train_envs             = 512
        self.train_steps            = 32

        self.eval_envs              = 512
        self.eval_steps             = 64

        self.batch_size             = 2000
        self.epochs                 = 4

        self.policy_layers          = 1
        self.policy_hidden_size     = 2048
        self.policy_dropout         = 0.0
        self.policy_activation      = "Tanh"

        self.value_layers           = 1
        self.value_hidden_size      = 2048
        self.value_dropout          = 0.0
        self.value_activation       = "Tanh"

        self.value_learning_rate    = 0.001
        self.policy_learning_rate   = 0.001

        self.gamma_factor           = 0.99
        self.lambda_factor          = 0.95
        self.etr                    = 5
        self.etv                    = 10
        self.compile                = True
        self.restore_path           = None
        self.device                 = "cuda:0"

        self.early_stopping = {
            "max_reward_fraction" : 0.95,
            "max_envs_fraction"   : 0.95
        }

ppo = Ppo()
