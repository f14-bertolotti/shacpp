class Shacrm:
    def __init__(self):
        self.seed                   = 42
        self.episodes               = 10000

        self.train_envs             = 512
        self.train_steps            = 32

        self.eval_envs              = 512
        self.eval_steps             = 64

        self.reward_batch_size      = 2000
        self.reward_epochs          = 10

        self.value_batch_size       = 2000
        self.value_epochs           = 10

        self.policy_layers          = 1
        self.policy_hidden_size     = 2048
        self.policy_dropout         = 0.3
        self.policy_activation      = "Tanh"

        self.value_layers           = 1
        self.value_hidden_size      = 2048
        self.value_dropout          = 0.0
        self.value_activation       = "Tanh"

        self.reward_layers          = 1
        self.reward_hidden_size     = 2048
        self.reward_dropout         = 0.0
        self.reward_activation      = "ReLU"

        self.value_learning_rate    = 0.001
        self.policy_learning_rate   = 0.001
        self.reward_learning_rate   = 0.001

        self.reward_cache_size      = 100000
        self.reward_bins            = 100
        self.value_cache_size       = 10000
        self.value_bins             = 10

        self.gamma_factor           = 0.99
        self.lambda_factor          = 0.95
        self.etr                    = 5
        self.etv                    = 10
        self.compile                = True
        self.restore_path           = None
        self.device                 = "cuda:0"

        self.max_reward             = float("+inf")

        self.value_clip_coefficient  = 1
        self.reward_clip_coefficient = 1
        self.policy_clip_coefficient = 1
        self.reward_ett              = 10
        self.value_ett               = 10
        self.policy_ett              = 10

        self.early_stopping = {
            "max_reward_fraction" : 0.9,
        }

shacrm = Shacrm()
