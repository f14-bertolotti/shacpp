class Ppo():
    def __init__(self):

        # environment configurations
        self.train_envs             = 512
        self.train_steps            = 32
        self.eval_envs              = 512
        self.eval_steps             = 64

        # policy model configuration
        self.policy_layers          = 1
        self.policy_hidden_size     = 64
        self.policy_feedforward     = 128
        self.policy_heads           = 1
        self.policy_dropout         = 0.1
        self.policy_activation      = "ReLU"
        self.policy_var             = 1

        # value model configuration
        self.value_layers           = 1
        self.value_hidden_size      = 64
        self.value_feedforward      = 128
        self.value_heads            = 1
        self.value_dropout          = 0.0
        self.value_activation       = "ReLU"

        # learning rate for policy/value/world models
        self.policy_learning_rate   = 0.001
        self.value_learning_rate    = 0.001

        # early stopping configurations
        self.early_stopping = {
            "max_reward_fraction" : 0.9,
            "max_envs_fraction"   : 0.9
        }

        self.batch_size             = 2000
        self.epochs                 = 4

        # training configurations
        self.is_deterministic       = True
        self.compile                = True
        self.restore_path           = None
        self.device                 = "cuda:0"
        self.seed                   = 42
        self.episodes               = 100000

        self.etr                    = 5   # epochs before envs. reset
        self.etv                    = 100 # epochs before evaluation

        # rl gamma and lambda coefficients
        self.gamma_factor           = 0.99
        self.lambda_factor          = 0.95

        # TBD
        self.observation_size = 0
        self.action_size      = 0
        self.action_space     = [0.0,0.0]
        self.agents           = 0
        self.dir              = ""
        self.environment      = ""



ppo = Ppo()
