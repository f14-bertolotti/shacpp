class Shacwm:
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

        # world model configuration
        self.world_layers           = 1
        self.world_heads            = 1
        self.world_hidden_size      = 64
        self.world_feedforward      = 128
        self.world_dropout          = 0.0
        self.world_activation       = "ReLU"

        # learning rate for policy/world models
        self.world_learning_rate    = 0.001
        self.policy_learning_rate   = 0.001

        # clip coefficients for policy/world models
        self.world_clip_coefficient  = None
        self.policy_clip_coefficient = 1

        # cache sizes for world models
        self.world_cache_size       = 30000

        # batch sizes for world models
        self.world_batch_size       = 1000
        
        # number of epochs to train world models
        self.world_epochs           = 10

        # number of episode to wait before training world models
        self.world_ett              = 10

        # number of bins for value/reward/world caches
        self.world_bins             = 25

        # accuracy thresholds at which world training stops
        # for the current episode
        self.world_stop_threshold    = None

        # thresholds for computing accuracy of world models
        self.world_tolerance         = 0.01

        # stregth of the loss term for actions outside the action space
        self.out_coefficient = 0.1

        # early stopping configurations
        self.early_stopping = {
            "max_reward_fraction" : 0.9,
            "max_envs_fraction"   : 0.9
        }

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


shacum = Shacum()
