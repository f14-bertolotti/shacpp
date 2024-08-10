from RewardModel import RewardModel
from WorldModel import WorldModel
from ActorModel import ActorModel
from ValueModel import ValueModel
from Dispersion import Dispersion as DispersionScenario
import itertools, termcolor, numpy, torch, copy, utils, vmas


def train_reward_model(
        episode, 
        reward_model, 
        reward_cache, 
        observation_cache, 
        action_cache, 
        reward_model_msk_data, 
        reward_model_obs_data, 
        reward_model_act_data, 
        reward_model_rew_data, 
        reward_pert_low, 
        reward_pert_high, 
        reward_model_batch_size, 
        reward_model_dataset_size, 
        reward_model_optimizer
    ):

    indexes = utils.pert(
        low  = reward_pert_low,
        high = reward_pert_high,
        peak = (reward_cache.flatten() - reward_cache.min()) / (reward_cache.max() - reward_cache.min() + 1e-5) * (reward_model_dataset_size-1),
    ).round().to(torch.long)

    reward_model_msk_data[indexes] = True
    reward_model_obs_data[indexes] = observation_cache.flatten(0,2).detach().clone()
    reward_model_act_data[indexes] = action_cache     .flatten(0,2).detach().clone()
    reward_model_rew_data[indexes] = reward_cache     .flatten(0,2).detach().clone()

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            reward_model_obs_data[reward_model_msk_data],
            reward_model_act_data[reward_model_msk_data],
            reward_model_rew_data[reward_model_msk_data],
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = reward_model_batch_size,
        drop_last  = True,
        shuffle    = True
    )

    for epoch in itertools.count(0):
        tpfn,tot = 0,0
        for step, (obs, act, tgt) in enumerate(dataloader,1):
            reward_model_optimizer.zero_grad()
            prd = reward_model(obs,act)
            loss = ((prd - tgt)**2).mean()
            loss.backward()
            reward_model_optimizer.step()

            tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=.1).float().sum(), tot + prd.size(0) 
            print("\rreward model training", episode, epoch, step, epoch, f"{loss.item():5.4f}", f"{tpfn/tot:5.4f}", reward_model_msk_data.sum().item(), end="")
        if tpfn/tot >= .9: break
        if epoch > 100: break
    print()



def unroll(actor_model, world, observations = None, unroll_steps = 64):
    world.world.zero_grad()
    if observations is None: 
        observations = torch.stack(world.reset()).transpose(0,1)
    else: 
        observations = observations.detach().clone()

    observation_cache = []
    action_cache      = []
    reward_cache      = []

    for step in range(1, unroll_steps+1):
        observation_cache.append(observations)
        
        actions = actor_model(observations)
        observations, rewards, done, info = world.step(actions.transpose(0,1))
        observations = torch.stack(observations).transpose(0,1)
        rewards      = torch.stack(rewards     ).transpose(0,1).unsqueeze(-1)

        action_cache.append(actions)    
        reward_cache.append(rewards)     
    
    return torch.stack(observation_cache), torch.stack(action_cache), torch.stack(reward_cache)

def train():

    seed = 42
    utils.seed_everything(seed)

    agents = 3
    train_envs = 128
    eval_envs = 512
    eval_epochs = 1
    device = "cuda:0"
    episodes = 10000
    train_steps = 32
    eval_steps = 64
    gamma = .99
    
    reward_model_dataset_size    = 10000
    reward_model_batch_size      = 500
    episode_to_reset             = 4
    
    gammas = torch.ones(train_steps, device=device, dtype=torch.float)
    gammas[1:] = gamma
    gammas = gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,train_envs,agents).unsqueeze(-1)
    
    reward_pert_low  = torch.zeros(train_envs * agents * train_steps, dtype = torch.float32 , device=device, requires_grad=False)
    reward_pert_high = torch.ones (train_envs * agents * train_steps, dtype = torch.float32 , device=device, requires_grad=False) * (reward_model_dataset_size-1)
    
    train_world = vmas.simulator.environment.Environment(
        DispersionScenario(
            device = device,
            radius = .05,
            agents = agents,
        ),
        n_agents           = agents        ,
        num_envs           = train_envs    ,
        device             = device        ,
        shared_reward      = False         ,
        grad_enabled       = True          ,
        continuous_actions = True          ,
        dict_spaces        = False         ,
        seed               = seed          ,
    )
    
    eval_world = vmas.simulator.environment.Environment(
        DispersionScenario(
            device = device,
            radius = .05,
            agents = agents,
        ),
        n_agents           = agents        ,
        num_envs           = eval_envs     ,
        device             = device        ,
        shared_reward      = False         ,
        grad_enabled       = False         ,
        continuous_actions = True          ,
        dict_spaces        = False         ,
        seed               = seed          ,
    )
    
    observation_size:int = numpy.prod(train_world.get_observation_space()[0].shape)
    action_size     :int = numpy.prod(train_world.get_action_space()[0].shape)
    
    value_model  = ValueModel  (observation_size = observation_size, action_size = action_size, agents = agents, layers=1, hidden_size=128, dropout=0.0, activation="Tanh", device = device)
    actor_model  = ActorModel  (observation_size = observation_size, action_size = action_size, agents = agents, layers=1, hidden_size=128, dropout=0.0, activation="Tanh", device = device)
    reward_model = RewardModel (observation_size = observation_size, action_size = action_size, agents = agents, layers = 3, hidden_size = 256, dropout=0.1, activation="ReLU", device = device)
    
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.0001) 
    actor_model_optimizer  = torch.optim.Adam( actor_model.parameters(), lr=0.001)
    
    reward_model_obs_data = torch.zeros(reward_model_dataset_size, observation_size, device=device)
    reward_model_act_data = torch.zeros(reward_model_dataset_size,      action_size, device=device)
    reward_model_rew_data = torch.zeros(reward_model_dataset_size,                1, device=device)
    reward_model_msk_data = torch.zeros(reward_model_dataset_size, dtype=torch.bool, device=device)
    
    for episode in range(1, episodes):
        
        observation_cache, action_cache, reward_cache = unroll(
            observations = (None if episode == 1 or episode % episode_to_reset == 0 else observation_cache[-1]), 
            world        = train_world,
            unroll_steps = train_steps,
            actor_model  = actor_model.sample
        )
    
        # train actor model ###########################################
        actor_model_optimizer.zero_grad()
        proxy_rewards = reward_model(observation_cache, action_cache)
        loss = -((proxy_rewards * gammas).sum(0) ).sum() / (train_steps * train_envs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_model.parameters(), .5)
        actor_model_optimizer.step()
        print(reward_cache.sum(0).mean(0).sum().item())
    
        # train reward model ##########################################
        train_reward_model(
            episode                       = episode                       ,
            reward_model                  = reward_model                  ,
            reward_cache                  = reward_cache                  ,
            observation_cache             = observation_cache             ,
            action_cache                  = action_cache                  ,
            reward_model_msk_data         = reward_model_msk_data         ,
            reward_model_obs_data         = reward_model_obs_data         ,
            reward_model_act_data         = reward_model_act_data         ,
            reward_model_rew_data         = reward_model_rew_data         ,
            reward_pert_low               = reward_pert_low               ,
            reward_pert_high              = reward_pert_high              ,
            reward_model_batch_size       = reward_model_batch_size       ,
            reward_model_dataset_size     = reward_model_dataset_size     ,
            reward_model_optimizer        = reward_model_optimizer
        )


        if episode % 10 == 0:
            torch.save({
                "actor_state_dict" : actor_model.state_dict(),   
            }, "actor.pkl")
    
    
        if episode % 10 == 0:
            
            # evaluate
            with torch.no_grad():
                actor_model.eval()

                _, _, eval_reward = unroll(
                    observations = None, 
                    world        = eval_world,
                    unroll_steps = eval_steps,
                    actor_model  = actor_model,
                )

                actor_model.train()

            print(termcolor.colored(f"eval reward {episode}, {eval_reward.sum() / eval_envs / eval_epochs}", "green"))
        
train()
