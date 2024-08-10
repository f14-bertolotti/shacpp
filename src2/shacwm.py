from RewardModel import RewardModel
from WorldModel import WorldModel
from ActorModel import ActorModelOFA, ActorModel
from ValueModel import ValueModelOFA, ValueModel
from Dispersion import Dispersion as DispersionScenario
import itertools, termcolor, numpy, torch, copy, utils, vmas


def train_world_model(
        episode,
        world_model,
        world_model_optimizer,
        observation_cache,
        action_cache,
        value_cache,
        reward_cache,
        world_model_msk_data,
        world_model_obs_data,
        world_model_act_data,
        world_model_rew_data,
        world_pert_low,
        world_pert_high,
        world_model_dataset_size,
        world_model_batch_size,
        gammas,
        gamma
    ):
 
    # train world model ###########################################
    trajectory_values = ((reward_cache * gammas).sum(0) + value_cache[-1] * (gamma ** reward_cache.size(0))).detach().mean(1).squeeze(1)

    indexes = utils.pert(
        low  = world_pert_low,
        high = world_pert_high,
        peak = (trajectory_values.flatten() - trajectory_values.min()) / (trajectory_values.max()-trajectory_values.min() + 1e-5) * (world_model_dataset_size-1),
    ).round().to(torch.long)


    world_model_msk_data[indexes] = True
    world_model_obs_data[indexes] = observation_cache.transpose(0,1).detach()
    world_model_act_data[indexes] = action_cache     .transpose(0,1).detach()
    world_model_act_data[indexes] = reward_cache     .transpose(0,1).detach()
    
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            world_model_obs_data[world_model_msk_data],
            world_model_act_data[world_model_msk_data],
            world_model_rew_data[world_model_msk_data],
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = world_model_batch_size,
        shuffle    = True,
    )

    for epoch in itertools.count(0):
        tpfn,tot = 0,0
        for step, (obs, act, rew) in enumerate(dataloader,1):
            src0, src1, tgt = obs[:,[0]], act[:,:-1], obs

            world_model_optimizer.zero_grad()
            prd = world_model(src0, src1)
            loss = ((prd - tgt)**2).mean()
            loss.backward()
            world_model_optimizer.step()

            tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=.1).float().sum(), tot + numpy.prod(prd.shape) 

            print(termcolor.colored(f"\r world model training episode:{episode:<5d}, epoch:{epoch:<3d}, step:{step:<3d}, mask:{world_model_msk_data.sum().item():<4d}, loss:{loss.item():5.4f}, acc:{tpfn/tot:5.4f}", "light_red"), end="")
        if tpfn/tot >= .8: break
        #if epoch >= 8:  break
    print()
    

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
            print(termcolor.colored(f"\rreward model training episode:{episode:<5d}, epoch:{epoch:<3d}, step:{step:<3d}, mask:{reward_model_msk_data.sum().item():<4d}, loss:{loss.item():5.4f}, acc:{tpfn/tot:5.4f}", "yellow"), end="")
        if tpfn/tot >= .9: break
        #if epoch >= 8: break
    print()

def train_value_model(
        episode,
        value_model, 
        value_model_optimizer, 
        agents, 
        train_steps, 
        train_envs, 
        observation_cache, 
        value_cache, 
        reward_cache, 
        value_model_training_epochs, 
        value_model_batch_size, 
        slam, 
        gamma, 
        device
    ):
    target_values = utils.compute_values(train_steps, train_envs, value_cache, agents, reward_cache, slam=slam, gamma=gamma, device=device)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            observation_cache.flatten(0,1).detach(),
            target_values    .flatten(0,1).detach(),
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = value_model_batch_size,
        drop_last  = True,
        shuffle    = True
    )
    
    for epoch in range(1, value_model_training_epochs+1):
        for step, (obs, tgt) in enumerate(dataloader, 1):
            value_model_optimizer.zero_grad()
            prd = value_model(obs)
            loss = ((prd - tgt)**2).mean()
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), .5)
            value_model_optimizer.step()
    
            print(termcolor.colored(f"\r value model training episode:{episode:<5d}, epoch:{epoch:<3d}, step:{step:<3d}, loss:{loss.item():5.4f}", "magenta"), end="")
    print()


def unroll(actor_model, world, observations = None, unroll_steps = 64, stop_gradient=False):

    world.world.zero_grad()
    if observations is None: 
        observations = torch.stack(world.reset()).transpose(0,1)

    observation_cache = []
    action_cache      = []
    reward_cache      = []

    for step in range(1, unroll_steps+1):
        observation_cache.append(observations.detach() if stop_gradient else observations)
        
        actions = actor_model(observations.detach() if stop_gradient else observations)
        observations, rewards, _, _ = world.step(actions.transpose(0,1))
        observations = torch.stack(observations).transpose(0,1)
        rewards      = torch.stack(rewards     ).transpose(0,1).unsqueeze(-1)

        action_cache.append(actions)    
        reward_cache.append(rewards)     
    
    return torch.stack(observation_cache), torch.stack(action_cache), torch.stack(reward_cache)


def train():

    #torch.autograd.set_detect_anomaly(True)

    seed = 42
    utils.seed_everything(seed)

    agents       = 3
    train_envs   = 128
    eval_envs    = 512
    device       = "cuda:0"
    episodes     = 10000
    train_steps  = 32
    eval_steps   = 64
    gamma        = .99
    slam         = .95
    target_alpha = .4

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
    
    reward_model_dataset_size    = 10000
    reward_model_batch_size      = 100
    world_model_dataset_size     = 10000
    world_model_batch_size       = 100
    
    value_model_training_epochs  = 8
    value_model_batch_size       = 100
    episode_to_reset             = 4
    
    gammas = torch.ones(train_steps, device=device, dtype=torch.float)
    gammas[1:] = gamma
    gammas = gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,train_envs,agents).unsqueeze(-1)
    
    reward_pert_low  = torch.zeros(train_envs * agents * train_steps, dtype = torch.float32 , device=device, requires_grad=False)
    reward_pert_high = torch.ones (train_envs * agents * train_steps, dtype = torch.float32 , device=device, requires_grad=False) * (reward_model_dataset_size-1)
    world_pert_low   = torch.zeros(train_envs, dtype = torch.float32 , device=device, requires_grad=False)
    world_pert_high  = torch.ones (train_envs, dtype = torch.float32 , device=device, requires_grad=False) * (world_model_dataset_size-1)

    observation_size:int = numpy.prod(train_world.get_observation_space()[0].shape)
    action_size     :int = numpy.prod(train_world.get_action_space()[0].shape)
    
    value_model  = ValueModel(observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device)
    actor_model  = ActorModel(observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device)
    world_model  = WorldModel   (observation_size = observation_size, action_size = action_size, agents = agents, layers = 3, hidden_size = 512, feedforward_size = 1024, heads=4, dropout=0.0, activation="gelu", device = device)
    reward_model = RewardModel  (observation_size = observation_size, action_size = action_size, agents = agents, layers = 3, hidden_size = 128, dropout=0.1, activation="ReLU", device = device)
    
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.0001) 
    world_model_optimizer  = torch.optim.Adam( world_model.parameters(), lr=0.001)
    actor_model_optimizer  = torch.optim.Adam( actor_model.parameters(), lr=0.001)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=0.001)
    
    reward_model_obs_data = torch.zeros(reward_model_dataset_size, observation_size, device=device)
    reward_model_act_data = torch.zeros(reward_model_dataset_size,      action_size, device=device)
    reward_model_rew_data = torch.zeros(reward_model_dataset_size,                1, device=device)
    reward_model_msk_data = torch.zeros(reward_model_dataset_size, dtype=torch.bool, device=device)
    
    world_model_obs_data = torch.zeros(world_model_dataset_size, train_steps, agents, observation_size, device=device)
    world_model_act_data = torch.zeros(world_model_dataset_size, train_steps, agents,      action_size, device=device)
    world_model_rew_data = torch.zeros(world_model_dataset_size, train_steps, agents,                1, device=device)
    world_model_msk_data = torch.zeros(world_model_dataset_size, dtype=torch.bool, device=device)

    prev_obs = None
    
    for episode in range(1, episodes):
        
        observation_cache, action_cache, reward_cache = unroll(
            observations  = (None if episode == 1 or episode % episode_to_reset == 0 else prev_obs), 
            world         = train_world,
            unroll_steps  = train_steps,
            actor_model   = actor_model.sample,
            stop_gradient = False,
        )
        value_cache = value_model(observation_cache.flatten(0,1)).view(train_steps, train_envs, agents, 1)
    
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

        # train world model ##########################################
        #train_world_model(
        #    episode                  = episode                  ,
        #    world_model              = world_model              ,
        #    world_model_optimizer    = world_model_optimizer    ,
        #    observation_cache        = observation_cache        ,
        #    action_cache             = action_cache             ,
        #    value_cache              = value_cache              ,
        #    reward_cache             = reward_cache             ,
        #    world_model_msk_data     = world_model_msk_data     ,
        #    world_model_obs_data     = world_model_obs_data     ,
        #    world_model_act_data     = world_model_act_data     ,
        #    world_model_rew_data     = world_model_rew_data     ,
        #    world_pert_low           = world_pert_low           ,
        #    world_pert_high          = world_pert_high          ,
        #    world_model_dataset_size = world_model_dataset_size ,
        #    world_model_batch_size   = world_model_batch_size   ,
        #    gammas                   = gammas                   ,
        #    gamma                    = gamma                    
        #)
 
        # train actor model ###########################################
        actor_model_optimizer.zero_grad()
        #proxy_obs = world_model(observation_cache[[0]].transpose(0,1).detach(), action_cache[:-1].transpose(0,1))
        #proxy_obs = proxy_obs.view(train_envs, agents, train_steps, observation_size).permute(2,0,1,3)
        
        proxy_rewards = reward_model(observation_cache, action_cache)
        loss = -((proxy_rewards * gammas).sum(0)  + (gamma ** train_steps) * value_cache[-1]).sum() / (train_steps * train_envs)
        loss.backward()
        actor_model_optimizer.step()
        #print("actor model training", loss.item())
    
        print(termcolor.colored(f" actor model training episode:{episode:<5d}, loss:{loss.item():5.4f}, rew:{reward_cache.sum(0).mean(0).sum().item():5.4f}", "cyan"))
 

        # train value model ###########################################
        train_value_model(
            episode                     = episode                     ,
            value_model                 = value_model                 ,
            value_model_optimizer       = value_model_optimizer       ,
            agents                      = agents                      ,
            train_steps                 = train_steps                 ,
            train_envs                  = train_envs                  ,
            observation_cache           = observation_cache           ,
            value_cache                 = value_cache                 ,
            reward_cache                = reward_cache                ,
            value_model_training_epochs = value_model_training_epochs ,
            value_model_batch_size      = value_model_batch_size      ,
            slam                        = slam                        ,
            gamma                       = gamma                       ,
            device                      = device
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

            print(termcolor.colored(f"eval reward {episode}, {eval_reward.sum() / eval_envs}", "green"))

        prev_obs = observation_cache[-1].detach().clone()
        del observation_cache
        del action_cache
        del reward_cache
        del value_cache 
        
train()
