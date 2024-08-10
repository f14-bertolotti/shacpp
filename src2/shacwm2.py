from RewardModel import RewardModel2, RewardModel
from WorldModel import WorldModel
from ActorModel import ActorModel
from ValueModel import ValueModel
from Dispersion import Dispersion as DispersionScenario
import itertools, termcolor, numpy, torch, utils, vmas

def train_world_model(
        episode,
        world_model,
        world_model_optimizer,
        observation_cache,
        action_cache,
        reward_cache,
        world_model_msk_data,
        world_model_obs_data,
        world_model_act_data,
        world_model_rew_data,
        world_model_val_data,
        world_pert_low,
        world_pert_high,
        world_model_dataset_size,
        world_model_batch_size,
        gammas,
        gamma
    ):
 
    # train world model ###########################################
    dataloader = None
    with torch.no_grad():
        trajectory_values = ((reward_cache * gammas).sum(0)).mean(1).squeeze(1)

        indexes = utils.pert(
            low  = world_pert_low,
            high = world_pert_high,
            peak = (trajectory_values.flatten() - trajectory_values.min()) / (trajectory_values.max()-trajectory_values.min() + 1e-5) * (world_model_dataset_size-1),
        ).round().to(torch.long)


        world_model_msk_data[indexes] = True
        world_model_obs_data[indexes] = observation_cache.transpose(0,1)
        world_model_act_data[indexes] = action_cache     .transpose(0,1)
        world_model_rew_data[indexes] = (reward_cache * gammas).sum(0)
        world_model_val_data[indexes] = 0
        
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                world_model_obs_data[world_model_msk_data],
                world_model_act_data[world_model_msk_data],
                world_model_rew_data[world_model_msk_data],
                world_model_val_data[world_model_msk_data],
            ),
            collate_fn = torch.utils.data.default_collate,
            batch_size = world_model_batch_size,
            shuffle    = True,
        )
    

    for epoch in itertools.count(0):
        tpfn_obs,tot_obs = 0,0
        tpfn_rew,tot_rew = 0,0
        #tpfn_val,tot_val = 0,0
        for step, (obs, act, rew, val) in enumerate(dataloader,1):

            world_model_optimizer.zero_grad()
            prd_obs, prd_rew, prd_val = world_model(obs[:,[0]], act[:,:-1])

            rew_loss = ((prd_rew - rew)**2).mean()

            loss = rew_loss
            loss.backward()
            world_model_optimizer.step()

            tpfn_rew,tot_rew = tpfn_rew + torch.isclose(prd_rew, rew, atol=.1).float().sum(), tot_rew + numpy.prod(prd_rew.shape) 

            print(termcolor.colored(f"\r world model training episode:{episode:<5d}, epoch:{epoch:<3d}, step:{step:<3d}, mask:{world_model_msk_data.sum().item():<4d}, loss:{loss.item():5.4f}, accrew:{tpfn_rew/tot_rew:5.4f}", "light_red"), end="")
        if tpfn_rew/tot_rew > .9: break
        if epoch > 8: break
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
        peak = (reward_cache.flatten(0,1).sum() - reward_cache.flatten(0,1).sum().min()) / (reward_cache.flatten(0,1).sum().max() - reward_cache.flatten(0,1).sum().min() + 1e-5) * (reward_model_dataset_size-1),
    ).round().to(torch.long)

    reward_model_msk_data[indexes] = True
    reward_model_obs_data[indexes] = observation_cache.flatten(0,1).detach().clone()
    reward_model_act_data[indexes] = action_cache     .flatten(0,1).detach().clone()
    reward_model_rew_data[indexes] = reward_cache     .flatten(0,1).detach().clone()

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

            tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=.1).float().sum(), tot + numpy.prod(prd.shape)
            print(termcolor.colored(f"\rreward model training episode:{episode:<5d}, epoch:{epoch:<3d}, step:{step:<3d}, mask:{reward_model_msk_data.sum().item():<4d}, loss:{loss.item():5.4f}, acc:{tpfn/tot:5.4f}", "yellow"), end="")
        if tpfn/tot >= .9: break
        #if epoch >= 8: break
    print()



def unroll(actor_model, world_model, reward_model, world, observations = None, unroll_steps = 64):

    world.world.zero_grad()
    if observations is None: 
        observations = torch.stack(world.reset()).transpose(0,1)

    proxy_obs = observations.detach().clone()

    observation_cache = []
    action_cache      = []
    reward_cache      = []

    proxy_obs_cache   = []
    proxy_act_cache   = []
    proxy_rew_cache   = []

    for step in range(1, unroll_steps+1):
        observation_cache.append(observations)
        proxy_obs_cache  .append(proxy_obs)
        
        actions   = actor_model(observations)
        proxy_act = actor_model(proxy_obs)
        

        observations, rewards, _, _ = world.step(actions.transpose(0,1))
        proxy_obs = world_model(proxy_obs, actions)
        proxy_rew = reward_model(proxy_obs, actions)
    
        observations = torch.stack(observations).transpose(0,1)
        rewards      = torch.stack(rewards     ).transpose(0,1).unsqueeze(-1)

        print("\r", torch.isclose(observations, proxy_obs, atol=.1).float().mean().item(), torch.isclose(rewards, proxy_rew, atol=.1).float().mean().item(), end="\n")
        
        action_cache.append(actions)    
        reward_cache.append(rewards)     

        proxy_act_cache.append(proxy_act)
        proxy_rew_cache.append(proxy_rew)
    print()
    
    return torch.stack(observation_cache).detach().clone(), torch.stack(action_cache).detach().clone(), torch.stack(reward_cache).detach().clone(), torch.stack(proxy_obs_cache), torch.stack(proxy_rew_cache)


def train():

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
        grad_enabled       = False         ,
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
    world_model_dataset_size     = 10000
    world_model_batch_size       = 500
    reward_model_batch_size      = 500

    episode_to_reset            = 2
    
    gammas = torch.ones(train_steps, device=device, dtype=torch.float)
    gammas[1:] = gamma
    gammas = gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,train_envs,agents).unsqueeze(-1)
    

    world_pert_low   = torch.zeros(train_envs, dtype = torch.float32 , device=device, requires_grad=False)
    world_pert_high  = torch.zeros(train_envs, dtype = torch.float32 , device=device, requires_grad=False) * (world_model_dataset_size-1)
    reward_pert_low  = torch.ones (train_envs * train_steps, dtype = torch.float32 , device=device, requires_grad=False)
    reward_pert_high = torch.ones (train_envs * train_steps, dtype = torch.float32 , device=device, requires_grad=False) * (world_model_dataset_size-1)

    observation_size:int = numpy.prod(train_world.get_observation_space()[0].shape)
    action_size     :int = numpy.prod(train_world.get_action_space()[0].shape)
    
    actor_model  = ActorModel  (observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device)
    world_model  = WorldModel  (observation_size = observation_size, action_size = action_size, agents = agents, layers = 3, hidden_size = 512, dropout=0.1, activation="ReLU", device = device)
    reward_model = RewardModel2(observation_size = observation_size, action_size = action_size, agents = agents, layers = 3, hidden_size = 128, dropout=0.1, activation="ReLU", device = device)
    
    
    actor_model_optimizer  = torch.optim.Adam( actor_model.parameters(), lr=0.001)
    world_model_optimizer  = torch.optim.Adam( world_model.parameters(), lr=0.0001)
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.0001)

    world_model_obs_data = torch.zeros(world_model_dataset_size, train_steps, agents, observation_size, device=device)
    world_model_act_data = torch.zeros(world_model_dataset_size, train_steps, agents, action_size     , device=device)
    world_model_rew_data = torch.zeros(world_model_dataset_size, train_steps, agents, 1               , device=device)
    world_model_msk_data = torch.zeros(world_model_dataset_size, dtype=torch.bool, device=device)
 
    reward_model_obs_data = torch.zeros(reward_model_dataset_size, agents, observation_size, device=device)
    reward_model_act_data = torch.zeros(reward_model_dataset_size, agents, action_size     , device=device)
    reward_model_rew_data = torch.zeros(reward_model_dataset_size, agents, 1               , device=device)
    reward_model_msk_data = torch.zeros(reward_model_dataset_size, dtype=torch.bool, device=device)
    
    prev_obs = None
    
    for episode in range(1, episodes):
        
        observation_cache, action_cache, reward_cache, proxy_obs_cache, proxy_rew_cache = unroll(
            observations  = (None if episode == 1 or episode % episode_to_reset == 0 else prev_obs), 
            world         = train_world,
            unroll_steps  = train_steps,
            actor_model   = actor_model.sample,
            world_model   = world_model.step,
            reward_model  = reward_model,
        )

        # train actor model ###########################################
        actor_model_optimizer.zero_grad()
        loss = -((proxy_rew_cache * gammas).sum(0)).sum() / (train_steps * train_envs)
        loss.backward()
        actor_model_optimizer.step()



        with torch.no_grad():
            indexes = torch.randperm(world_model_dataset_size)[:train_envs]
            world_model_msk_data[indexes] = True
            world_model_obs_data[indexes] = observation_cache.transpose(0,1)
            world_model_act_data[indexes] = action_cache     .transpose(0,1)
            world_model_rew_data[indexes] = reward_cache     .transpose(0,1)
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
            tpfn_obs, tot_obs = 0, 0
            for step,(obs,act,rew) in enumerate(dataloader):
                world_model_optimizer.zero_grad()
                prd_obs = world_model.step(obs.flatten(0,1), act.flatten(0,1))
                prd_obs = prd_obs.view(obs.size(0), train_steps, agents, observation_size)[:,:-1]
                tgt_obs = obs[:,1:]
                tgt_rew = rew
                loss_obs = ((prd_obs - tgt_obs)**2).mean()
                loss = loss_obs
                loss.backward()
                world_model_optimizer.step()

                tpfn_obs,tot_obs = tpfn_obs + torch.isclose(prd_obs, tgt_obs, atol=.005).float().sum(), tot_obs + numpy.prod(prd_obs.shape) 

                print(f"\r{episode:<5d}, epoch:{epoch:<3d}, step:{step:<3d}, {loss_obs.item():6.5f}, {tpfn_obs/tot_obs.item():6.5f}, {world_model_msk_data.sum()}", end="")
                
            if tpfn_obs/tot_obs > .95: break
            if epoch > 100: break

        print()


    
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

 
        ## train actor model ###########################################
        #actor_model_optimizer.zero_grad()
        #_,rew,_ = world_model(observation_cache[[0]].transpose(0,1), action_cache.transpose(0,1))
        #loss = -(rew).mean()
        #loss.backward()
        #actor_model_optimizer.step()

        if episode % 10 == 0:
            torch.save({
                "actor_state_dict" : actor_model.state_dict(),   
            }, "actor.pkl")
    
    
        if episode % 10 == 0:
            
            # evaluate
            with torch.no_grad():
                actor_model.eval()

                _, _, eval_reward, _, _ = unroll(
                    observations = None, 
                    world        = eval_world,
                    unroll_steps = eval_steps,
                    actor_model  = actor_model,
                    world_model  = world_model.step,
                    reward_model = reward_model 
                )

                actor_model.train()

            print(termcolor.colored(f"eval reward {episode}, {eval_reward.sum() / eval_envs}", "green"))

        prev_obs = observation_cache[-1].detach().clone()
        del observation_cache
        del action_cache
        del reward_cache
        del proxy_rew_cache

if __name__ == "__main__":
    train()
