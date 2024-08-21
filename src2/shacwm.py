from RewardModel import *
from ActorModel import *
from ValueModel import *
from Dispersion import Dispersion as DispersionScenario
import numpy
import torch
import utils
import click
import utils
import vmas
import tqdm
import json
import os

def train_reward_model(
        episode, 
        model, 
        episode_data, 
        cached_data,
        batch_size, 
        dataset_size, 
        training_epochs,
        optimizer,
        logger,
    ):

    peak_cache = episode_data["rewards"].flatten(0,1).sum(1).sum(1)
    indexes = utils.pert(
        low  = cached_data["pert_low"],
        high = cached_data["pert_high"],
        peak = (peak_cache - peak_cache.min()) / (peak_cache.max() - peak_cache.min() + 1e-5) * (dataset_size-1),
    ).round().to(torch.long)

    cached_data["mask"        ][indexes] = True
    cached_data["observations"][indexes] = episode_data["observations"].flatten(0,1).detach().clone()
    cached_data["actions"     ][indexes] = episode_data["actions"]     .flatten(0,1).detach().clone()
    cached_data["rewards"     ][indexes] = episode_data["rewards"]     .flatten(0,1).detach().clone()

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            cached_data["observations"][cached_data["mask"]],
            cached_data["actions"     ][cached_data["mask"]],
            cached_data["rewards"     ][cached_data["mask"]],
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = batch_size,
        drop_last  = True,
        shuffle    = True
    )

    for epoch in range(training_epochs):
        tpfn,tot = 0,0
        tpfn_nonzero, tot_nonzero = 0,0
        for step, (obs, act, tgt) in enumerate(dataloader,1):
            optimizer.zero_grad()
            prd = model(obs,act)
            loss = ((prd - tgt)**2).mean()
            loss.backward()
            optimizer.step()

            tpfn,tot = tpfn + torch.isclose(prd, tgt, atol=.1).float().sum().item(), tot + numpy.prod(prd.shape).item() 
            tpfn_nonzero, tot_nonzero = tpfn_nonzero + torch.isclose(prd[tgt>0], tgt[tgt>0], atol=.1).float().sum().item(), tot_nonzero + numpy.prod(prd[tgt>0].shape).item()
            logger.info(json.dumps({
                "episode"         : episode,
                "epoch"           : epoch,
                "step"            : step,
                "loss"            : loss.item(),
                "accuracy"        : tpfn/(tot+1e-7),
                "accuracy_nz"     : tpfn_nonzero/(tot_nonzero+1e-7),
                "dataset_size"    : cached_data["mask"].sum().item(),
                "dataset_size_nz" : (cached_data["rewards"][cached_data["mask"]] > 0).sum().item(),
            }))

def train_value_model(
        episode,
        model, 
        optimizer, 
        episode_data,
        training_epochs, 
        batch_size, 
        slam, 
        gamma, 
        logger,
    ):
    
    target_values = utils.compute_values(
        values  = episode_data["values"] ,
        rewards = episode_data["rewards"],
        dones   = episode_data["dones"]  ,
        slam    = slam                   ,
        gamma   = gamma
    )

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            episode_data["observations"].flatten(0,1).detach(),
            target_values               .flatten(0,1).detach(),
        ),
        collate_fn = torch.utils.data.default_collate,
        batch_size = batch_size,
        drop_last  = False,
        shuffle    = True
    )
    
    for epoch in range(1, training_epochs+1):
        for step, (obs, tgt) in enumerate(dataloader, 1):
            optimizer.zero_grad()
            prd = model(obs)
            loss = ((prd - tgt)**2).mean()
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
    
            logger.info(json.dumps({
                "episode"         : episode,
                "epoch"           : epoch,
                "step"            : step,
                "loss"            : loss.item(),
            }))

def train_actor_model(
        episode,
        reward_model,
        actor_model,
        episode_data,
        optimizer,
        gammas,
        logger,
    ):
    steps, envs = episode_data["observations"].size(0), episode_data["observations"].size(1)

    # compute proxy rewards
    reward_model.eval()
    proxy_rewards = reward_model(episode_data["observations"].flatten(0,1), episode_data["actions"].flatten(0,1)).view(*episode_data["rewards"].shape)
    reward_model.train()
    
    # compute value cache mask
    dead_runs = episode_data["dones"][0,:,0]
    live_runs = dead_runs.logical_not()
    live_steps = episode_data["dones"][:,live_runs,0].logical_not().sum(0) - 1
    
    # compute loss
    optimizer.zero_grad()
    loss = -((proxy_rewards * gammas * episode_data["dones"].unsqueeze(-1).logical_not()).sum() + ((gammas * episode_data["values"])[live_steps,live_runs]).sum()) / (steps * envs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_model.parameters(), 1)
    optimizer.step()
    logger.info(json.dumps({
            "episode" : episode,
            "loss"    : loss.item(),
            "done"    : episode_data["dones"][-1,:,0].sum().int().item(),
            "reward"  : episode_data["rewards"].sum(0).mean(0).sum().item()
    }))

@torch.no_grad
def evaluate(
        episode,
        actor_model,
        reward_model, 
        value_model,
        world,
        steps,
        envs,
        logger
    ):
    actor_model .eval()
    reward_model.eval()
    value_model .eval()
    
    eval_episode = unroll(
        observations = None, 
        world        = world,
        unroll_steps = steps,
        actor_model  = actor_model,
        value_model  = value_model
    )
    proxy_rewards = reward_model(eval_episode["observations"].flatten(0,1), eval_episode["actions"].flatten(0,1)).view(*eval_episode["rewards"].shape)
    
    actor_model .train()
    reward_model.train()
    value_model .train()
   
    logger.info(json.dumps({
        "episode"            : episode,
        "done"               : eval_episode["dones"][-1,:,0].sum().int().item(),
        "reward"             : eval_episode["rewards"].sum().item() / envs,
        "reward_loss"        : ((eval_episode["rewards"] - proxy_rewards)**2).mean().item(),
        "reward_accuracy"    : torch.isclose(proxy_rewards, eval_episode["rewards"], atol=.1).float().mean().item(),
        "reward_accuracy_nz" : torch.isclose(proxy_rewards[eval_episode["rewards"]>0], eval_episode["rewards"][eval_episode["rewards"]>0], atol=.1).float().mean().item()
    }))

    return {
        "rewards" : eval_episode["rewards"].sum().item() / envs
    }


def unroll(
        actor_model,
        value_model,
        world      ,
        observations = None,
        actions      = None,
        unroll_steps = 64
    ):

    world.world.zero_grad()
    if observations is None: 
        observations = torch.stack(world.reset()).transpose(0,1)
    if actions is None:
        actions = torch.zeros(observations.size(0), observations.size(1), 2, device=observations.device)

    observation_cache = []
    action_cache      = []
    reward_cache      = []
    done_cache        = []

    for step in range(1, unroll_steps+1):
        observation_cache.append(observations)
        
        actions = actor_model(observations, actions)
        observations, rewards, dones, _ = world.step(actions.transpose(0,1))
        observations = torch.stack(observations).transpose(0,1)
        rewards      = torch.stack(rewards     ).transpose(0,1).unsqueeze(-1)

        action_cache.append(actions)    
        reward_cache.append(rewards)     
        done_cache  .append(dones)

    observation_cache = torch.stack(observation_cache)
    action_cache      = torch.stack(action_cache)
    reward_cache      = torch.stack(reward_cache)
    done_cache        = torch.stack(done_cache)
    value_cache       = value_model(observation_cache.flatten(0,1)).view(observation_cache.size(0), observation_cache.size(1), observation_cache.size(2), 1)

    return { 
            "observations" : observation_cache, 
            "actions"      : action_cache, 
            "rewards"      : reward_cache, 
            "values"       : value_cache,
            "dones"        : done_cache.unsqueeze(-1).repeat(1,1,observations.size(1)).float()
    }

@click.command
@click.option("--device"            , "device"            , type=str          , default="cuda:0" , help="random device")
@click.option("--seed"              , "seed"              , type=int          , default=42       , help="random seed")
@click.option("--episodes"          , "episodes"          , type=int          , default=500      , help="episodes before resetting the environement")
@click.option("--etr"               , "etr"               , type=int          , default=5        , help="training etr between evaluations")
@click.option("--value-batch-size"  , "value_batch_size"  , type=int          , default=512      , help="value model batch size")
@click.option("--reward-batch-size" , "reward_batch_size" , type=int          , default=512      , help="reward model batch size")
@click.option("--value-epochs"      , "value_epochs"      , type=int          , default=4        , help="value model epochs")
@click.option("--reward-epochs"     , "reward_epochs"     , type=int          , default=4        , help="reward model epochs")
@click.option("--etv"               , "etv"               , type=int          , default=10       , help="number of etv")
@click.option("--agents"            , "agents"            , type=int          , default=5        , help="number of agents")
@click.option("--train-envs"        , "train_envs"        , type=int          , default=512      , help="number of train environments")
@click.option("--eval-envs"         , "eval_envs"         , type=int          , default=512      , help="number of evaluation environments")
@click.option("--train-steps"       , "train_steps"       , type=int          , default=32       , help="number of steps for the training rollout")
@click.option("--eval-steps"        , "eval_steps"        , type=int          , default=64       , help="number of steps for the evaluation rollout")
@click.option("--gamma"             , "gamma_factor"      , type=float        , default=.99      , help="reward discount factor")
@click.option("--lambda"            , "lambda_factor"     , type=float        , default=.95      , help="td-lambda factor")
@click.option("--dir"               , "dir"               , type=click.Path() , default="./"     , help="directory in which store logs and checkpoints")
def run(
        dir,
        seed,
        episodes,
        agents,
        train_envs,
        train_steps,
        eval_envs,
        eval_steps,
        value_batch_size,
        value_epochs,
        reward_batch_size,
        reward_epochs,
        etr,
        gamma_factor,
        lambda_factor,
        etv,
        device
    ):

    utils.seed_everything(seed)

    eval_logger   = utils.get_file_logger(os.path.join(dir,  "eval.log"))
    reward_logger = utils.get_file_logger(os.path.join(dir,"reward.log"))
    value_logger  = utils.get_file_logger(os.path.join(dir, "value.log"))
    policy_logger = utils.get_file_logger(os.path.join(dir,"policy.log"))

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
    
    gammas = torch.ones(train_steps, device=device, dtype=torch.float)
    gammas[1:] = gamma_factor
    gammas = gammas.cumprod(0).unsqueeze(-1).unsqueeze(-1).repeat(1,train_envs,agents).unsqueeze(-1)
    
    observation_size:int = numpy.prod(train_world.get_observation_space()[0].shape)
    action_size     :int = numpy.prod(train_world.get_action_space()[0].shape)
    
    value_model  = ValueModel (observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device)
    actor_model  = ActorModel (observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 128, dropout=0.0, activation="Tanh", device = device)
    reward_model = RewardModel(observation_size = observation_size, action_size = action_size, agents = agents, layers = 1, hidden_size = 1024, dropout=0.0, activation="Tanh", device = device)
    
    reward_model_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.0001) 
    actor_model_optimizer  = torch.optim.Adam( actor_model.parameters(), lr=0.001)
    value_model_optimizer  = torch.optim.Adam( value_model.parameters(), lr=0.001)
    
    reward_model_cache = {
        "observations" : torch.zeros(reward_model_dataset_size, agents, observation_size, device=device),
        "actions"      : torch.zeros(reward_model_dataset_size, agents,      action_size, device=device),
        "rewards"      : torch.zeros(reward_model_dataset_size, agents,                1, device=device),
        "mask"         : torch.zeros(reward_model_dataset_size, dtype=torch.bool, device=device),
        "pert_low"     : torch.zeros(train_envs * train_steps, dtype = torch.float32 , device=device, requires_grad=False),
        "pert_high"    : torch.ones (train_envs * train_steps, dtype = torch.float32 , device=device, requires_grad=False) * (reward_model_dataset_size-1)
    }
   
    prev_obs, prev_act = None, None
    
    for episode in (bar:=tqdm.tqdm(range(1, episodes))):
        
        # unroll episode #############################################
        episode_data = unroll(
            observations  = (None if episode == 1 or episode % etr == 0 else prev_obs), 
            actions       = (None if episode == 1 or episode % etr == 0 else prev_act),
            world         = train_world,
            unroll_steps  = train_steps,
            actor_model   = actor_model.sample,
            value_model   = value_model
        )
    
        # train actor model ###########################################
        train_actor_model(
            episode      = episode               ,
            reward_model = reward_model          ,
            actor_model  = actor_model           ,
            episode_data = episode_data          ,
            optimizer    = actor_model_optimizer ,
            gammas       = gammas                ,
            logger       = policy_logger,
        )

         
        # train reward model ##########################################
        train_reward_model(
            episode         = episode                   ,
            model           = reward_model              ,
            optimizer       = reward_model_optimizer    ,
            episode_data    = episode_data              ,
            cached_data     = reward_model_cache        ,
            batch_size      = reward_batch_size         ,
            dataset_size    = reward_model_dataset_size ,
            training_epochs = reward_epochs             ,
            logger          = reward_logger
        )

        # train value model ###########################################
        train_value_model(
            episode         = episode               ,
            model           = value_model           ,
            optimizer       = value_model_optimizer ,
            episode_data    = episode_data          ,
            training_epochs = value_epochs          ,
            batch_size      = value_batch_size      ,
            slam            = lambda_factor         ,
            gamma           = gamma_factor          ,
            logger          = value_logger
        )
    
    
        if episode % etv == 0:
            torch.save({
                "actor_state_dict"  : actor_model.state_dict(),   
                "reward_state_dict" : reward_model.state_dict(),
                "value_state_dict"  : value_model.state_dict(),
            }, os.path.join(dir,"models.pkl"))

    
        if episode % etv == 0:
            eval_data = evaluate(
                episode      = episode      ,
                actor_model  = actor_model  ,
                reward_model = reward_model ,
                value_model  = value_model  ,
                world        = eval_world   ,
                steps        = eval_steps   ,
                envs         = eval_envs    ,
                logger       = eval_logger
            ) 
            eval_reward = eval_data["rewards"]
            bar.set_description(f"reward:{eval_reward:5.3f}")

        prev_obs = episode_data["observations"][-1].detach().clone()
        prev_act = episode_data["actions"     ][-1].detach().clone()
        del episode_data

if __name__ == "__main__":
    run()
