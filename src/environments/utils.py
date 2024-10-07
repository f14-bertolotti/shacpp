import environments
import vmas

def get_environment(name:str, envs:int, agents:int, device:str, grad_enabled:bool, seed:int)->vmas.simulator.environment.Environment:
    
    match name:
        case "dispersion" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Dispersion(
                    device = device ,
                    radius = .05    ,
                    agents = agents ,
                ),
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "transport" :
            return vmas.make_env(
                scenario="transport",
                n_agents=agents,
                num_envs=envs,
                device=device,
                continuous_actions=True,
                seed=seed,
                grad_enabled=grad_enabled,
            )
        case _:
            raise ValueError(f"Unknown environment {name}.")
