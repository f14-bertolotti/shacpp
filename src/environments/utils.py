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
            return vmas.simulator.environment.Environment(
                environments.scenarios.Transport(),
                package_mass       = 10.0         ,
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "sampling" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Sampling() ,
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "reverse_transport" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.ReverseTransport(),
                package_mass       = 10.0         ,
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "flocking" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Flocking(),
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )

        case "discovery" :
            return vmas.simulator.environment.Environment(
                environments.scenarios.Discovery(
                    agents_per_target = min(agents, 2),
                ),
                n_agents           = agents       ,
                num_envs           = envs         ,
                device             = device       ,
                grad_enabled       = grad_enabled ,
                continuous_actions = True         ,
                dict_spaces        = False        ,
                seed               = seed         ,
            )
        case "ant" :
            return environments.scenarios.Ant(
                device       = device       ,
                num_envs     = envs         ,
                seed         = seed         ,
                grad_enabled = grad_enabled ,
            )
        case _:
            raise ValueError(f"Unknown environment {name}.")
