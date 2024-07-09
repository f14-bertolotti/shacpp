
== Very Large Batch Size
I tweaked PPO in several ways without success into solving some of these environments.
However, I noticed that the agents behave very similar although they should differentiate their behavior pretty soon in the environment.
The reason for this, I believe, stand in the training data.
When randomly moving, the agent rarely perform a "differentiated" behavior that is useful.
Most of the time, only one agent reach a reward while the other have no success at all. 
This pushes PPO to promote the successful behavior, as a consequence, all agents converge to the same behavior.
However, if we increase the batch size enough than each batch will contain some instances in which a differentiated behavior has been helpful.
By minimizing such batch PPO cannot push all agents together as it would hinder those instances in which the differentiated behavior has helped.

Dispersion , 3 agents , envs 256, bs 512  , achieve 1 out of 3 goals   , agents have the same behaviors.
Dispersion , 3 agents , envs 256, bs 4096 , achieve 1.4 out of 3 goals , agents shows some level of differentiation but they seem still have a behavior dependent between each other.
Dispersion , 3 agents , envs 512, bs 8192 , achieve 1.6 out of 3 goals , agents shows some level of differentiation but they seem still have a behavior dependent between each other.



== Shared Reward vs Per Agent Reward
A per agent rewards favors a competitive behavior
A shared reward favors a cooperative behavior

When using a agent based reward, in the dispersion environment, agents go all
towards the nearest reward rather than split to achieve all the rewards

When using a shared reward, in the dispersion environment, the behavior of agents that did not contributed to the achievement 
are also promoted. To end this results in only one agent learning a good behavior

== Dispersion environment
At the start of the dispersion environment all agents start at the same position
with the same observation.
However, the agents get different rewards. 
With agents that share the policy this causes problems
as for no reason one action is good or bad.

== Transformer
The transformer does not work when we use a single embedding per observation vector.
Instead, it works incredibly well, when we use an embedding per each observation in the observation vector.

Probably, this is caused by the fact that computation happens between attention heads
and having only an attention `agents x agents` (instead of `(agents · obs_size)`) does not 
allow for much computation.
