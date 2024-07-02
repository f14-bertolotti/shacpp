
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



