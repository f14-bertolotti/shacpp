# Reviewer 1

Better discussion on when SHAC++ matches SHAC's performance is needed.

Higher number of agents.

Discussion on partial observability. Not enough support at claim that SHAC++ can handle these scenarios

varying level of observability

Discussing high variance

add training times

# Reviewer 2

revise introduction to remove "methods cannot handle both single-agent and multi-agent scenarios, and also cannot be applied to non-differentiable environments"

Specify that the goal is not to outperform SHAC, but to provide a more general algorithm

MAPPO is already used in the multi-agent setting

Yes modeling the environment may introduce biases degrading performance. 
On the other hand, using the derivative of the environment may lead to complex dynamics and unstable gradients as shown in [TODO].
This can also lead in performance of degradation, while the work of [TODO] takes one road we explore the other.

Yes, we use the local observation to estimate the reward function and state transition. 
This is a deliberate choice to remain faithful to the partial observability of the environment.
While this is possible using PPO or SHAC++, this is not possible for SHAC as the backpropagation needs global information to compute the gradients.

We use a single parameter name, theta, to represent all possible parameter choices for the reward, state, and policy function.
For example, if we wanted separate network, we could set theta=[theta_1, theta_2, theta_3].
if we wanted shared network, we could set theta=[theta_1, theta_1, theta_1].
Ultimately, while this choice may seem confusing, it allows the highest degree of generality.

include observation and reward function accuracy

add experiment with one of shac's experiments

add SAC and explain MAPPO is already added.

Visual representation of cooperation

l,r are the upper and lower bound of the action space

# Reviewer 3

nothing

# Reviewer 4

We will rephrase that sentence.

theoretically characterize the additional smoothness od the network

add experiment with one of shac's experiments

add TD-MPC2 (https://arxiv.org/abs/2310.16828), DreamerV3 (https://arxiv.org/abs/2301.04104)

Yes modeling the environment may introduce biases degrading performance...

______________________

# Reviewer 1

> In some cases, SHAC++ shows slightly lower performance.

**Regarding lower performance** In VMAS all agents experience partial observability, some cases more severe than others. In SHAC++ reward and world model learn to predict the their output from partial observation. Instead SHAC world model and reward function have access to the full state. This can skew the results in favor of SHAC. Despite this, both algorithms perform similarly.

> How well does it scale with larger numbers of agents? Specifically, have the authors conducted any experiments with agent populations larger than those presented?

We executed some experiment up to 7 agents, but discarded the runs as the performance was too heavily impacted to perform the 3 runs with different seeds. Scaling the number of agents has two main effects:
1. It become easier to discover action that lead to rewards, in this sense the enviroment become less sparse.
2. It become more difficult to coordinate between the agents. 
For the dispersion environment 2. seems to be the main factor limiting the performance.
For the transport environement 1. seems to be the main factor pushing the agents to perform better.

> The paper mentions challenges in scenarios with partial observability, such as the Discovery task. Are there any planned modifications or extensions to SHAC++ to better handle these scenarios?

We believe that partial observability is an intrinsic property of some enviroments that RL algorithm need to deal with. SHAC can access the global state through its gradient in order to inform the policy update, however, this feels like cheating. On the other hand, SHAC++ uses reward and world model that access only local observation. While this choice may skew the performance in favor of SHAC, we believe that this is a more realistic scenario. We can include this discussion in the paper.

> Could the authors provide more insights into the factors contributing to the higher variance observed in some scenarios with SHAC++?

In SHAC++ also the reward and world model are intialized randomly. This can lead to this model to learn different solutions that impact the final reward differently, leading to higher variance. While a full analysis of the components that impact this variance would be definetely valuable, we believe that the number of potential factors (NN arch., NN size, environment type, no. agents) would render this analysis too complex for the scope of this paper.

> What are the computational requirements and training times for SHAC++ compared to SHAC and PPO?

We can include the training times in the paper. However, it should be noted that the performance also depends heavily on some parameters such as training epochs for value, policy (and, reward and world model for SHAC++). Therefore, this would not be indicative on what method is the fastest to train. Nonetheless, SHAC++ is definetely more computationally heavy this is because it needs to train the reward and world model in addition to what SHAC does. However, for a comparison consider that for the sampling enviroment (which is the slowest in VMAS) SHAC++ takes 16 hours to complete the 20000 episodes similarly to PPO, while SHAC takes about 10 hours.
