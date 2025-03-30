# Reviewer 1

> In some cases, SHAC++ shows slightly lower performance.

**skewed results** In VMAS all agents experience partial observability, 
some cases more severe than others. 
In SHAC++ reward and world model learn to predict their output from partial observation. 
Instead, SHAC world model and reward function have access to the full state. 
This can skew the results in favor of SHAC. Despite this, both algorithms perform similarly. 

**Early stopping** We also note that as soon as a run achieves the 90% of the maximum reward, 
we stop the training, this can lead to some variance in the final performance.  

> How well does it scale with larger numbers of agents? Specifically, have the authors conducted any experiments with agent populations larger than those presented?

We executed some experiment up to 7 agents, 
but discarded the runs as the performance was too heavily impacted to perform the 3 runs with different seeds. Scaling the number of agents has two main effects:
1) It becomes easier to discover action that lead to rewards, in this sense, the environment become less sparse.
2) It becomes more difficult to coordinate between the agents to achieve the goal. 
For the dispersion environment 2) it seems to be the main factor limiting the performance.
For the transport environment 1) it seems to be the main factor pushing the agents to perform better.

> The paper mentions challenges in scenarios with partial observability, such as the Discovery task. Are there any planned modifications or extensions to SHAC++ to better handle these scenarios?

We believe that partial observability is an intrinsic property of some environments that RL algorithm need to deal with. SHAC can access the global state through its gradient in order to inform the policy update, however, this feels like cheating. On the other hand, SHAC++ uses reward and world model that access only local observation. While this choice may skew the performance in favor of SHAC, we believe that this is a more realistic scenario. We can include this discussion in the paper, and we plan to add this as an extension to the SHAC++ framework in future work.

> Could the authors provide more insights into the factors contributing to the higher variance observed in some scenarios with SHAC++?

In SHAC++ also the reward and world model are initialized randomly. 
This can lead to this model to learn different solutions that impact the final reward differently, leading to higher variance. 
While a full analysis of the components that impact this variance would be definitely valuable, 
we believe that the number of potential factors (NN architecture, NN size, environment type, number of agents) would render this analysis too complex for the scope of this paper.

> What are the computational requirements and training times for SHAC++ compared to SHAC and PPO?

We can include the training times in the paper. However, it should be noted that the performance also depends heavily on some parameters such as training epochs for value, policy (and, reward and world model for SHAC++). Therefore, this would not be indicative on what method is the fastest to train. Nonetheless, SHAC++ is definitely more computationally heavy this is because it needs to train the reward and world model in addition to what SHAC does. 
However, for a comparison consider that for the sampling environment (which is the slowest in VMAS) SHAC++ takes 16 hours to complete the 20000 episodes similarly to PPO, while SHAC takes about 10 hours.

# Reviewer 2

> problematic claims that are misaligned with each other and with the supporting evidence. 

We apologize for the confusion. We'll revise the introduction to clarify our goals. SHAC demonstrated excellent performance in single-agent, differentiable settings. Our work aims to lift these constraints by developing SHAC++, which handles multi-agent and non-differentiable environments while maintaining comparable performance to SHAC.

> lower performance in some cases

**Goal**: Lift SHAC's requirements to broaden its applicability---see previous point.

**Skewed results**: In VMAS, agents only access local observations. SHAC++ respects this limitation while SHAC accesses global state through its gradients, giving SHAC an advantage. Despite this, both algorithms perform similarly.

**Early stopping** We also note that as soon as a run achieves the 90% of the maximum reward, we stop the training, this can lead to some variance in the final performance.  

> The proposed method is not evaluated in such complex environments; the test environments are very simple

While these environments present significant challenges for PPO/MAPPO (as shown by their poor performance), we acknowledge they could be more complex. We'll revise our paper to better reflect this and explain our environmental choices in the discussion section.

> The performance of SHAC++ is questionable in more complex MARL

We understand that more complex environments may offer more challenging scenarios. However, we want to stress that these environments are already incapacitating for PPO/MAPPO. 
We plan to evaluate SHAC++ in more complex environments with larger agent populations in the next revision. We would appreciate the reviewer's suggestions for suitable differentiable, scalable multi-agent environments that could provide more comprehensive benchmarks.

> Numerous practical algorithms already exist for such environments, for example MAPPO, HAPPO, and QMIX in multi-agent scenarios

**PPO & MAPPO confusion**: We always use MAPPO where PPO is evaluated for multi-agent setting (we just briefly mention in the paper). We will fix this in the paper.

**Why MAPPO and not others**: While HAPPO and QMIX are valid candidates, we limited baselines to maintain statistical significance. HAPPO's non-homogeneous design (non-shared parameters between agents) makes it computationally expensive and creates an unfair comparison. In our experiments, we consistently use parameter sharing for scalability. MAPPO supports both shared and non-shared parameters, making it appropriate for our evaluation.

**Why not SAC**: We appreciate the author suggestion to include SAC and we will consider adding it to the baselines for the next revision. However, the original SHAC paper already included a SAC baseline. Ultimately, we believe that the inclusion of SAC would not add much to the main discussion.

**Why not other algorithms**: We excluded PODS, CE-APG, and BPTT as they are precursors to SHAC. SAM-RL requires a differentiable renderer beyond the differentiable environment, making it even less practical (though our framework could learn this component too). DiffTORI's test-time optimization approach is difficult to extend to multi-agent settings. AHAC addresses stiff dynamics by truncating trajectories, which could improve SHAC's performance but doesn't address its fundamental limitations.

> The authors estimate the reward function and state transition using only local agent states and actions as input. This approach may limit the method's generality and correctness

Yes, this is intentional for realistic, partially observable environments. SHAC++ can use local information, unlike SHAC (needing global state). We prioritize this realism (details in paper).

> The parameters for the policy, value, reward, and state transition networks are all denoted by $\theta$

This is intentional, providing flexibility. Parameters can be fully separate ($\theta=[\theta_{\text{policy}}, \theta_{\text{reward}}, \theta_{\text{value}}, \theta_{\text{world}}]$) or have varying degrees of sharing.

> modeling the environment instead of directly utilizing its information might introduce biases and lead to performance degradation

We agree that learned models introduce potential for inaccuracy. Policy performance depends on model quality - accurate models provide informative gradients, while inaccurate ones introduce noise. However, we can mitigate this by bootstrapping the world model using existing trajectories before policy training, improving stability.

> To what SHAC+ stands for?

SHAC+ refers to a variant where only the reward function is learned while using the true environment dynamics.

> Visual analysis of the trained policies' behavior would be more convincing

We will include snapshot from trajectories from the learned policy.

> Confusion on $l$ and $r$

$l$ and $r$ are the lower and upper bounds of the action space (e.g., for actions in range (0,1), $l=0$ and $r=1$).

# Reviewer 3
We thank the reviewer for the positive feedback.

# Reviewer 4
> a strange way of phrasing the contribution

We understand the reviewer point of view. We will revise the introduction to better reflect the goal: to lift the requirements of the original SHAC framework to broaden the applicability of the algorithm.

> the paper conflates two contributions

We believe that SHAC has demonstrated excellent performance in single-agent setting. In this work, we aim to broaden its applicability by lifting some of its requirement without impacting the performance. Our framework extends SHAC to multi-agent, non-differentiable and partially observable environments. We believe this to be a fair contribution to field. However, we understand that the introduction may have phrased this in a confusing way. We will revise the introduction to better reflect the goal of the paper.

> Not convincing evaluation

We choose VMAS because it offers differentiable and single/multi-agent environments. Further, these environments are able to scale in complexity by increasing the number of agents. Ultimately, we believe VMAS to be a good testbed to assess the scaling performance of RL algorithm in general. We will add a discussion of the choice of the environments in the paper, comparing them to other environments that could be used for evaluation (e.g., PettingZoo, StarCraft II, etc.).

> results do not seem significant enough to make a deep statement about SHAC/SHAC++ in multi-agent settings.

We believe that the results are significant in showing that SHAC outperforms heavily PPO/MAPPO even in presence of stiff dynamics. Further, SHAC++ is able to match SHAC's performance despite being able to access only partial information. For example, consider that when SHAC achieves early stopping threshold, also SHAC++ is able to surpass the same threshold. On the other hand, when SHAC++ surpasses the threshold, SHAC is not always able to do so. 

> It could be interesting if there are ways to theoretically characterize the additional smoothness and desirable properties that come from the inductive bias of the network.

We agree. While, we do explore the theoretical desirable property, Sect D.2 provides a comparison between the gradients of SHAC++ with an mlp and a transformer. 

> why not using the full VMAS suite?

The VMAS suite is fairly extensive offering 21 different environments. A full evaluation of these environments even with a small degree of statistical significance would require thousands of runs. Therefore, we limited ourselves to the 4 mentioned environment which we believe to be representative of the whole suite, 
and capture the main challenges we would like to address, like cooperation, partial observability, and stiff dynamics. 
We will update Sect.A to offer more insight on this choice.

> The most convincing evaluation would be in comparison to exactly the settings from the original SHAC paper

We understand the author point of view. In this revision, we mainly focused on a multi-agent evaluation. We believe that an evaluation with respect to some original environment would be valuable. We will work towards this direction for the next revision. [Ci penso io a questo] 

> Omitted references

We will make sure to include TD-MPC2 and DreamerV3 in the related work section.

> I am also curious about the design space of improving bad simulator gradients. 

While we do not perform an extensive analysis on the design choices for improving the environment gradients. Section D.2 offers a preliminary results comparing the transformer gradients with the mlp gradient. Our intuition is that the neural network architecture has definitely an impact. However, a more in depth study comparing normalization layer, weigh decay, optimizers, etc. would be needed in order to draw meaningful conclusions.


