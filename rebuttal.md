# Reviewer 1

> In some cases, SHAC++ shows slightly lower performance.

**skewed results** In VMAS all agents experience partial observability, some cases more severe than others. In SHAC++ reward and world model learn to predict the their output from partial observation. Instead SHAC world model and reward function have access to the full state. This can skew the results in favor of SHAC. Despite this, both algorithms perform similarly. 

**Early stopping** We also note that as soon as a run achieves the 90% of the maximum reward, we stop the training, this can lead to some variance in the final performance.  

> How well does it scale with larger numbers of agents? Specifically, have the authors conducted any experiments with agent populations larger than those presented?

We executed some experiment up to 7 agents, but discarded the runs as the performance was too heavily impacted to perform the 3 runs with different seeds. Scaling the number of agents has two main effects:
1. It become easier to discover action that lead to rewards, in this sense, the enviroment become less sparse.
2. It become more difficult to coordinate between the agents. 
For the dispersion environment 2. seems to be the main factor limiting the performance.
For the transport environement 1. seems to be the main factor pushing the agents to perform better.

> The paper mentions challenges in scenarios with partial observability, such as the Discovery task. Are there any planned modifications or extensions to SHAC++ to better handle these scenarios?

We believe that partial observability is an intrinsic property of some enviroments that RL algorithm need to deal with. SHAC can access the global state through its gradient in order to inform the policy update, however, this feels like cheating. On the other hand, SHAC++ uses reward and world model that access only local observation. While this choice may skew the performance in favor of SHAC, we believe that this is a more realistic scenario. We can include this discussion in the paper.

> Could the authors provide more insights into the factors contributing to the higher variance observed in some scenarios with SHAC++?

In SHAC++ also the reward and world model are intialized randomly. This can lead to this model to learn different solutions that impact the final reward differently, leading to higher variance. While a full analysis of the components that impact this variance would be definetely valuable, we believe that the number of potential factors (NN arch., NN size, environment type, no. agents) would render this analysis too complex for the scope of this paper.

> What are the computational requirements and training times for SHAC++ compared to SHAC and PPO?

We can include the training times in the paper. However, it should be noted that the performance also depends heavily on some parameters such as training epochs for value, policy (and, reward and world model for SHAC++). Therefore, this would not be indicative on what method is the fastest to train. Nonetheless, SHAC++ is definetely more computationally heavy this is because it needs to train the reward and world model in addition to what SHAC does. However, for a comparison consider that for the sampling enviroment (which is the slowest in VMAS) SHAC++ takes 16 hours to complete the 20000 episodes similarly to PPO, while SHAC takes about 10 hours.

# Reviewer 2

> problematic claims that are misaligned with each other and with the supporting evidence. 

We are sorry for the confusion. We will revise the introduction to better reflect the goal of the paper. Briefly, we believe that SHAC demostrated excellent perfomance in single-agent and differentiable scenarios. In this work, we aim to lift these requirements (single-agent and differentiability) to broaden the applicability of SHAC. This led us to develop SHAC++ that can handle multi-agent and non-differentiable environments while mantaining performance comparable to SHAC.

> lower performance in some cases

**Goal**: We want to stress the fact that the goal is not to outperform SHAC, but to lift the requirements of the original framework (single-agent and differentiability) while maintaining comparable performance.

**Skewed results**: In VMAS as in many other enviroments, the agents are able to experience only local observetions and not the whole world state. While SHAC++ remains faithful to accessing only this partial information, SHAC access the whole global information through its gradients. This puts SHAC in a more favorable position. Despite this, SHAC++ and SHAC perform similarly.

**Early stopping** We also note that as soon as a run achieves the 90% of the maximum reward, we stop the training, this can lead to some variance in the final performance.  

> The proposed method is not evaluated in such complex environments; the test environments are very simple.

We believe that the chose enviroments offer a good degree of complexity especially when the number of agents is increased. This is also reflected by the poor performance achieved by PPO/MAPPO. Despite this, we recognize that the statement in the abstract may be misleading. We will revise the abstract and introduction to better reflect the complexity of the environments.

> The performance of SHAC++ is questionable in more complex MARL.

We understand that more complex enviroments with more agents may offer more challenging scenarios and more compelling results. However, we want to stress that these enviroments are already incapacitating for PPO. [RUN experiment with more agents]. Further, we will consider adding more complex enviroments for future revision, we kindly ask the reviewer to suggest a differentiable enviroment that can scale in the number of agents which he would believe to be adequate for the evaluation of SHAC++.

> Numerous practical algorithms already exist for such environments, for example MAPPO, HAPPO, and QMIX in multi-agent scenarios.

**PPO & MAPPO confusion**: While we do not evaluate against also HAPPO and QMIX, we always use MAPPO where PPO is evaluated for multi-agent setting. Since the MAPPO formulation is fairly natural extension to PPO, we always referred to PPO in the figures and tables, and only briefly mention that MAPPO is used in the case of multi-agent setting. We will revise the paper to better reflect this.

**Why MAPPO and not the others**: While HAPPO and QMIX are both valid candidate for baseline. In order to offers an ablation study and a degree of statistical significance we had to limit the number of baselines. We also note that HAPPO authors designed their algorithm to be non-homogeneous (parameters between agents are not shared). This choice renders its application problematic as istantiating and traiing several NN at the same time quickly becomes computationally infeasible. Meanwhile MAPPO can be applied with both shared and non-shared parameters. In our experiments, in order to be consisted while scaling the number of agents, we always apply parameter sharing. 

**Why not SAC**: We appreciate the author suggestion to include SAC and we will consider adding it to the baselines for the next revision. However, the original SHAC paper already included a SAC baseline. Ultimately, we believe that the inclusion of SAC would not add much to the main discussion.

**Why not the algorithm in related work section**: We did not include PODS, CE-APG, and BPTT as baselines as we see these a precursors of SHAC. The authors of SAM-RL introduce a differentiable renderer on top of the differentiable enviroment which is even more difficult to obtain in practice (Although also this component could be replaced with a learned model in our framework). Instead, DiffTORI applies a test-time optimization on the agent trajectory. This choice render the extension of DiffTORI to the multi-agent setting difficult. The authors of AHAC address the issue of stiff dynamic by simply truncating trajectories, while this choice can boost performance of SHAC, it mantains the fundamental limitation of the original framework.

We will provide a more detailed explanation in the paper.

> The authors estimate the reward function and state transition using only local agent states and actions as input. This approach may limit the method's generality and correctness.

Yes, this is a deliberate choice on our part. Since in some enviroment their state may be only partially observable then a broad algorithm need to be able to handle this. While with SHAC this is not possible, since for gradient computation the global state is needed, SHAC++ can handle this by unsing the local information available to the agents. Despite this choice may skew the performance in favor of SHAC, we believe that this is a more realistic scenario. We will include this discussion in the paper.

> The parameters for the policy, value, reward, and state transition networks are all denoted by $\theta$. 

This is a deliberate choice that allows us to denote a genera framework where the parameters can be arbitrarly shared. For example, all parameter may be separate such as in $\theta=[\theta_{\text{policy}}, \theta_{\text{reward}}, \theta_{\text{value}}, \theta_{\text{world}}]$. Or, $\theta$ could be designed with some degree of sharing.

> modeling the environment instead of directly utilizing its information might introduce biases and lead to performance degradation.

Modeling the enviroment means that the policy will be only as good as the enviroment model is. If the world model is very accurate, the gradient will be informative. If the world model is inaccurate or only partially accurate, the gradient will be noisy leading to degradation. However, while we train the world model in tandem with the policy, it is possible to bootstrap the world model with already available traces. This can lead to a more stable training of the policy. We will include this discussion in the paper.

> To what SHAC+ stands for?

The SHAC++ framework allows for replacing only certain components with learned models. For example, if a differentialble enviroment is available, but not a differentiable reward. Then we can only learn the latter. SHAC+ refers to the case where only the reward is learned. We will provide a more detailed explanation in the paper.

> Visual analysis of the trained policies' behavior would be more convincing.

We will include snapshot from trajectories from the learned policy.

> Confusion on $l$ and $r$.

$l$ and $r$ denote the lower and upper bound of the action space. For example, if the agent is supposed to output action in range of (0,1) then $l=0$ and $r=1$. We will provide a better explanation in the paper.


# Reviewer 3
We thank the reviewer for the positive feedback.

# Reviewer 4
> a strange way of phrasing the contribution

We understand the reviewer point of view. We will revise the introduction to better reflect the goal: to lift the requirements of the original SHAC framework to broaden the applicability of the algorithm.

> the paper conflates two contributions

We believe that SHAC has demonstrated excellent performance in single-agent setting. In this work, we aim to broaden its applicability by lifting some of its requirement without impacting the performance. Our framework extends SHAC to multi-agent, non-differentiable and partially observable environments. We believe this to be a fair contribution to field. However, we understand that the introduction may have phrased this in a confusing way. We will revise the introduction to better reflect the goal of the paper.

> Not convincing evaluation

We choose VMAS because it offers differentiable and single/multi-agent enviroments. Further, these enviroments are able to scale in complexity by increasing the number of agents. Ultimately, we believe VMAS to be a good testbed to assess the scaling performance of RL algorithm in general. 

> results do not seem significant enough to make a deep statement about SHAC/SHAC++ in multi-agent settings.

We believe that the results are significant in showing that SHAC outperforms heavily PPO/MAPPO even in presence of stiff dynamics. Further, SHAC++ is able to match SHAC's performance despite being able to access only partial information. For example, consider that when SHAC achieves early stopping threshold, also SHAC++ is able to surpass the same threshold. On the other hand, when SHAC++ surpasses the threshold, SHAC is not always able to do so. 

> It could be interesting if there are ways to theoretically characterize the additional smoothness and desirable properties that come from the inductive bias of the network.

We agree. While, we do explore the theoretical desirable property, Sect D.2 provides a comparison between the gradients of SHAC++ with an mlp and a transformer. 

> why not using the full VMAS suite?

The VMAS suite is fairly extensive offering 21 different enviroments. A full evualtion of this enviroments even with a small degree of statistical significance would require thousands of runs. Therefore, we limited ourself to the 4 mentioned enviromens which we believe to be representative of the whole suite. We will update Sect.A to offer more insight on this choice.

> The most convincing evaluation would be in comparison to exactly the settings from the original SHAC paper

We understand the author point of view. In this revision, we mainly focused on a multi-agent evaluation. We believe that an evaluation with respect to some original enviroment would be valuable. We will work towards this direction for the next revision. [Ci penso io a questo] 

> Omitted references

We will make sure to include TD-MPC2 and DreamerV3 in the related work section.

> I am also curious about the design space of improving bad simulator gradients. 

While we do not perform an extensive analysis on the design choices for improving the enviromen gradients. Section D.2 offers a preliminary results comparing the transformer gradients with the mlp gradient. Our intuition is that the neural network architecture has definetely an impact. However, a more in depth study comparing normalization layer, weigh decay, optimizers and etc. would be needed in order to draw meaningful conclusions.


