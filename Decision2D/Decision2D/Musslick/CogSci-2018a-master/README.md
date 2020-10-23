# CogSci-2018a

Source code for analyses reported in

Musslick, S., Cohen, J. D., Shenhav, A. (submitted). Estimating the costs of cognitive control from task performance: theoretical validation and potential pitfalls. Cognitive Science Society Conference.

## CogSci_Sim1_Validation.m

This simulation estimates the cost of cognitive control under correct assumptions about agent’s probability outcome function and value function. The results of this simulation are shown in Figure 2. 

## CogSci_Sim2a_SensitivityAnalysis_TaskAutomaticity.m

This simulation estimates the difference in the cost of cognitive control for two agents. The estimation procedure is performed under the assumption that both agents equal in terms of their task automaticity. However, the true task automaticity was varied for both agents. The results indicate the ability to recover the correct relationship between the agents’ control cost parameters as a function of the true task automaticity. The results of this simulation are shown in Figure 3a. 

## CogSci_Sim2b_SensitivityAnalysis_RewardSensitivity.m

This simulation estimates the difference in the cost of cognitive control for two agents. The estimation procedure is performed under the assumption that both agents equal in terms of their reward sensitivity. However, the true reward sensitivity was varied for both agents. The results indicate the ability to recover the correct relationship between the agents’ control cost parameters as a function of the true reward sensitivity. The results of this simulation are shown in Figure 3b. 

## CogSci_Sim2c_SensitivityAnalysis_AccuracyBias.m

This simulation estimates the difference in the cost of cognitive control for two agents. The estimation procedure is performed under the assumption that both agents equal in terms of their accuracy bias. However, the true accuracy bias was varied for both agents. The results indicate the ability to recover the correct relationship between the agents’ control cost parameters as a function of the true accuracy bias. The results of this simulation are shown in Figure 3c.

##  CogSci_Sim3a_VariabilityAnalysis_TaskAutomaticity.m

This simulation determines the correlation between the true control cost parameter and the estimated control cost parameter across a pool of agents with different control costs. The estimation procedure is performed under the assumption that agents equal in terms of their task automaticity. However, the variability of task automaticity between simulated agents was varied across simulations. The results demonstrate the ability to recover true cost parameters as a function of between-agent variability in task automaticity. The results of this simulation are shown in Figure 4a.

##  CogSci_Sim3b_VariabilityAnalysis_RewardSensitivity.m

This simulation determines the correlation between the true control cost parameter and the estimated control cost parameter across a pool of agents with different control costs. The estimation procedure is performed under the assumption that agents equal in terms of their reward sensitivity. However, the variability of reward sensitivity between simulated agents was varied across simulations. The results demonstrate the ability to recover true cost parameters as a function of between-agent variability in reward sensitivity. The results of this simulation are shown in Figure 4b.

##  CogSci_Sim3c_VariabilityAnalysis_AccuracyBias.m

This simulation determines the correlation between the true control cost parameter and the estimated control cost parameter across a pool of agents with different control costs. The estimation procedure is performed under the assumption that agents equal in terms of their accuracy bias. However, the variability of accuracy bias between simulated agents was varied across simulations. The results demonstrate the ability to recover true cost parameters as a function of between-agent variability in accuracy bias. The results of this simulation are shown in Figure 4c.

## CogSci_Sim4a_acrossExperimentAnalysis_TaskAutomaticity.m

This simulation determines the correlation of agents’ estimated control costs parameters between two simulated experiments. Agents’ true control cost parameters were sampled such that there is no correlation between an agent’s control cost parameter in experiment 1 and its control cost parameter in experiment 2. The estimation procedure was performed under the assumption that all agents equal in terms of their task automaticity. However, the true correlation between an agent’s task automaticity in experiment 1 and its task automaticity in experiment 2 was systematically varied. The results demonstrate how spurious correlations of agents’ control cost parameters between experiments can arise as a function of true correlations of task automaticity between experiments. 

## CogSci_Sim4b_acrossExperimentAnalysis_RewardSensitivity.m

This simulation determines the correlation of agents’ estimated control costs parameters between two simulated experiments. Agents’ true control cost parameters were sampled such that there is no correlation between an agent’s control cost parameter in experiment 1 and its control cost parameter in experiment 2. The estimation procedure was performed under the assumption that all agents equal in terms of their reward sensitivity. However, the true correlation between an agent’s reward sensitivity in experiment 1 and its reward sensitivity in experiment 2 was systematically varied. The results demonstrate how spurious correlations of agents’ control cost parameters between experiments can arise as a function of true correlations of reward sensitivity between experiments. 

## CogSci_Sim4c_acrossExperimentAnalysis_AccuracyBias.m

This simulation determines the correlation of agents’ estimated control costs parameters between two simulated experiments. Agents’ true control cost parameters were sampled such that there is no correlation between an agent’s control cost parameter in experiment 1 and its control cost parameter in experiment 2. The estimation procedure was performed under the assumption that all agents equal in terms of their accuracy bias. However, the true correlation between an agent’s accuracy bias in experiment 1 and its accuracy bias in experiment 2 was systematically varied. The results demonstrate how spurious correlations of agents’ control cost parameters between experiments can arise as a function of true correlations of accuracy bias between experiments. 

