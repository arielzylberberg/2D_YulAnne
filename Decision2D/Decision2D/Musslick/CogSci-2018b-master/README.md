# CogSci-2018b

Source code for analyses reported in

Musslick, S., Jang, S. J., Shvartsman, M., Shenhav, A., Cohen, J. D. (submitted). Constraints associated with cognitive control and the stability-flexibility dilemma. Cognitive Science Society Conference.

## General

Both simulations described below investigate the performance of a recurrent neural network in a task switching experiment. Each of the two processing units in the control module receive an external input signal that indicates (cues) the currently relevant task. The dynamics of the network unfold over the course of trials and bias the processing of a corresponding stimulus dimension. On each trial, a decision module accumulates evidence for two stimulus dimensions towards one of two responses until a threshold is reached. The gain parameter of the network’s activation function modulates the distance between two control state attractors of the rule module.

## CogSci_Simulation1.m

This simulation investigates the network’s performance (overall performance, response incongruency effects and task switch costs) as a function of the gain parameter in a task switching sequence with 50% task switch probability. The results of this simulation are shown in Figure 2.

## CogSci_Simulation2.m

This simulation investigates the optimal gain that maximizes the network’s accuracy as a function of task switch probability in the experiment sequence. The optimal gain, as well as resulting performance metrics (overall performance, response incongruency effects and task switch costs) are determined. The results of this simulation are shown in Figure 3. 
