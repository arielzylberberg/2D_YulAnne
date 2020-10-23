function [c1_hat, c2_hat, offset1_hat, offset2_hat, estimatedCostFunction1, estimatedCostFunction2, estimatedControlSignals1, estimatedControlSignals2] = runEstimationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, varargin)
% RUNESTIMATIONEXPERIMENT  Runs an experiment to estimate the control cost
% function for two simulated agents
%
% Required arguments:
% 
%   agent                          ...structure that contains information about agent
%   experiment                 ...structure that contains information about experiment
%
% Outputs:
%
%  c_data                          ...structure that contains information about true and estimated c parameters for both subjects
%  offset_data                   ...structure that contains information about true and estimated offset parameters for both subjects
%  sample_data                 ...structure that contains information about estimated cost samples for both subjects
%
% Author: Sebastian Musslick

    runFullSweep = 0;
    if(~isempty(varargin))
        runFullSweep = varargin{1};
    end

    % define space of control signals
    controlSignalSpace = agent.controlSignalSpace;

    % define cost functions for both subjects
    costFnc1 = agent.costFnc1;
    costFnc2 = agent.costFnc2;

    % define outcome probability function for both subjects
    outcomeProbabilityFnc1 = agent.outcomeProbabilityFnc1;
    outcomeProbabilityFnc2 = agent.outcomeProbabilityFnc2;

    % define value function
    valueFnc1 = agent.valueFnc1;
    valueFnc2 = agent.valueFnc2;

    %%% EXPERIMENT

    % set up reward manipulations
    rewards = experiment.rewards;

    % submit EVC agents 1 & 2 to experiment (generate outcome probabilities from reward manipulations)
    
    [outcomeProbabilities1] = runEVCAgent(controlSignalSpace, outcomeProbabilityFnc1, valueFnc1, costFnc1, rewards);
    [outcomeProbabilities2] = runEVCAgent(controlSignalSpace, outcomeProbabilityFnc2, valueFnc2, costFnc2, rewards);
    
    %%% FIT COST FUNCTION

    % set assumed control signal space
    assumedControlSignalSpace1 = experiment.assumedControlSignalSpace1;
    assumedControlSignalSpace2 = experiment.assumedControlSignalSpace2;

    % set assumed outcome probability function for both subjects
    assumedOutcomeProbabilityFnc1 = experiment.assumedOutcomeProbabilityFnc1;
    assumedOutcomeProbabilityFnc2 = experiment.assumedOutcomeProbabilityFnc2;

    % set assumed value function for both subjects
    assumedValueFnc1 = experiment.assumedValueFnc1;
    assumedValueFnc2 = experiment.assumedValueFnc2;

    % estimate cost function for EVC agents 1 & 2
    [estimatedCostFunction1, estimatedControlSignals1]  = estimateCostFnc(assumedControlSignalSpace1, assumedOutcomeProbabilityFnc1, assumedValueFnc1, outcomeProbabilities1, rewards);
    [estimatedCostFunction2, estimatedControlSignals2]  = estimateCostFnc(assumedControlSignalSpace2, assumedOutcomeProbabilityFnc2, assumedValueFnc2, outcomeProbabilities2, rewards);

    % compute fit to cost function
    [c1_hat, offset1_hat] = fitExponentialCostFnc(estimatedControlSignals1, estimatedCostFunction1, c_searchSpace, offset_searchSpace, runFullSweep);
    [c2_hat, offset2_hat] = fitExponentialCostFnc(estimatedControlSignals2, estimatedCostFunction2, c_searchSpace, offset_searchSpace, runFullSweep);
    
    end

    
    