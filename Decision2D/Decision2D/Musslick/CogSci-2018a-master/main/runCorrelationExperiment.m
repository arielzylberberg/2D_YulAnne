function [c_hat, offset_hat, estimatedCostFunction, estimatedControlSignals] = runCorrelationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, varargin)
% RUNCORRELATIONEXPERIMENT  Runs an experiment to estimate the control cost for a single subject
% function for two simulated agents
%
% Required arguments:
% 
%   agent                          ...structure that contains information about agent
%   experiment                 ...structure that contains information about experiment
%
% Outputs:
%
%  c_data                          ...structure that contains information about true and estimated c parameters for the subject
%  offset_data                   ...structure that contains information about true and estimated offset parameters for the subject
%  sample_data                 ...structure that contains information about estimated cost samples for the subject
%
% Author: Sebastian Musslick

    correctOffset = 0; % if set to 1, then non-zero integrated cost estimates are corrected under the assumption that offset is 0 (this is a reasonable assumption since offset does not have an effect on control signal allocation policy)
    runFullSweep = 0;
    
    if(~isempty(varargin))
        functionalForm = varargin{1};
        
        if(length(varargin) >= 2)
            correctOffset = varargin{2};
        end
        
        if(length(varargin) >= 3)
            runFullSweep = varargin{3};
        end
    else 
        functionalForm = 'exponential';
    end
    
        

    % define space of control signals
    controlSignalSpace = agent.controlSignalSpace;

    % define cost functions for subject
    costFnc = agent.costFnc;

    % define outcome probability function for subject
    outcomeProbabilityFnc = agent.outcomeProbabilityFnc;

    % define value function
    valueFnc = agent.valueFnc;

    %%% EXPERIMENT

    % set up reward manipulations
    rewards = experiment.rewards;

    % submit EVC agent to experiment (generate outcome probabilities from reward manipulations)
    [outcomeProbabilities] = runEVCAgent(controlSignalSpace, outcomeProbabilityFnc, valueFnc, costFnc, rewards);
    
    %%% FIT COST FUNCTION

    % set assumed control signal space
    assumedControlSignalSpace = experiment.assumedControlSignalSpace;

    % set assumed outcome probability function for the subject
    assumedOutcomeProbabilityFnc = experiment.assumedOutcomeProbabilityFnc;

    % set assumed value function for the subject
    assumedValueFnc = experiment.assumedValueFnc;

    % estimate cost function for EVC agents 1 & 2
    [estimatedCostFunction, estimatedControlSignals]  = estimateCostFnc(assumedControlSignalSpace, assumedOutcomeProbabilityFnc, assumedValueFnc, outcomeProbabilities, rewards);

    % correct offset
    if(correctOffset)
        estimatedCostFunction_full = estimatedCostFunction;
        estimatedControlSignals_full = estimatedControlSignals;
        nullSignals = estimatedCostFunction == 0;
        estimatedCostFunction(nullSignals) = [];
        estimatedControlSignals(nullSignals) = [];
    end
    
    % compute fit to cost function
    if(strcmp(functionalForm, 'linear'))
            [c_hat, offset_hat] = fitLinearCostFnc(estimatedControlSignals, estimatedCostFunction, c_searchSpace, offset_searchSpace);
    elseif(strcmp(functionalForm, 'exponential'))
            [c_hat, offset_hat] = fitExponentialCostFnc(estimatedControlSignals, estimatedCostFunction, c_searchSpace, offset_searchSpace);
    elseif((strcmp(functionalForm, 'quadratic')))
            [c_hat, offset_hat] = fitQuadraticCostFnc(estimatedControlSignals, estimatedCostFunction, c_searchSpace, offset_searchSpace);
    else
        % by default, fit exponential
        [c_hat, offset_hat] = fitExponentialCostFnc(estimatedControlSignals, estimatedCostFunction, c_searchSpace, offset_searchSpace, runFullSweep);
    end
    
    % correct offset
    if(correctOffset)
        estimatedCostFunction = estimatedCostFunction_full;
        estimatedControlSignals = estimatedControlSignals_full;
        if(~strcmp(functionalForm, 'exponential'))
            estimatedCostFunction(~nullSignals) = estimatedCostFunction(~nullSignals)  - offset_hat;
        end
    end

    
    end

    
    