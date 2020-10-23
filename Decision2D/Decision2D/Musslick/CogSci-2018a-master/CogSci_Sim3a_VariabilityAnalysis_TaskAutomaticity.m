% Correlation Analysis
%
%
%

clear all;
clc;

addpath('main');

%%% META PARAMETERS
logFolderName = 'logfiles';

similarityMetric = 'corr'; % options: 'corr' - correlation; 'rankord' - rank order coefficient, 'jaccard' - jaccard index

numSubj = 100;                      % number of subjects
numStdConditions = 10;       % number of standard deviation conditions
numReps = 10;                     % number of simulation repetitions per standard deviation conditions

std_range = [0 10];                  % range of tested standard deviations
control_cost_range = [1 4];    % range of tested control cost values

std_values = linspace(std_range(1), std_range(2), numStdConditions);
control_cost_values = linspace(control_cost_range(1), control_cost_range(2), numSubj);

%%% DEFINE DEFAULT AGENT

% parameters
default_controlCost = 1;
default_taskAutomaticity = -7.5;
default_controlEfficacy = 15;
default_accuracyBias = 0;
default_rewardSensitivty = 1;

% define space of control signals
agent.controlSignalSpace = 0:0.01:1;

% define cost functions for both subjects
agent.costFnc = @(u) exp(default_controlCost * u) - 1;

% define outcome probability function for both subjects
agent.outcomeProbabilityFnc = @(u) 1./(1+exp(-default_controlEfficacy*u - default_taskAutomaticity));

% define value function
agent.valueFnc = @(u) default_rewardSensitivty * u + default_accuracyBias;

%%% EXPERIMENT & FIT PROCEDURE

% set up reward manipulations
experiment.rewards = 0:1:100;

% set assumed control signal space
experiment.assumedControlSignalSpace = agent.controlSignalSpace;

% set assumed outcome probability function for both subjects
experiment.assumedOutcomeProbabilityFnc = agent.outcomeProbabilityFnc;

% set assumed value function for both subjects
experiment.assumedValueFnc = agent.valueFnc;

% search space for parameter fitting
c_searchSpace = -2:0.1:10;
offset_searchSpace = -8:0.1:15;

%% Simulation Loop

overwrite = 1;
runFullSweep = 1;

% generate log file name
logfileName = ['taskAutomaticity_corrAnalysis_' ...
                        num2str(std_range(1)*100) '_' num2str(std_range(end)*100) '_' ...
                        num2str(control_cost_range(1)*100) '_' num2str(control_cost_range(end)*100) '_'...
                        num2str(numReps) '_' ...
                        num2str(numSubj)];
                    
filePath = [logFolderName '/' logfileName '.mat'];

if exist(filePath, 'file') == 2 && ~overwrite     % if log file exists load it
    load(filePath);
else                                         % if log file doesn't exist, generate it 

    sim_log = nan(numStdConditions, numReps);

    % for each standard_deviationcondition
    for std_cond_idx = 1:length(std_values) 

        standard_deviation= std_values(std_cond_idx);

        % for each repetition
        for rep = 1:numReps

            c_hat_log = nan(1, numSubj);     % estimated control cost parameters

            % for each subject
            for subj = 1:numSubj

                % pick control cost value
                c = control_cost_values(subj);
                agent.costFnc = @(u) exp(c * u) - 1;

                % vary motivational parameter
                taskAutomaticity = default_taskAutomaticity + standard_deviation* randn;
                agent.outcomeProbabilityFnc = @(u) 1./(1+exp(-default_controlEfficacy*u - taskAutomaticity));

                [c_hat, offset_hat, estimatedCostFunction, estimatedControlSignals] = runCorrelationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, [], 0, runFullSweep);
                c_hat_log(subj) = c_hat;

            end
            
            % remove nan conditions
            control_cost_values_used = control_cost_values;
            removeIdx = find(isnan(c_hat_log));
            c_hat_log(removeIdx) = [];
            control_cost_values_used(removeIdx) = [];
            
            % perform correlation analysis
            switch similarityMetric
                case 'corr'
                    R = corr([control_cost_values_used', c_hat_log']);
                    similarity = R(2);
            end

            sim_log(std_cond_idx, rep) = similarity;

        end

        disp(['progress: ' num2str(std_cond_idx) '/' num2str(length(std_values) )]);

    end
    
    save(filePath);
    
end

%% STATS

regressor = [];
dependentVariable = [];
    
for rep = 1:numReps
    
    regressor = [regressor std_values];
    dependentVariable = [dependentVariable sim_log(:, rep)];
    
end

X = regressor(:);
Y = dependentVariable(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');

disp(['RT regression: corr ~ std_values, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);


%% SENSITIVITY ANALYSIS FOR TASK AUTOMATICITY a

var_label = 'Task Automaticity a';
plotCorrelationAnalysis_CogSci(std_values, sim_log, var_label);
