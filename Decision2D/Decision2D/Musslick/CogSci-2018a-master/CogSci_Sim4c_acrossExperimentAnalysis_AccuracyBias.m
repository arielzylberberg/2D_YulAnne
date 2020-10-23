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
numCorrConditions = 10;       % number of standard deviation conditions
numReps = 10;                     % number of simulation repetitions per standard deviation conditions

corr_range = [0 1];                  % range of tested standard deviations
control_cost_range = [1 4];    % range of tested control cost values
control_cost_scale = 1.5;
default_control_cost = 2.5;

corr_values = linspace(corr_range(1), corr_range(2), numCorrConditions);
control_cost_values = linspace(control_cost_range(1), control_cost_range(2), numSubj);

%%% DEFINE DEFAULT AGENT

% parameters
default_controlCost = 1;
default_taskAutomaticity = -7.5;
default_controlEfficacy = 15;
default_accuracyBias = 0;
default_rewardSensitivity = 2;

taskAutomaticity_scale = 6;
rewardSensitivity_scale = 2;
accuracyBias_scale = 10;

% define space of control signals
agent.controlSignalSpace = 0:0.01:1;

% define cost functions for both subjects
agent.costFnc = @(u) exp(default_controlCost * u) - 1;

% define outcome probability function for both subjects
agent.outcomeProbabilityFnc = @(u) 1./(1+exp(-default_controlEfficacy*u - default_taskAutomaticity));

% define value function
agent.valueFnc = @(u) default_rewardSensitivity * u + default_accuracyBias;

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
c_searchSpace = -1:0.1:15;
offset_searchSpace = -13:0.1:4;

%% Simulation Loop

overwrite = 0 ;
runFullSweep = 1;

% generate log file name
logfileName = ['accuracyBias_acrossExpAnalysis_' ...
                        num2str(corr_range(1)*100) '_' num2str(corr_range(end)*100) '_' ...
                        num2str(control_cost_range(1)*100) '_' num2str(control_cost_range(end)*100) '_'...
                        num2str(numReps) '_' ...
                        num2str(numSubj)];
                    
filePath = [logFolderName '/' logfileName '.mat'];

if exist(filePath, 'file') == 2 && ~overwrite    % if log file exists load it
    load(filePath);
else                                         % if log file doesn't exist, generate it 

    sim_log = nan(numCorrConditions, numReps);

    % for each correlationcondition
    for corr_cond_idx = 1:length(corr_values) 

        correlation= corr_values(corr_cond_idx);

        % for each repetition
        for rep = 1:numReps
            
            % generate two uncorrelated vectors of participant's control costs
            controlCostExp1 = nan(1, numSubj);
            controlCostExp2 = nan(1, numSubj);
            for subj = 1:numSubj
                [Za, Zb] = randomcorrelatednum(0);
                controlCostExp1(subj) = Za * control_cost_scale + default_control_cost;
                controlCostExp2(subj) = Zb * control_cost_scale + default_control_cost;
            end
            
            % generate two correlated vectors of participant's motivational variable
            accuracyBiasExp1 = nan(1, numSubj);
            accuracyBiasExp2 = nan(1, numSubj);
            for subj = 1:numSubj
                [Za, Zb] = randomcorrelatednum(correlation);
                accuracyBiasExp1(subj) = Za * accuracyBias_scale + default_accuracyBias;
                accuracyBiasExp2(subj) = Zb * accuracyBias_scale + default_accuracyBias;
            end
            
            c_hat_log_Exp1 = nan(1, numSubj);     % estimated control cost parameters
            c_hat_log_Exp2 = nan(1, numSubj);     % estimated control cost parameters

            % for each subject
            for subj = 1:numSubj

                % EXPERIMENT 1
                
                % pick control cost value
                c = controlCostExp1(subj);
                agent.costFnc = @(u) exp(c * u) - 1;

                % vary motivational parameter
                accuracyBias = accuracyBiasExp1(subj);
                agent.valueFnc = @(u) default_rewardSensitivity * u + accuracyBias;

                [c_hat, offset_hat, estimatedCostFunction, estimatedControlSignals] = runCorrelationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, [], 0, runFullSweep);
                c_hat_log_Exp1(subj) = c_hat;
                
                % EXPERIMENT 2
                
                % pick control cost value
                c = controlCostExp2(subj);
                agent.costFnc = @(u) exp(c * u) - 1;

                % vary motivational parameter
                accuracyBias = accuracyBiasExp2(subj);
                agent.valueFnc = @(u) default_rewardSensitivity * u + accuracyBias;

                [c_hat, offset_hat, estimatedCostFunction, estimatedControlSignals] = runCorrelationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, [], 0, runFullSweep);
                c_hat_log_Exp2(subj) = c_hat;

            end
            
            % remove nan conditions
            removeIdx = isnan(c_hat_log_Exp1) | isnan(c_hat_log_Exp2);
            c_hat_log_Exp1(removeIdx) = [];
            c_hat_log_Exp2(removeIdx) = [];
            
            % perform correlation analysis
            switch similarityMetric
                case 'corr'
                    R = corr([c_hat_log_Exp1', c_hat_log_Exp2']);
                    similarity = R(2);
            end

            sim_log(corr_cond_idx, rep) = similarity;

        end

        disp(['progress: ' num2str(corr_cond_idx) '/' num2str(length(corr_values) )]);

    end
    
    save(filePath);
    
end

%% STATS

regressor = [];
dependentVariable = [];
    
for rep = 1:numReps
    
    regressor = [regressor corr_values];
    dependentVariable = [dependentVariable sim_log(:, rep)];
    
end

X = regressor(:);
Y = dependentVariable(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');

disp(['RT regression: corr_cost ~ corr_motivation_var, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);


%% SENSITIVITY ANALYSIS FOR ACCURACY BIAS

var_label = 'Accuracy Bias b';
plotBetweenExperimentAnalysis_CogSci(corr_values, sim_log, var_label);
