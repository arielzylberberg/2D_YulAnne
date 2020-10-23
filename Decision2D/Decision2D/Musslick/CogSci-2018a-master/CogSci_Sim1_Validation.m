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
numFncFormConditions = 1;       % number of standard deviation conditions
numReps = 1;                     % number of simulation repetitions per standard deviation conditions

std_range = [0 0];                  % range of tested standard deviations
control_cost_range = [1 4];    % range of tested control cost values

functionalForms = [1 2 3];  % 1 - exponential, 2 - quadratic, 3 -  linear
functionalForm_labels = {'exponential', 'quadratic', 'linear'};
control_cost_values = linspace(control_cost_range(1), control_cost_range(2), numSubj);

%%% DEFINE DEFAULT AGENT

% parameters
default_controlCost = 1;
default_taskAutomaticity = -7.5;
default_controlEfficacy = 15;
default_accuracyBias = 0;
default_rewardSensitivity = 1;

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

% correct offset ...if set to 1, then non-zero integrated cost estimates are corrected under the assumption that offset is 0 (this is a reasonable assumption since offset does not have an effect on control signal allocation policy)
correctOffset = 1;

%% Simulation Loop

overwrite = 1;

% generate log file name
logfileName = ['CogSci_Sim1_Validation_' ...
                        num2str(control_cost_range(1)*100) '_' num2str(control_cost_range(end)*100) '_'...
                        num2str(numReps) '_' ...
                        num2str(numSubj)];
                    
filePath = [logFolderName '/' logfileName '.mat'];

if exist(filePath, 'file') == 2  && ~overwrite   % if log file exists load it
    load(filePath);
else                                         % if log file doesn't exist, generate it 

    sim_log_c_hat= nan(numFncFormConditions, numReps, length(control_cost_values));
    sim_log_estimCostFnc = nan(numFncFormConditions, numReps, length(control_cost_values), length(experiment.rewards));
    sim_log_estimCtrlSig = nan(numFncFormConditions, numReps, length(control_cost_values), length(experiment.rewards));
    

    % for each standard_deviationcondition
    for ff_cond_idx = 1:length(functionalForms) 

        functionalForm = functionalForms(ff_cond_idx);

        % for each repetition
        for rep = 1:numReps

            c_hat_log = nan(1, numSubj);     % estimated control cost parameters

            % for each subject
            for subj = 1:numSubj

                % pick control cost value
                c = control_cost_values(subj);
                
                switch functionalForm
                    
                    case 1 % expnential
                        agent.costFnc = @(u) exp(c * u) - 1;
                    case 2 % quadratic
                        agent.costFnc = @(u) c * u.^2;
                    case 3 % linear
                        agent.costFnc = @(u) c * u;
                end

                [c_hat, offset_hat, estimatedCostFunction, estimatedControlSignals] = runCorrelationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, functionalForm_labels(functionalForm), correctOffset);
                c_hat_log(subj) = c_hat;
                sim_log_estimCostFnc(ff_cond_idx, rep, subj, :) = estimatedCostFunction;
                sim_log_estimCtrlSig(ff_cond_idx, rep, subj, :) = estimatedControlSignals;

            end
            
            % log outcomes
            sim_log_c_hat(ff_cond_idx, rep, :) = c_hat_log;

        end

        disp(['progress: ' num2str(ff_cond_idx) '/' num2str(length(functionalForms) )]);

    end
    
    save(filePath);
    
end

%% STATS

for ff_cond_idx = 1:length(functionalForms) 
    
    regressor = [];
    dependentVariable = [];
    functionalForm = functionalForms(ff_cond_idx);
    
    for rep = 1:numReps
        
        control_cost_values_used = control_cost_values;
        control_cost_values_estimated = sim_log_c_hat(ff_cond_idx, rep, :);
        removeIdx = find(isnan(control_cost_values_estimated));
        control_cost_values_estimated(removeIdx) = [];
        control_cost_values_used(removeIdx) = [];

        regressor = [regressor control_cost_values_estimated];
        dependentVariable = [dependentVariable control_cost_values_used];
        
    end
    
    X = regressor(:);
    Y = dependentVariable(:);
    lm = fitlm(X,Y,'linear');
    an = anova(lm,'summary');

    disp(['RT regression: RT ~ ' functionalForm_labels{functionalForm} ' cost param, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                        ', t(' num2str(an{1, 2}) ...
                                                                                                        ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                        ', p = ' num2str(lm.Coefficients.pValue(2))]);


end


%% PLOT FUNCTIONS
plotsettings;

c = 2;
subj = 1;
rep = 1;
c_idx = find(control_cost_values == c);

legendLabels = {'Exponential', 'Quadratic', 'Linear'};

controlSignalSpace = agent.controlSignalSpace;

costFnc1 = @(u) exp(c * u) - 1;
costFnc2 = @(u) c * u.^2;
costFnc3 = @(u) c * u;
estimatedCostFunction1 = squeeze(sim_log_estimCostFnc(1, rep, c_idx, :))';
estimatedControlSignals1 = squeeze(sim_log_estimCtrlSig(1, rep, c_idx, :))';
estimatedCostFunction2 = squeeze(sim_log_estimCostFnc(2, rep, c_idx, :))';
estimatedControlSignals2 = squeeze(sim_log_estimCtrlSig(2, rep, c_idx, :))';
estimatedCostFunction3 = squeeze(sim_log_estimCostFnc(3, rep, c_idx, :))';
estimatedControlSignals3 = squeeze(sim_log_estimCtrlSig(3, rep, c_idx, :))';

% plot cost functions
colors = [plotSettings.colors(cContrast1, :); ...
               plotSettings.colors(cContrast2, :); ...
               plotSettings.colors(cContrast3, :)];
           
costPlotData = [costFnc1(controlSignalSpace); ...
                         costFnc2(controlSignalSpace); ...
                         costFnc3(controlSignalSpace);];
                     
estimatedCostPlotData = [estimatedCostFunction1; ...
                                         estimatedCostFunction2; ...
                                         estimatedCostFunction3;];
                                     
optimalControlSignalPlotData = [estimatedControlSignals1; ...
                                                    estimatedControlSignals2; ...
                                                    estimatedControlSignals3;];
                                                
plotCostFnc_CogSci(controlSignalSpace, costPlotData, colors, optimalControlSignalPlotData, estimatedCostPlotData, [], [], legendLabels);


%% PLOT EVC SURFACE
c = 3;
costFnc1 = @(u) exp(c * u) - 1;
[outcomeProbabilities1, optimalControlSignals1, EVCMap1, maxEVC1] = runEVCAgent(agent.controlSignalSpace, agent.outcomeProbabilityFnc, agent.valueFnc, costFnc1, experiment.rewards);

plotEVCMap_CogSci(agent.controlSignalSpace, experiment.rewards, EVCMap1, optimalControlSignals1, maxEVC1, 1)

