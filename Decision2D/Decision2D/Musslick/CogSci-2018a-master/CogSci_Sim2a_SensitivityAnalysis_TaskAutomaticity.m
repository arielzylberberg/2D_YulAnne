% Sensitivity Analysis
%
%
%

clear all;
clc;

addpath('main');

%%% META PARAMETERS
c1 = 2;
c2 = 3;
offset1 = -1;
offset2 = -1;

measureOfFit = 'delta_c_hat';

logFolderName = 'logfiles';

%%% DEFINE DEFAULT AGENT

% define space of control signals
agent.controlSignalSpace = 0:0.01:1;

% define cost functions for both subjects
agent.costFnc1 = @(u) exp(c1 * u) - 1;
agent.costFnc2 = @(u) exp(c2 * u) - 1;

% define outcome probability function for both subjects
agent.outcomeProbabilityFnc1 = @(u) 1./(1+exp(-15*u - (-7.5)));
agent.outcomeProbabilityFnc2 = @(u) 1./(1+exp(-15*u - (-7.5)));

% define value function
agent.valueFnc1 = @(u) u;
agent.valueFnc2 = @(u) u;

%%% EXPERIMENT & FIT PROCEDURE

% set up reward manipulations
experiment.rewards = 0:1:100;

% set assumed control signal space
experiment.assumedControlSignalSpace1 = agent.controlSignalSpace;
experiment.assumedControlSignalSpace2 = agent.controlSignalSpace;

% set assumed outcome probability function for both subjects
experiment.assumedOutcomeProbabilityFnc1 = agent.outcomeProbabilityFnc1;
experiment.assumedOutcomeProbabilityFnc2 = agent.outcomeProbabilityFnc2;

% set assumed value function for both subjects
experiment.assumedValueFnc1 = agent.valueFnc1;
experiment.assumedValueFnc2 = agent.valueFnc2;

%% PLOT RANGE OF TESTED VALUES
a1_range = -16:0.5:1;

range = a1_range;
x = agent.controlSignalSpace;
y = nan(length(range), length(x));

for i = 1:length(range)
    
    a1 = range(i);
    
    fnc = @(u) 1./(1+exp(-15*u - a1));
    y(i,:) = fnc(x);
    
end

legendLabel{1} = ['$a_{Si} = ' num2str(range(1)) '$'];
legendLabel{2} = ['$a_{Si} = ' num2str(range(end)) '$'];
legendLocation = 'south';

plotFunctionRange(x, y, 'Control Signal Intensity u', '$P(O_{correct} | u, S)$', 'Task Automaticity', legendLabel, legendLocation, 1);


%% SENSITIVITY ANALYSIS FOR TASK AUTOMATICITY a

overwrite = 0;
runFullSweep = 1;

% define parameter space
a1_range = -16:0.5:1;
a2_range = -16:0.5:1;

assumed_a1 = -7.5;
assumed_a2 = assumed_a1;

% generate log file name
logfileName = ['taskAutomaticity_' ...
                        measureOfFit '_' ... 
                        num2str(a1_range(1)*100) '_' num2str(a1_range(end)*100) '_' ...
                        num2str(a2_range(1)*100) '_' num2str(a2_range(end)*100) '_'...
                        assumed_a1*100];
   
filePath = [logFolderName '/' logfileName '.mat'];

% check if log file exists,   

if exist(filePath, 'file') == 2  && ~overwrite   % if log file exists load it
    load(filePath);

else                                         % if log file doesn't exist, generate it 

    goodnessOfFit  = nan(length(a1_range), length(a2_range));
    c_diff = nan(length(a1_range), length(a2_range));
    
    progress = 0;

    % run search
    disp('start parameter search for task automaticity');
    for a1_idx = 1:length(a1_range)

        a1 = a1_range(a1_idx);

        for a2_idx = 1:length(a2_range)

            a2 = a2_range(a2_idx);

            % set tested outcome probability function
            
            agent.outcomeProbabilityFnc1 = @(u) 1./(1+exp(-15*u - a1));
            agent.outcomeProbabilityFnc2 = @(u) 1./(1+exp(-15*u - a2));
            
            experiment.assumedOutcomeProbabilityFnc1 = @(u) 1./(1+exp(-15*u - assumed_a1));
            experiment.assumedOutcomeProbabilityFnc2 = @(u) 1./(1+exp(-15*u - assumed_a2));

            % run cost estimation experiment
            c_searchSpace = -1:0.1:15;
            offset_searchSpace = -15:0.1:2;
            [c1_hat, c2_hat, offset1_hat, offset2_hat, estimatedCostFunction1, estimatedCostFunction2, estimatedControlSignals1, estimatedControlSignals2]  = runEstimationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, runFullSweep);

            % compute goodness of fit
            [c_data, offset_data, sample_data] = packEstimationResults(c1, c2, c1_hat, c2_hat, offset1, offset2, offset1_hat, offset2_hat, estimatedCostFunction1, estimatedCostFunction2, estimatedControlSignals1, estimatedControlSignals2, agent.controlSignalSpace);
            [goodnessOfFit(a1_idx, a2_idx), c_diff(a1_idx, a2_idx)] = computeGoodnessOfFit(c_data, offset_data, sample_data, measureOfFit);

            % compute & display computation progress
            progress = ((a1_idx-1)*length(a2_range) + a2_idx)  / (length(a1_range) * length(a2_range)) * 100;
            disp([num2str(progress) '% completed, ' num2str(a1) '_' num2str(a2)]);
        end

    end

    save(filePath);
end

c_diff = c_diff';
goodnessOfFit = c_diff;

%% plot sensitivity analysis results

offset = 1:8:length(a1_range);
distanceMeasure = 1:8:length(a2_range);

offsetTicks = a1_range(offset); % - assumed_a1;
distanceMeasureTicks = a2_range(distanceMeasure); % - assumed_a2;

offsetLabel = {'True Task Automaticity' 'of Agent 1'};
distanceMeasureLabel = {'True Task Automaticity' , 'of Agent 2'};
titleLabel = '';

true_ratio = 1;
offset_param = find(abs(a1_range-assumed_a1) == min(abs(a1_range-assumed_a1)));
ratio_param = find(abs(a2_range-assumed_a2) == min(abs(a2_range-assumed_a2)));
assumedParameterization = [offset_param ratio_param];

plotSensitivityAnalysis_CogSci(offset, distanceMeasure, goodnessOfFit, c_diff, measureOfFit, offsetTicks, distanceMeasureTicks, offsetLabel, distanceMeasureLabel, c1, c2, assumedParameterization, titleLabel)
