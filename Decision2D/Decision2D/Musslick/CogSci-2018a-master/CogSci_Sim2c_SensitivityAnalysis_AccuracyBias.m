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

% measureOfFit = 'MSE';
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
agent.valueFnc1 = @(R) R;
agent.valueFnc2 = @(R) R;

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
b1_range = -40:2:40;

range = b1_range;
x = experiment.rewards;
y = nan(length(range), length(x));

for i = 1:length(range)
    
    b1 = range(i);
    
    fnc = @(R) R + b1;
    y(i,:) = fnc(x);
    
end

legendLabel{1} = ['$b_{Si} = ' num2str(range(1)) '$'];
legendLabel{2} = ['$b_{Si} = ' num2str(range(end)) '$'];
legendLocation = 'southeast';

plotFunctionRange(x, y, 'Reward ($)', '$V(O_{correct})$', 'Accuracy Bias', legendLabel, legendLocation, 1);


%% SENSITIVITY ANALYSIS FOR ACCURACY BIAS

overwrite = 0;
runFullSweep = 1;

% define parameter space
b1_range = -40:2:40;
b2_range = -40:2:40;

assumed_b1 = 0;
assumed_b2 = assumed_b1;

% generate log file name
logfileName = ['accuracyBias' ...
                        measureOfFit '_' ... 
                        num2str(b1_range(1)*100) '_'  num2str(b1_range(end)*100) '_' ...
                        num2str(b2_range(1)*100) '_'  num2str(b2_range(end)*100) '_'...
                        num2str(assumed_b1*100)];
   
filePath = [logFolderName '/' logfileName '.mat'];

% check if log file exists,   

if exist(filePath, 'file') == 2  && ~overwrite      % if log file exists load it
    load(filePath);

else                                         % if log file doesn't exist, generate it 

    goodnessOfFit  = nan(length(b1_range), length(b2_range));
    c_diff = nan(length(b1_range), length(b2_range));
    
    progress = 0;

    % run search
    disp('start parameter search for accuracy bias');
    for b1_idx = 1:length(b1_range)

        b1 = b1_range(b1_idx);

        for b2_idx = 1:length(b2_range)

            b2 = b2_range(b2_idx);

            % set tested outcome probability function
            
            agent.valueFnc1 = @(R) R + b1;
            agent.valueFnc2 = @(R) R + b2;
            
            experiment.assumedValueFnc1 = @(R) R + assumed_b1;
            experiment.assumedValueFnc2 = @(R) R + assumed_b2;

            % run cost estimation experiment
            c_searchSpace = -1:0.1:15;
            offset_searchSpace = -10:0.1:30;
            [c1_hat, c2_hat, offset1_hat, offset2_hat, estimatedCostFunction1, estimatedCostFunction2, estimatedControlSignals1, estimatedControlSignals2]  = runEstimationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, runFullSweep);

            % compute goodness of fit
            [c_data, offset_data, sample_data] = packEstimationResults(c1, c2, c1_hat, c2_hat, offset1, offset2, offset1_hat, offset2_hat, estimatedCostFunction1, estimatedCostFunction2, estimatedControlSignals1, estimatedControlSignals2, agent.controlSignalSpace);
            [goodnessOfFit(b1_idx, b2_idx), c_diff(b1_idx, b2_idx)] = computeGoodnessOfFit(c_data, offset_data, sample_data, measureOfFit);

            % compute & display computation progress
            progress = ((b1_idx-1)*length(b2_range) + b2_idx)  / (length(b1_range) * length(b2_range)) * 100;
            disp([num2str(progress) '% completed']);
        end

    end

    save(filePath);
end

c_diff = c_diff';
goodnessOfFit = c_diff;

%% plot sensitivity analysis results

offset = 1:10:length(b1_range);
distanceMeasure = 1:10:length(b2_range);

offsetTicks = b1_range(offset); % - assumed_b1;
distanceMeasureTicks = b2_range(distanceMeasure); % - assumed_b2;

offsetLabel = {'True Accuracy Bias', 'of Agent 1'};
distanceMeasureLabel = {'True Accuract Bias', 'of Agent 2'};
titleLabel = '';

true_ratio = 1;
offset_param = find(abs(b1_range-assumed_b1) == min(abs(b1_range-assumed_b1)));
ratio_param = find(abs(b2_range-assumed_b2) == min(abs(b2_range-assumed_b2)));
assumedParameterization = [offset_param ratio_param];

plotSensitivityAnalysis_CogSci(offset, distanceMeasure, goodnessOfFit, c_diff, measureOfFit, offsetTicks, distanceMeasureTicks, offsetLabel, distanceMeasureLabel, c1, c2, assumedParameterization, titleLabel)
