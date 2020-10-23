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
v1_range = 0.1:0.1:5;

range = v1_range;
x = experiment.rewards;
y = nan(length(range), length(x));

for i = 1:length(range)
    
    v1 = range(i);
    
    fnc = @(R) v1 * R;
    y(i,:) = fnc(x);
    
end

legendLabel{1} = ['$b_{Si} = ' num2str(range(1)) '$'];
legendLabel{2} = ['$b_{Si} = ' num2str(range(end)) '$'];
legendLocation = 'southeast';

plotFunctionRange(x, y, 'Reward ($)', '$V(O_{correct})$', 'Reward Sensitivity', legendLabel, legendLocation, 1);

%% SENSITIVITY ANALYSIS FOR REWARD SENSITIVITY

overwrite = 0;
runFullSweep = 1;

% define parameter space
v1_range = 0.1:0.1:5;
v2_range = 0.1:0.1:5;

assumed_v1 = 1;
assumed_v2 = assumed_v1;

% generate log file name
logfileName = ['rewardSensitivity' ...
                        measureOfFit '_' ... 
                        num2str(v1_range(1)*100) '_'  num2str(v1_range(end)*100) '_' ...
                        num2str(v2_range(1)*100) '_'  num2str(v2_range(end)*100) '_'...
                        num2str(assumed_v1*100)];
   
filePath = [logFolderName '/' logfileName '.mat'];

% check if log file exists,   

if exist(filePath, 'file') == 2  && ~overwrite     % if log file exists load it
    load(filePath);

else                                         % if log file doesn't exist, generate it 

    goodnessOfFit  = nan(length(v1_range), length(v2_range));
    c_diff = nan(length(v1_range), length(v2_range));
    
    progress = 0;

    % run search
    disp('start parameter search for reward sensitivity');
    for v1_idx = 1:length(v1_range)

        v1 = v1_range(v1_idx);

        for v2_idx = 1:length(v2_range)

            v2 = v2_range(v2_idx);

            % set tested outcome probability function
            
            agent.valueFnc1 = @(R) v1 * R;
            agent.valueFnc2 = @(R) v2 * R ;
            
            experiment.assumedValueFnc1 = @(R) assumed_v1 * R;
            experiment.assumedValueFnc2 = @(R) assumed_v2 * R;

            % run cost estimation experiment
            c_searchSpace = -1:0.1:15;
            offset_searchSpace = -15:0.1:6;
            [c1_hat, c2_hat, offset1_hat, offset2_hat, estimatedCostFunction1, estimatedCostFunction2, estimatedControlSignals1, estimatedControlSignals2]  = runEstimationExperiment(agent, experiment, c_searchSpace, offset_searchSpace, runFullSweep);

            % compute goodness of fit
            [c_data, offset_data, sample_data] = packEstimationResults(c1, c2, c1_hat, c2_hat, offset1, offset2, offset1_hat, offset2_hat, estimatedCostFunction1, estimatedCostFunction2, estimatedControlSignals1, estimatedControlSignals2, agent.controlSignalSpace);
            [goodnessOfFit(v1_idx, v2_idx), c_diff(v1_idx, v2_idx)] = computeGoodnessOfFit(c_data, offset_data, sample_data, measureOfFit);

            % compute & display computation progress
            progress = ((v1_idx-1)*length(v2_range) + v2_idx)  / (length(v1_range) * length(v2_range)) * 100;
            disp([num2str(progress) '% completed']);
        end

    end

    save(filePath);
end

c_diff = c_diff';
goodnessOfFit = c_diff;

%% plot sensitivity analysis results

offset = 1:15:length(v1_range);
distanceMeasure = 1:15:length(v2_range);

offsetTicks = v1_range(offset); % - assumed_v1;
distanceMeasureTicks = v2_range(distanceMeasure); % - assumed_v2;

offsetLabel = {'True Reward Sensitivity', 'of Agent 1'};
distanceMeasureLabel = {'True Reward Sensitivity', 'of Agent 2'};
titleLabel = '';

true_ratio = 1;
offset_param = find(abs(v1_range-assumed_v1) == min(abs(v1_range-assumed_v1)));
ratio_param = find(abs(v2_range-assumed_v2) == min(abs(v2_range-assumed_v2)));
assumedParameterization = [offset_param ratio_param];

plotSensitivityAnalysis_CogSci(offset, distanceMeasure, goodnessOfFit, c_diff, measureOfFit, offsetTicks, distanceMeasureTicks, offsetLabel, distanceMeasureLabel, c1, c2, assumedParameterization, titleLabel)
