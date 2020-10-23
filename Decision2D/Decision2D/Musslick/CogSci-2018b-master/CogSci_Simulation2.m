%% TASK ENVIRONMENT
clear all;
close all;
clc;

addpath('main');

% task environment
numTrialsPerMiniBlock = 6;
numMiniBlocks = 100;
congruentTrials = 1;
numSequences = 10;

% Create structure with params for simulation
ddmpParams.bias = 0.5;                % DDM bias (centered at 0.5)
ddmpParams.c = 0.04;                   % DDM noise
ddmpParams.T0 = 0.2;                  % DDM non-decision time
ddmpParams.z = 0.0475;              % DDM threshold
tau = 0.9;                                    % rate of integration for net input
inputWeight = 1;

% conditions
fixedGain = 1.2;
frequencies = [0.1:0.2:0.9];

% FIND OPTIMAL GAIN 

% grid search parameters
gains = linspace(1, 2, 50);
thresholds =   ddmpParams.z;        
parameters = combvec(gains, thresholds);

% log components
frequency_log = nan(length(frequencies), numSequences);
optimal_gain = nan(length(frequencies), numSequences);
optimizationCriterionArray = zeros(length(frequencies), numSequences,size(parameters,2));
overallRT_global = nan(length(frequencies), numSequences);
overallER_global = nan(length(frequencies), numSequences);
incongruencyCostRT_global = nan(length(frequencies), numSequences);
incongruencyCostER_global = nan(length(frequencies), numSequences);
switchCostRT_global = nan(length(frequencies), numSequences);
switchCostER_global = nan(length(frequencies), numSequences);
allDrift = nan(length(frequencies), numSequences, numMiniBlocks*numTrialsPerMiniBlock);
transition = nan(length(frequencies), numSequences, numMiniBlocks*numTrialsPerMiniBlock);
incongruency = nan(length(frequencies), numSequences, numMiniBlocks*numTrialsPerMiniBlock);

for freqIdx = 1:length(frequencies)

    for sequenceIdx = 1:numSequences
        frequency = frequencies(freqIdx);
        [~, sequenceFile] = generateRandomSequence(numTrialsPerMiniBlock, numMiniBlocks, frequency, congruentTrials);

        disp(['Optimizing Switch Frequency ' num2str(frequency)]);
        disp('+++++++');

        bestParams = [];
        bestVal = 10e16;

        for i = 1:size(parameters,2)
            p = transpose(parameters(:,i));
            [optimizationCriterion, performance, allRR, allmeanRT, allmeanAccuracy, allActivation] = gainInvestigation_simulateSequence(p, ddmpParams, sequenceFile, tau, inputWeight);
            optimizationCriterionArray(freqIdx, sequenceIdx, i) = optimizationCriterion;
            if(optimizationCriterion < bestVal)
                bestVal = optimizationCriterion;
                bestParams = p;
    %             disp('got better'); 
    %             disp(p);
    %             disp(optimizationCriterion);
                bestAllMeanRT = allmeanRT;
                bestAllMeanAccuracy = allmeanAccuracy;
                bestPerformance = performance;
            end
        end

        optimal_gain(freqIdx,sequenceIdx) = bestParams(1);
        frequency_log(freqIdx,sequenceIdx) = frequency;

        % perform analysis
        gain = optimal_gain(freqIdx,sequenceIdx);

        [overallRT_global(freqIdx,sequenceIdx), overallER_global(freqIdx,sequenceIdx), incongruencyCostRT_global(freqIdx,sequenceIdx), incongruencyCostER_global(freqIdx,sequenceIdx), switchCostRT_global(freqIdx,sequenceIdx), switchCostER_global(freqIdx,sequenceIdx), ...
         overallRT_local(freqIdx,sequenceIdx), overallER_local(freqIdx,sequenceIdx), incongruencyCostRT_local(freqIdx,sequenceIdx), incongruencyCostER_local(freqIdx,sequenceIdx), switchCostRT_local(freqIdx,sequenceIdx), switchCostER_local(freqIdx,sequenceIdx), ...
         relActivation_global(freqIdx,sequenceIdx), relActivation_local(freqIdx,sequenceIdx), irrelActivation_global(freqIdx,sequenceIdx), irrelActivation_local(freqIdx,sequenceIdx),  ...
         allDrift, transition, incongruency] = CogSci_Simulation2_analysisLocal(gain, ddmpParams, sequenceFile, tau, inputWeight);

        % NOW compute performance for fixed gain


        [overallRT_global_unoptimized(freqIdx, sequenceIdx), overallER_global_unoptimized(freqIdx, sequenceIdx), incongruencyCostRT_global_unoptimized(freqIdx, sequenceIdx), incongruencyCostER_global_unoptimized(freqIdx, sequenceIdx), switchCostRT_global_unoptimized(freqIdx, sequenceIdx), switchCostER_global_unoptimized(freqIdx, sequenceIdx), ...
         overallRT_local_unoptimized(freqIdx, sequenceIdx), overallER_local_unoptimized(freqIdx, sequenceIdx), incongruencyCostRT_local_unoptimized(freqIdx, sequenceIdx), incongruencyCostER_local_unoptimized(freqIdx, sequenceIdx), switchCostRT_local_unoptimized(freqIdx, sequenceIdx), switchCostER_local_unoptimized(freqIdx, sequenceIdx), ...
         relActivation_global_unoptimized(freqIdx,sequenceIdx), relActivation_local_unoptimized(freqIdx,sequenceIdx), irrelActivation_global_unoptimized(freqIdx,sequenceIdx), irrelActivation_local_unoptimized(freqIdx,sequenceIdx),  ...
         allDrift_unoptimized, transition_unoptimized, incongruency_unoptimized] = CogSci_Simulation2_analysisLocal(fixedGain, ddmpParams, sequenceFile, tau, inputWeight);

    end
end

save(['logfiles/CogSci_Simulation2_' num2str(numSequences) '.mat']);

%% STATS

CogSci_Simulation2_Stats;

% compute performance for optimized gain

optimal_gain_mean = mean(optimal_gain,2);
overallRT_global_mean = mean(overallRT_global,2);
overallER_global_mean = mean(overallER_global, 2);
incongruencyCostRT_global_mean = mean(incongruencyCostRT_global, 2);
incongruencyCostER_global_mean = mean(incongruencyCostER_global, 2);
switchCostRT_global_mean = mean(switchCostRT_global, 2);
switchCostER_global_mean = mean(switchCostER_global, 2);
overallRT_local_mean = mean(overallRT_local,2);
overallER_local_mean = mean(overallER_local, 2);
incongruencyCostRT_local_mean = mean(incongruencyCostRT_local, 2);
incongruencyCostER_local_mean = mean(incongruencyCostER_local, 2);
switchCostRT_local_mean = mean(switchCostRT_local, 2);
switchCostER_local_mean = mean(switchCostER_local, 2);
relActivation_global_mean = mean(relActivation_global,2);
irrelActivation_global_mean = mean(irrelActivation_global,2);
relActivation_local_mean = mean(relActivation_local,2);
irrelActivation_local_mean = mean(irrelActivation_local,2);


optimal_gain_sem = std(optimal_gain,[],2)/sqrt(numSequences);
overallRT_global_sem = std(overallRT_global,[],2)/sqrt(numSequences);
overallER_global_sem = std(overallER_global,[],2)/sqrt(numSequences);
incongruencyCostRT_global_sem = std(incongruencyCostRT_global,[],2)/sqrt(numSequences);
incongruencyCostER_global_sem = std(incongruencyCostER_global,[],2)/sqrt(numSequences);
switchCostRT_global_sem = std(switchCostRT_global,[],2)/sqrt(numSequences);
switchCostER_global_sem = std(switchCostER_global,[],2)/sqrt(numSequences);
overallRT_local_sem = std(overallRT_local,[],2)/sqrt(numSequences);
overallER_local_sem = std(overallER_local,[],2)/sqrt(numSequences);
incongruencyCostRT_local_sem = std(incongruencyCostRT_local,[],2)/sqrt(numSequences);
incongruencyCostER_local_sem = std(incongruencyCostER_local,[],2)/sqrt(numSequences);
switchCostRT_local_sem = std(switchCostRT_local,[],2)/sqrt(numSequences);
switchCostER_local_sem = std(switchCostER_local,[],2)/sqrt(numSequences);
relActivation_global_sem = std(relActivation_global,[],2)/sqrt(numSequences);
irrelActivation_global_sem = std(irrelActivation_global,[],2)/sqrt(numSequences);
relActivation_local_sem = std(relActivation_local,[],2)/sqrt(numSequences);
irrelActivation_local_sem = std(irrelActivation_local,[],2)/sqrt(numSequences);

% NOW compute performance for fixed gain

overallRT_global_unoptimized_mean = mean(overallRT_global_unoptimized,2);
overallER_global_unoptimized_mean = mean(overallER_global_unoptimized, 2);
incongruencyCostRT_global_unoptimized_mean = mean(incongruencyCostRT_global_unoptimized, 2);
incongruencyCostER_global_unoptimized_mean = mean(incongruencyCostER_global_unoptimized, 2);
switchCostRT_global_unoptimized_mean = mean(switchCostRT_global_unoptimized, 2);
switchCostER_global_unoptimized_mean = mean(switchCostER_global_unoptimized, 2);
overallRT_local_unoptimized_mean = mean(overallRT_local_unoptimized,2);
overallER_local_unoptimized_mean = mean(overallER_local_unoptimized, 2);
incongruencyCostRT_local_unoptimized_mean = mean(incongruencyCostRT_local_unoptimized, 2);
incongruencyCostER_local_unoptimized_mean = mean(incongruencyCostER_local_unoptimized, 2);
switchCostRT_local_unoptimized_mean = mean(switchCostRT_local_unoptimized, 2);
switchCostER_local_unoptimized_mean = mean(switchCostER_local_unoptimized, 2);
relActivation_global_unoptimized_mean  = mean(relActivation_global_unoptimized ,2);
irrelActivation_global_unoptimized_mean  = mean(irrelActivation_global_unoptimized ,2);
relActivation_local_unoptimized_mean = mean(relActivation_local_unoptimized,2);
irrelActivation_local_unoptimized_mean = mean(irrelActivation_local_unoptimized,2);

overallRT_global_unoptimized_sem = std(overallRT_global_unoptimized,[], 2)/sqrt(numSequences);
overallER_global_unoptimized_sem = std(overallER_global_unoptimized,[],2)/sqrt(numSequences);
incongruencyCostRT_global_unoptimized_sem = std(incongruencyCostRT_global_unoptimized,[],2)/sqrt(numSequences);
incongruencyCostER_global_unoptimized_sem = std(incongruencyCostER_global_unoptimized,[],2)/sqrt(numSequences);
switchCostRT_global_unoptimized_sem = std(switchCostRT_global_unoptimized,[],2)/sqrt(numSequences);
switchCostER_global_unoptimized_sem = std(switchCostER_global_unoptimized,[],2)/sqrt(numSequences);
overallRT_local_unoptimized_sem = std(overallRT_local_unoptimized,[],2)/sqrt(numSequences);
overallER_local_unoptimized_sem = std(overallER_local_unoptimized,[],2)/sqrt(numSequences);
incongruencyCostRT_local_unoptimized_sem = std(incongruencyCostRT_local_unoptimized,[],2)/sqrt(numSequences);
incongruencyCostER_local_unoptimized_sem = std(incongruencyCostER_local_unoptimized,[],2)/sqrt(numSequences);
switchCostRT_local_unoptimized_sem = std(switchCostRT_local_unoptimized,[],2)/sqrt(numSequences);
switchCostER_local_unoptimized_sem = std(switchCostER_local_unoptimized,[],2)/sqrt(numSequences);
relActivation_global_unoptimized_sem  = std(relActivation_global_unoptimized ,[], 2)/sqrt(numSequences);
irrelActivation_global_unoptimized_sem  = std(irrelActivation_global_unoptimized ,[], 2)/sqrt(numSequences);
relActivation_local_unoptimized_sem = std(relActivation_local_unoptimized,[], 2)/sqrt(numSequences);
irrelActivation_local_unoptimized_sem = std(irrelActivation_local_unoptimized,[], 2)/sqrt(numSequences);

%% PLOTS

colors = [51 153 255; 255 153 51; 200 200 200]/255;
fontsize = 15;
markerSize = 10;

xdata = frequencies;

plotSettings.colors = [253 120 21; ... % orange
              31 104 172; ... % blue
              44 155 37; ... % green
              0     0   0  ; ... % black
            142 142 142; ... % grey 
            255 255 255; ... % white 
            55 146 171; ... % cyan
            255 0 0;] / 255; % red
        
fig1 = figure(1);
set(fig1, 'Position', [100 700 620 180]);

subplot(1,2,1);
errorbar(xdata, optimal_gain_mean, optimal_gain_sem, 'k', 'LineWidth', 3); hold on;
hold off;
xlim([min(xdata) max(xdata)]);
ylabel({'Optimal Gain'}, 'FontSize', fontsize, 'FontWeight','bold');
xlabel('Task Switch Probability', 'FontSize', fontsize, 'FontWeight','bold');
set(gca, 'FontSize', fontsize, 'FontWeight','bold');

subplot(1,2,2);
errorbar(xdata, relActivation_global_mean, relActivation_global_sem, 'Color', plotSettings.colors(1,:), 'LineWidth', 3); hold on;
errorbar(xdata, irrelActivation_global_mean, irrelActivation_global_sem, 'Color', plotSettings.colors(2,:), 'LineWidth', 3); 
scatter(frequency_log(:)*100, relActivation_global(:), markerSize, plotSettings.colors(1,:)); 
scatter(frequency_log(:)*100, irrelActivation_global(:), markerSize, plotSettings.colors(2,:)); 
hold off;
xlim([min(xdata) max(xdata)]);
ylabel({'Mean Control', 'Signal Intensity '}, 'FontSize', fontsize, 'FontWeight','bold');
xlabel('Task Switch Probability', 'FontSize', fontsize, 'FontWeight','bold');
leg = legend('Relevant Task', 'Irrelevant Task', 'Location', 'west');
rect = [0.60, 0.52, .25, .25];
set(leg, 'Position', rect)
set(leg, 'FontSize', fontsize);
set(gca, 'FontSize', fontsize, 'FontWeight','bold');


