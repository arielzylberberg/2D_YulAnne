%% TASK ENVIRONMENT
clear all;
clc;

addpath('main');

% task environment
numTrialsPerMiniBlock = 6; 
numMiniBlocks = 100; % must be even for counter balancing
frequency = 0.50;
congruentTrials = 1;
numSequences = 10;

% Create structure with params for simulation
ddmpParams.bias = 0.5;                % DDM bias (centered at 0.5)
ddmpParams.c = 0.04;                   % DDM noise
ddmpParams.T0 = 0.2;                  % DDM non-decision time
ddmpParams.z = 0.0475;              % DDM threshold
tau = 0.9;                                    % rate of integration for net input
inputWeight = 1;

%% investigate effect of gain (plots w/white background, vertical)

gains = 0.1:0.1:3;
gains(1) = [];

gain_log = nan(numSequences, length(gains));
overallPerformanceRT_g = nan(numSequences, length(gains));
switchCostsRT_g = nan(numSequences, length(gains));
incongruencyCostsRT_g = nan(numSequences, length(gains));

overallPerformanceER_g = nan(numSequences, length(gains));
switchCostsER_g = nan(numSequences, length(gains));
incongruencyCostsER_g = nan(numSequences, length(gains));

for seqIdx = 1:numSequences
    
    [params, sequenceFile] = generateRandomSequence(numTrialsPerMiniBlock, numMiniBlocks, frequency, congruentTrials);

    miniBlockTrial = params.CTS.Sequence{2}.miniBlockTrial;
    transition = params.CTS.Sequence{2}.TaskTransition;
    task = params.CTS.Sequence{2}.Task;
    incongruency = params.CTS.Sequence{2}.incongruency;

    for gainIdx = 1:length(gains)

        g = gains(gainIdx);
        p = g;
        [optimizationCriterion, performance, allRR, allmeanRT, allmeanAccuracy, allActivation, allDrift] = gainInvestigation_simulateSequence(p, ddmpParams, sequenceFile, tau, inputWeight);

        gain_log(seqIdx, gainIdx) = g;
        
        meanRT = nanmean(allmeanRT(transition == 0 | transition == 1));
        switchRT = nanmean(allmeanRT(transition == 1 & miniBlockTrial == 1));
        repetitionRT = nanmean(allmeanRT(transition == 0 & miniBlockTrial == 1));
        congruentRT = nanmean(allmeanRT(incongruency == 0));
        incongruentRT = nanmean(allmeanRT(incongruency == 1));

        meanER = 1-nanmean(allmeanAccuracy(transition == 0 | transition == 1));
        switchER = 1-nanmean(allmeanAccuracy(transition == 1 & miniBlockTrial == 1));
        repetitionER = 1-nanmean(allmeanAccuracy(transition == 0 & miniBlockTrial == 1));
        congruentER = 1-nanmean(allmeanAccuracy(incongruency == 0 & transition == 0));
        incongruentER = 1-nanmean(allmeanAccuracy(incongruency == 1 & transition == 0));

        overallPerformanceRT_g(seqIdx, gainIdx) = meanRT;
        switchCostsRT_g(seqIdx, gainIdx) = switchRT-repetitionRT;
        incongruencyCostsRT_g(seqIdx, gainIdx) = incongruentRT-congruentRT;

        overallPerformanceER_g(seqIdx, gainIdx) =meanER;
        switchCostsER_g(seqIdx, gainIdx) = switchER-repetitionER;
        incongruencyCostsER_g(seqIdx, gainIdx) = incongruentER-congruentER;
        
        
    end

    disp(['finished sequence ' num2str(seqIdx) '/' num2str(numSequences)]);
end

%% STATS

overallPerformanceRT_g_mean = mean(overallPerformanceRT_g);
switchCostsRT_g_mean = mean(switchCostsRT_g);
incongruencyCostsRT_g_mean = mean(incongruencyCostsRT_g);

overallPerformanceER_g_mean = mean(overallPerformanceER_g);
switchCostsER_g_mean = mean(switchCostsER_g);
incongruencyCostsER_g_mean = mean(incongruencyCostsER_g);

overallPerformanceRT_g_sem = std(overallPerformanceRT_g)/sqrt(length(numSequences));
switchCostsRT_g_sem = std(switchCostsRT_g)/sqrt(length(numSequences));
incongruencyCostsRT_g_sem = std(incongruencyCostsRT_g)/sqrt(length(numSequences));

overallPerformanceER_g_sem = std(overallPerformanceER_g)/sqrt(length(numSequences));
switchCostsER_g_sem = std(switchCostsER_g)/sqrt(length(numSequences));
incongruencyCostsER_g_sem = std(incongruencyCostsER_g)/sqrt(length(numSequences));

CogSci_Simulation1_Stats;

%% PLOTS

plotScatter = 0;
%
colors = [51 153 255; 255 153 51; 200 200 200]/255;
fontsize = 15;
markerSize = 10;

xdata = gains;

fig1 = figure(1);
set(fig1, 'Position', [100 100 540 170]);

% Switch Cost RT
subplot(1,2,1);
errorbar(xdata, switchCostsRT_g_mean, switchCostsRT_g_sem, 'k', 'LineWidth', 3); hold on;
if(plotScatter)
    scatter(gain_log(:),  switchCostsRT_g(:), markerSize, 'k');
end
hold off;
xlim([min(xdata) max(xdata)]);
ylabel({'', 'Switch Cost'}, 'FontSize', fontsize, 'FontWeight','bold');
xlabel('Gain', 'FontSize', fontsize, 'FontWeight','bold');
set(gca, 'FontSize', fontsize, 'FontWeight','bold');

% Switch Cost ER
subplot(1,2,2);
errorbar(xdata, switchCostsER_g_mean, switchCostsER_g_sem, 'k', 'LineWidth', 3); hold on;
if(plotScatter)
    scatter(gain_log(:),  switchCostsER_g(:), markerSize, 'k');
end
hold off;
xlim([min(xdata) max(xdata)]);
ylabel({'', 'Switch Cost'}, 'FontSize', fontsize, 'FontWeight','bold');
xlabel('Gain', 'FontSize', fontsize, 'FontWeight','bold');
set(gca, 'FontSize', fontsize, 'FontWeight','bold');

fig2 = figure(2);
set(fig2, 'Position', [100 300 540 140]);

% Incongruency Cost RT
subplot(1,2,1);
errorbar(xdata, incongruencyCostsRT_g_mean, incongruencyCostsRT_g_sem, 'k', 'LineWidth', 3); hold on;
if(plotScatter)
    scatter(gain_log(:),  incongruencyCostsRT_g(:), markerSize, 'k');
end
hold off;
xlim([min(xdata) max(xdata)]);
ylabel({'Incongruency', 'Cost'}, 'FontSize', fontsize, 'FontWeight','bold');
% xlabel('Gain', 'FontSize', fontsize, 'FontWeight','bold');
set(gca, 'XTickLabels', '');
set(gca, 'FontSize', fontsize, 'FontWeight','bold');

% Incongruency Cost ER
subplot(1,2,2);
errorbar(xdata, incongruencyCostsER_g_mean, incongruencyCostsER_g_sem, 'k', 'LineWidth', 3); hold on;
if(plotScatter)
    scatter(gain_log(:),  incongruencyCostsER_g(:), markerSize, 'k');
end
hold off;
xlim([min(xdata) max(xdata)]);
ylabel({'Incongruency', 'Cost'}, 'FontSize', fontsize, 'FontWeight','bold');
% xlabel('Gain', 'FontSize', fontsize, 'FontWeight','bold');
set(gca, 'XTickLabels', '');
set(gca, 'FontSize', fontsize, 'FontWeight','bold');

fig3 = figure(3);
set(fig3, 'Position', [100 500 540 140]);

% overall RT
subplot(1,2,1);
errorbar(xdata, overallPerformanceRT_g_mean, overallPerformanceRT_g_sem, 'k', 'LineWidth', 3); hold on;
if(plotScatter)
    scatter(gain_log(:),  overallPerformanceRT_g(:), markerSize, 'k');
end
hold off;
xlim([min(xdata) max(xdata)]);
ylabel({'Mean', 'Performance'}, 'FontSize', fontsize, 'FontWeight','bold');
% xlabel('Gain', 'FontSize', fontsize, 'FontWeight','bold');
set(gca, 'XTickLabels', '');
set(gca, 'FontSize', fontsize, 'FontWeight','bold');

% overall ER
subplot(1,2,2);
errorbar(xdata, overallPerformanceER_g_mean, overallPerformanceER_g_sem, 'k', 'LineWidth', 3); hold on;
if(plotScatter)
    scatter(gain_log(:),  overallPerformanceER_g(:), markerSize, 'k');
end
hold off;
xlim([min(xdata) max(xdata)]);
ylabel({'Mean', 'Performance'}, 'FontSize', fontsize,'FontWeight','bold');
% xlabel('Gain', 'FontSize', fontsize, 'FontWeight','bold');
set(gca, 'XTickLabels', '');
set(gca, 'FontSize', fontsize, 'FontWeight','bold');


%% plot model dynamics for different gains
% task environment
numTrialsPerMiniBlock = 10;
[params, sequenceFile] = generateToySequence(numTrialsPerMiniBlock, numMiniBlocks, frequency, congruentTrials);

gains_tested = [1 3];

% model parameters
alpha = 0;
mag_noise = 0;
gain = gains_tested(1);
plot_run = 0;
% set connection weights
W_input         = [1 1];
W_selfexcite    = [1 1];
W_inhib         = [-1.0 -1.0];

% create net
gainNet = gNet(gain, tau, alpha, mag_noise, plot_run);
gainNet.configure(W_input, W_selfexcite, W_inhib);


% Get data file
dataFile = load(sequenceFile);
% pull sequence with corresponding switch frequency
blockIndex = 2;
inputSeq = dataFile.params.CTS.Sequence{blockIndex}.Task;
%Convert inputsequence into one suitable for Gnet
inputSeq_block2 = zeros(2, length(inputSeq));
miniBlockTrial = dataFile.params.CTS.Sequence{blockIndex}.miniBlockTrial;
for i = 1:length(inputSeq_block2)
   if inputSeq(i) == 1
       inputSeq_block2(1,i) = 1;
   else
       inputSeq_block2(2,i) = 1;
   end
end
inputSequence = inputSeq_block2;

% create input sequence
nTrials = size(inputSequence,2);
timeStepsPerTrial = 1;
responseStart = 1;
responseStop = 1;
[inputSeq respSeq] = gNet.buildSwitchSequence(nTrials, timeStepsPerTrial, responseStart, responseStop);
gainNet.setSequence(inputSequence, respSeq);

% run net
act = gainNet.runSequence();
activationTrajectory = act(:, (numTrialsPerMiniBlock):(numTrialsPerMiniBlock*2));

gainNet.plotNet_CogSci([0 1], activationTrajectory);