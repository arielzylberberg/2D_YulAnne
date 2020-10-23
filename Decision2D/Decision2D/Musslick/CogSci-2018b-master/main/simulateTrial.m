function [meanRT, meanProbabilityLeftResponse, allRTs, allPLowerResp] = simulateTrial(numberOfSimulations, colorCoherence, motionCoherence, currentColor, currentMotion, ddmpColor, ddmpMotion, controlIntensityColor, controlIntensityMotion, T0, decisionThreshold, dt, varargin)

% example values for parameters

% numberOfSimulations = 10000;    % number of evidence accumulation sims
% colorCoherence = 0.1;                   % drift for color evidence accumulation process (see gainFlexSim_Exp2_slurm.py)
% motionCoherence = 0.1;                  % same for motion
% currentColor = 1;                       % 1 if color indictaes left response, -1 if it's right
% currentMotion = 1;                       % 1 if motion indictaes left response, -1 if it's right
% ddmpColor.bias = 0;         % bias (starting point) of evidence accumulation process for color
% ddmpMotion.bias = 0;       % bias (starting point) of evidence accumulation process for color
% ddmpColor.noise = 0.01;% noise of accumulation process for color
% ddmpMotion.noise = 0.01;% noise of accumulation process for motion
% controlIntensityColor = 0.5;  % activity of color goal unit
% controlIntensityMotion = 0.5;  % activity of color goal unit
% T0 = 0.2; % non-decision time (see gainFlexSim_Exp2_slurm.py)
% decisionThreshold = 0.025; % threshold for response accumulation process



showWarning = 0;

if(~isempty(varargin))
    plotTrial = varargin{1};
    
    if(length(varargin) >= 2) 
        showWarning = varargin{2};
    end
    
else
    plotTrial = 0;
end

allRTs = nan(1, numberOfSimulations);
allPLowerResp = nan(1, numberOfSimulations);

% determine the drift of color evidence accumulation based on the color identiy (currentColor) and the coherence
ddmpColor.drift = currentColor * getDriftFromCoherence(colorCoherence, 'color');

% determine the drift of motion evidence accumulation based on the motion identiy (currentMotion) and the coherence
ddmpMotion.drift = currentMotion * getDriftFromCoherence(motionCoherence, 'motion');

% for each simulation run an evidence accumulation process
for sim = 1:numberOfSimulations;
   
    [allRTs(sim), allPLowerResp(sim)] = simulateResponse(T0, decisionThreshold, dt, ddmpColor, ddmpMotion, controlIntensityColor, controlIntensityMotion, showWarning);
    
end

meanRT = nanmean(allRTs);
meanProbabilityLeftResponse = nanmean(allPLowerResp);

%% PLOT

if(plotTrial) 
    
%     T0 = 0.2;
%     decisionThreshold = 0.025;
%     
%     ddmpColor.drift  = 0.1;
%     ddmpColor.bias = 0;
%     ddmpColor.dt = 0.001;
%     ddmpColor.noise = 0.02;
%     
%     ddmpMotion.drift  = -0.1;
%     ddmpMotion.bias = ddmpColor.bias;
%     ddmpMotion.dt = ddmpColor.dt;
%     ddmpMotion.noise = ddmpColor.noise;
%     
%     controlIntensityColor = 0.7978;
%     controlIntensityMotion = 0.40776;
    
    [~, ~, decisionVariableTrace, colorEvidenceTrace, motionEvidenceTrace] = simulateResponse(T0, decisionThreshold, dt, ddmpColor, ddmpMotion, controlIntensityColor, controlIntensityMotion, 0);
    figure(1);
    plot(decisionVariableTrace, 'k', 'LineWidth', 3); hold on;
    plot(colorEvidenceTrace, 'b', 'LineWidth', 3); 
    plot(motionEvidenceTrace, 'g', 'LineWidth', 3); 
    plot(repmat(decisionThreshold, 1, length(decisionVariableTrace)), '--k', 'LineWidth', 2);
    plot(repmat(-decisionThreshold, 1, length(decisionVariableTrace)), '--k', 'LineWidth', 2); hold off;
    legend('Decision Variable', 'Color Evidence', 'Motion Evidence');
    xlabel('Time', 'FontSize', 12);
    ylabel('Value', 'FontSize', 12);
    
end

end