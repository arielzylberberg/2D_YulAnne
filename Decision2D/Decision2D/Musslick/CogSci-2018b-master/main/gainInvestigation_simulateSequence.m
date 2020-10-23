function [optimizationCriterion, performance, allRR, allmeanRT, allmeanAccuracy, allActivation, allDrift] = gainInvestigation_simulateSequence(p, ddmpParams, sequenceFile, tau, inputWeight)

gain = p(1);

% Get data file
dataFile = load(sequenceFile);

% pull sequence with corresponding switch frequency
blockIndex = 2;

inputSeq = dataFile.params.CTS.Sequence{blockIndex}.Task;

%Convert inputsequence into one suitable for Gnet
% inputSeq_block1 = zeros(2, length(inputSeq));
inputSeq_block2 = zeros(2, length(inputSeq));
miniBlockTrial = dataFile.params.CTS.Sequence{blockIndex}.miniBlockTrial;

for i = 1:length(inputSeq_block2)
   if inputSeq(i) == 1
       inputSeq_block2(1,i) = 1;
   else
       inputSeq_block2(2,i) = 1;
   end
end
 
%Run sequence in gNet and get activation, just block 2 for now
% gain = gain; %Comment out when done
act = simulate_gNet_fun(inputSeq_block2, gain, tau, inputWeight, miniBlockTrial);

allActivation = act();
colorControlSignalActivity = act(1,:);
motionControlSignalActivity = act(2,:);
%%
% Set constants for decision module, comment all out when done
% numberOfSimulations = 100;    % number of evidence accumulation sims
colorCoherence = .1;           % drift for color evidence accumulation process (see gainFlexSim_Exp2_slurm.py)
motionCoherence = .1;          % same for motion
currentColor = 1;               % 1 if color indictaes left response, -1 if it's right
currentMotion = 1;              % 1 if motion indictaes left response, -1 if it's right
%ddmpColor.bias = 0;             % bias (starting point) of evidence accumulation process for color
%ddmpMotion.bias = 0;            % bias (starting point) of evidence accumulation process for color
%ddmpColor.noise = 0.05;         % noise of accumulation process for color
%ddmpMotion.noise = 0.05;        % noise of accumulation process for motion
%ddmpColor.dt = 0.001;           % dt for Color
%ddmpMotion.dt = 0.001;          % dt for Motion
controlIntensityColor = 0.5;    % activity of color goal unit
controlIntensityMotion = 0.5;   % activity of color goal unit 

ddmp.bias = ddmpParams.bias;                % DDM bias (centered at 0.5)
ddmp.c = ddmpParams.c;                   % DDM noise
ddmp.T0 = ddmpParams.T0;                  % DDM non-decision time
if(length(p) >= 2)
    ddmp.z = p(2);
else
    ddmp.z = ddmpParams.z;                  % Decision threshold
end

% Convert params into comparable numbers (-1 and 1)
currColor = dataFile.params.CTS.Sequence{blockIndex}.curColor(:); % 2 for second block
currColor(currColor == 1) = -1;
currColor(currColor == 2) = 1;

currMotion = dataFile.params.CTS.Sequence{blockIndex}.curMotion(:);
currMotion(currMotion == 1) = -1;
currMotion(currMotion == 2) = 1;

correctResponse = dataFile.params.CTS.Sequence{blockIndex}.correctResponse(:);
correctResponse(correctResponse == 1) = -1;
correctResponse(correctResponse == 2) = 1;

% simulate Trial by Trial
RR = zeros(1,length(inputSeq));
allmeanRT = zeros(1,length(inputSeq));
allmeanProbLeftResp = zeros(1,length(inputSeq));
allmeanAccuracy = zeros(1,length(inputSeq));
allDrift = zeros(1,length(inputSeq));

for i = 1:length(inputSeq)
    % Set variables per trial
    controlIntensityColor = colorControlSignalActivity(i);
    controlIntensityMotion = motionControlSignalActivity(i);
    currentColor = currColor(i);
    currentMotion = currMotion(i);
    
    controlSignalSpace = [controlIntensityColor; ... % control signal for color
                               controlIntensityMotion];    % control signal for motion
    
    % Using defined params, simulate trial to get meanRT and
    % meanProbLeftResponse
    [meanRT, meanProbabilityLeftResponse, drift] = runDDM(colorCoherence, motionCoherence, currentColor, currentMotion, ddmp, controlSignalSpace);
    
    allDrift(i) = drift;
    
    % Probability of left means -1 (drift down)
    if (correctResponse(i) == -1)
        accuracy = meanProbabilityLeftResponse;
    else
        accuracy = 1 - meanProbabilityLeftResponse;
    end
    allmeanAccuracy(i) = accuracy;
    
    % Save variables, use RR to get reward rate per trial
    allmeanRT(i) = meanRT;
    allmeanProbLeftResp(i) = meanProbabilityLeftResponse;
    RR(i) = accuracy; %/meanRT;

end

allRR = RR;
performance = mean(RR);
optimizationCriterion = -performance;
