function act = simulate_gNet_fun(inputSequence, gain, tau, varargin)

% set net parameters
%gain = 3; %example gain
%tau = 0.5;
alpha = 0.8;
mag_noise = 0;
plot_run = 0;

% set connection weights
W_input         = [1 1];
W_selfexcite    = [1 1];
W_inhib         = [-1.0 -1.0];


if(~isempty(varargin))
    W_input = [varargin{1} varargin{1}];
    
    % add pre-cue trials and log position
    newInputSequence = [];
    newInputSequence_cues = [];
    
    if(length(varargin) >= 2)
        miniBlockTrial = varargin{2};
        
        for trial = 1:size(inputSequence,2)
    
            if(miniBlockTrial(trial) == 1)
                
                newInputSequence = [newInputSequence inputSequence(:, trial)];
                newInputSequence_cues = [newInputSequence_cues size(newInputSequence,2)];
            end
            
            newInputSequence = [newInputSequence inputSequence(:, trial)];

        end
         
        inputSequence = newInputSequence;
        
    end
end

% create input sequence
nTrials = size(inputSequence,2);
timeStepsPerTrial = 1;
responseStart = 1;
responseStop = 1;
[inputSeq respSeq] = gNet.buildSwitchSequence(nTrials, timeStepsPerTrial, responseStart, responseStop);

% create net
gainNet = gNet(gain, tau, alpha, mag_noise, plot_run);
gainNet.configure(W_input, W_selfexcite, W_inhib);
gainNet.setSequence(inputSequence, respSeq);

% run net
gainNet.runSequence();

act = gainNet.act_log;
if(length(varargin) >= 2)
    act(:, newInputSequence_cues) = [];
end

%% plot MSE profile
%{
gains = [3 2.5];
timeSteps = 20;
%gainNet.plotDynamics(gains, timeSteps);
gainNet.plotMSEprofile(gains, timeSteps);

%% train net

gainNet.trainSequence();
figure(1);
plot(sum(gainNet.MSE_respLog,1)); % gain_respLog
figure(2);
plot(gainNet.gain_respLog);
%% training plots: dependence on response deadline

timeStepsPerTrial = 20;
nTrials = 500;
gain_init = 2;
deadlines = 2:14;
gainNet.plotGainRespDeadline(deadlines, timeStepsPerTrial, nTrials, gain_init);

%% training plots: dependence on response switch frequency

timeStepsPerTrial = 20;
responseStart = 1;
responseStop = 4;
nTrials = 400;
gain_init = 2;
frequencies = 0.1:0.1:1;
gainNet.plotGainFlexibility(frequencies, timeStepsPerTrial, responseStop, nTrials,gain_init);

%% plot error landscape

gains = 0:0.1:10;
timeStepsPerTrial = 20;
responseDeadline = 7;
gainNet.plotErrorLandscape(responseDeadline, timeStepsPerTrial, gains);

%}
