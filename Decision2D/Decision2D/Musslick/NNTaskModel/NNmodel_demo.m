
% set up network parameters
nHidden = 10;              % number of hidden units
learningRate = 0.3;         % learning rate
bias = -2;                      % weight from bias units to hidden & output units
init_scale = 0.5;              % max. magnitude of initial weights
thresh = 0.0001;            % mean-squared error stopping criterion
decay = 0.0000;             % weight penalization parameter
hiddenPathSize = 1;        % group size of hidden units that receive the same weights from the task layer)
outputPathSize = 1;         % group size of output units that receive the same weights from the task layer)
trainingIterations = 2000;   % training iterations

% generate training data
inputPatterns = repmat([0 0; 0 1; 1 0; 1 1], 3, 1);
taskPatterns = [repmat([1 0 0],4,1); ...
                       repmat([0 1 0],4,1);
                       repmat([0 0 1],4,1);];
outputPatterns = nan(size(inputPatterns,1),1);

% AND task
outputPatterns(1:4) = inputPatterns(1:4, 1) & inputPatterns(1:4, 2); 
outputPatterns(5:8) = inputPatterns(5:8, 1) | inputPatterns(1:4, 2); 
outputPatterns(9:12) = xor(inputPatterns(9:12, 1), inputPatterns(1:4, 2));


% set up model
taskNet = NNmodel(nHidden, learningRate, bias, init_scale, thresh, decay, hiddenPathSize, outputPathSize);

% set training data
taskNet.setData(inputPatterns, taskPatterns, outputPatterns);

% initialize network
taskNet.configure(); 

% train network
taskNet.trainOnline(trainingIterations);

% plot learning curve
plot(taskNet.MSE_log);
