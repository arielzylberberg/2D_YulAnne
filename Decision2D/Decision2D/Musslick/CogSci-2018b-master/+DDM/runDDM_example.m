%% run DDM

drift = 0.08927999999999997;    % drift
ddmp.z = 0.2645;			% threshold
bias = 0.5;                     % starting point
ddmp.c = 0.5;				% noise
ddmp.T0 = 0.15;         % T0

drift = [0.08 0.07 0.07];
ddmp.z = [0.2645 0.2645 0.3845];

[meanERs,meanRTs,meanDTs,condRTs,condVarRTs, condSkewRTs] = DDM.ddmSimFRG(drift,bias,ddmp,1);
meanERs
