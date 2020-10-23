function [meanRT, meanProbabilityLeftResponse, drift] = runDDM(colorCoherence, motionCoherence, currentColor, currentMotion, ddmp, controlSignalSpace)

% controlSignalSpace = combvec(0.1:0.1:1, 0.1:0.1:1);
% motionCoherence = 0.2;
% colorCoherence = 0.2;
% currentColor = 1;
% currentMotion = 1;
% ddmp.z = 0.45;
% ddmp.bias = 0.5;
% ddmp.T0 = 0.2;
% ddmp.c = 0.5;

% determine decision threshold
if(size(controlSignalSpace,1) >= 3)
        ddmp.z = controlSignalSpace(3,:);
end

% compute drift rate:
automaticComponentColor = colorCoherence * currentColor;
automaticComponentMotion = motionCoherence * currentMotion;
automaticComponent = automaticComponentColor + automaticComponentMotion;

bias = ddmp.bias;
drift = automaticComponentColor * controlSignalSpace(1,:) + automaticComponentMotion *controlSignalSpace(2,:) + automaticComponent;

[meanProbabilityLeftResponse,meanRT] = DDM.ddmSimFRG(drift,bias,ddmp,1);

% subplot(2,1,1);
% plot(1-unique(meanProbabilityLeftResponse));
% subplot(2,1,2);
% plot(unique(meanRT));



