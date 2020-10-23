function [RT, probabilityLowerResponse, decisionVariableTrace, colorEvidenceTrace, motionEvidenceTrace] = simulateResponse(T0, decisionThreshold, dt, ddmpColor, ddmpMotion, controlIntensityColor, controlIntensityMotion, varargin) 

     % maximum RT (to stop integration processes that do not converge
     maxRT = 10;

     if(~isempty(varargin)) 
         showWarning = varargin{1};
     else
         showWarning = 1;
     end
     
     % initialize accumulation process for response
     RT = 0;                                                                               % reaction time
     probabilityLowerResponse = 0;                                          % if error then ER = 1
     
     % initialize accumulation process for color
     evidenceColor= ddmpColor.bias;                         % starting point for color accumulation process
     colorEvidenceTrace = evidenceColor;                   % trajectory of evidence towards either red or green color
     
     % initialize accumulation process for motion
     evidenceMotion = ddmpMotion.bias;                    %starting point for motion accumulation process
     motionEvidenceTrace = evidenceMotion;              % trajectory of evidence towards either up or down

     % while decision variable for response is below threshold 
     LLR = 0;   % initially there is no bias towards either left or right reesponse
     decisionVariableTrace = LLR;                                   % trajectory of evidence towards either left or right response
     
     if((controlIntensityColor == controlIntensityMotion) & ddmpColor.drift == -ddmpMotion.drift)
         RT = inf;
         probabilityLowerResponse = 0.5;
         return;
     end
     
     while (abs(LLR) < decisionThreshold)
         
         % compute evidence for color and motion
         evidenceColor =  accumulateEvidence(evidenceColor, ddmpColor.drift, ddmpColor.dt, ddmpColor.noise);
         evidenceMotion =  accumulateEvidence(evidenceMotion, ddmpMotion.drift, ddmpMotion.dt, ddmpMotion.noise);
         
         % compute log likelihood ratio for response (e.g. left/right)
         LLR =  log( ( controlIntensityColor * exp(evidenceColor) * (1 + exp(evidenceMotion)) + ...
                     controlIntensityMotion * exp(evidenceMotion) * (1 + exp(evidenceColor)) ) / ...
                     ( controlIntensityColor * (1 + exp(evidenceMotion)) + controlIntensityMotion * (1 + exp(evidenceColor))));
                     
        % log evidence
        colorEvidenceTrace = [colorEvidenceTrace evidenceColor];
        motionEvidenceTrace = [motionEvidenceTrace evidenceMotion];
        decisionVariableTrace = [decisionVariableTrace LLR];
                 
        RT = RT + dt;
        
        if(RT > maxRT)
            RT = inf;
            if(showWarning)
                warning(['Maxium RT reached (' num2str(maxRT) '. Stopping response integration process']);
            end
            break
        end
            
        
     end
     
     if(sign(LLR) < 0)
         probabilityLowerResponse = 1;
     else
         probabilityLowerResponse = 0;
     end

     RT= RT + T0;
 end