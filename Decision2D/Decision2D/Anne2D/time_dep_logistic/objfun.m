function [dev B stats yPred] = objfun(theta, D,S)
Q = opt_unpack(theta,S);

v2struct(D) %data
v2struct(Q) % parameters

skew=normcdf(alpha1*(dur-mu)/sigma1); %for skew normal

if 1
    X = [s_this_coh, ...   %overall baseline sensitivity
        (dur.^gamma1).*s_this_coh, ...   %time-dependent  sensitivity
        skew.*normpdf(dur,mu,sigma1).* s_this_coh.* I_low_other...   %time-dependent change in sensitivity for other stimulus low coherence
        dur ...  %duration bias
        I_low_other, ... % other stimulus low coherence bias
        dur.* I_low_other];  % time dependent other stimulus low coherence bias
    
%     for k=2:max(session);%     
%         X=[X (session==k).*(dur.^gamma1).*s_this_coh];
%     end    

else
    X = [s_this_coh, ...
        (dur.^gamma1).*s_this_coh, ...
        skew.*normpdf(dur,mu,sigma1).* s_this_coh.* I_low_other...
        ];
end

[B, dev, stats] = glmfit(X,this_choice,'binomial','logit');

yPred = glmval(B,X,'logit');

%theta'
%dev


