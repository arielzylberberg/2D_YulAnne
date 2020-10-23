classdef SigmaSq < Fit.D2.Common.SigmaSq
% Fit.D2.IrrIxn.SigmaSq
% 
% 2016 YK wrote the initial version.
% 
% Note:
% It might be hard to determine if SigmaSq is based on
% the drift after bias or before bias.
%
% SigmaSqConst just returns a scalar 1.
% SigmaSqLinearPreDrift uses the original cond,
% SigmaSqLinearPostDrift uses the drift value.
    
properties (SetAccess = protected)
    Drift
end
methods
    function v = get_conds_rel(W)
        % Returns the original drift vector
        assert(~isempty(W.Drift), 'Use W.set_Drift() to assign Drift!');
        [~, v] = W.Drift.get_drift_vec;
    end
    function set_Drift(W, obj)
        W.Drift = obj;
    end
end
end