classdef SigmaSqLinearPostDrift ...
        < Fit.D2.IrrIxn.SigmaSq ...
        & Fit.D2.Common.SigmaSqLinear
methods 
    function conds_rel = get_conds_rel(W)
        % Returns the original drift vector
        assert(~isempty(W.Drift), 'Use W.set_Drift() to assign Drift!');
        [drift_rel, conds_rel0] = W.Drift.get_drift_vec;
        max_drift_rel = max(abs(drift_rel(:)));
        max_conds_rel = max(abs(conds_rel0(:)));
        
        conds_rel = drift_rel ./ max_drift_rel .* max_conds_rel;
    end
end
end