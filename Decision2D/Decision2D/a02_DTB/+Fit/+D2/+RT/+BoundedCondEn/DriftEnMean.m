classdef DriftEnMean < Fit.D2.RT.BoundedCondEn.DriftEn
    % Fit.D2.RT.BoundedCondEn.DriftEnMean
    %
    % Gives coh repeated n_tr instead of CondEn.
    %
    % 2015 YK wrote the initial version.
methods
    function W = DriftEnMean(varargin)
        W = W@Fit.D2.RT.BoundedCondEn.DriftEn(varargin{:});
    end
    function v = get_cond_t(W)
        % Enforce nCond x nt form
        cond = W.Data.get_cond;
        cond = cond(:, W.get_dim_rel_W);
        
        nt = W.get_nt;
        
        v = repmat(cond, [1, nt]);
    end
end
end