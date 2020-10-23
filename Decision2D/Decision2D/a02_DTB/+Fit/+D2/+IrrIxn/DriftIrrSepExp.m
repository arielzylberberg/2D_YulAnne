classdef DriftIrrSepExp < Fit.D2.IrrIxn.DriftIrrSep
    % Fit.d2.IrrIxn.DriftIrrSep
    
    % 2017 YK wrote the initial version.
    
methods
    function W = DriftIrrSepExp(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init_params0(W)
        W.init_params0@Fit.D2.IrrIxn.DriftIrrSep;
        
        W.add_params({
            {'log10_t_st', log10(0.12), log10(0.01), log10(0.5)}
            {'log10_t_half', log10(0.1), log10(0.01), log10(0.5)}
            });
    end
    function drift_cond_t = get_drift_cond_t(W)
        %%
        [drift_vec, cond_rel, cond_irr] = W.get_drift_vec;
        bias_vec = W.cond2drift(zeros(size(cond_rel)), cond_irr);
        
        %%
        t = W.t;        
        t_st = 10.^W.th.log10_t_st;
        t_half = 10.^W.th.log10_t_half;
        
        wt = 2.^(-(t - t_st)/t_half);
        wt(t < t_st) = 1;
        
        %%
        drift_cond_t = ...
            bsxfun(@times, drift_vec, wt) ...
            + bsxfun(@times, bias_vec, 1 - wt);
    end
end
end