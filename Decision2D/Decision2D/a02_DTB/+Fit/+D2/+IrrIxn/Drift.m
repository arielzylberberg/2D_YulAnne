classdef Drift < Fit.D2.Common.Drift
    % Fit.D2.IrrIxn.Drift
    %
    % 2016 YK wrote the initial version.
methods
    function W = Drift(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init_params0(W)
        W.init_params0@Fit.D2.Common.Drift;        
        W.add_params({
            {'k_irr', 0, -10, 10}
            {'k_abs_irr', 0, -10, 10}
            });
    end
    function [drift_vec, cond_rel, cond_irr] = get_drift_vec(W)
        if W.get_dim_rel_W == 1
            [cond_rel, cond_irr] = ...
                ndgrid(W.get_conds_rel, W.get_conds_irr);
        else
            [cond_irr, cond_rel] = ...
                ndgrid(W.get_conds_irr, W.get_conds_rel);
        end
        cond_rel = cond_rel(:);
        cond_irr = cond_irr(:);
        
        drift_vec = W.cond2drift(cond_rel, cond_irr);
    end
    function drift = cond2drift(W, cond_rel, cond_irr)
        % drift = cond2drift(W, cond_rel, cond_irr)
        
        k = W.get_th_('k');
        bias = W.get_th_('bias');
        k_irr = W.get_th_('k_irr');
        k_abs_irr = W.get_th_('k_abs_irr');
        
        drift = k .* (cond_rel - bias) ...
                + k_irr .* cond_irr ...
                + k_abs_irr .* abs(cond_irr);
    end
    function conds_irr = get_conds_irr(W)
        conds = W.Data.get_conds;
        dim_irr_W = W.get_dim_irr_W;
        if isempty(dim_irr_W)
            conds_irr = [];
        else
            conds_irr = conds{dim_irr_W};
        end        
    end
    function b = get_cond_bias(W)
        % b(cond_irr,1) : the subjective point of equality
        if isfield(W.th, 'bias')
            b = W.th.bias;
        else
            b = nan;
        end
    end
end
%% Plot
methods
    function plot(W)
        [y, x, sep] = W.get_drift_vec;
        
        subplot(2,1,1);
        plotsep(x, y, @hsv2, ...
            {'Marker', 'o', 'LineStyle', 'none'}, ...
            'sep', sep);
        
        subplot(2,1,2);
        plotsep(sep, y, @hsv2, ...
            {'Marker', 'o', 'LineStyle', 'none'}, ...
            'sep', x);
    end
end
end