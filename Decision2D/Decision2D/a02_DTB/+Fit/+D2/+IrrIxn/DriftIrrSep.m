classdef DriftIrrSep < Fit.D2.IrrIxn.DriftConst
    % Fit.D2.IrrIxn.DriftIrrSep
    %
    % 2016 YK wrote the initial version.
    
%% Settings
properties
    to_sep_by_abs = false;
    to_fix_bias_irr = true;
end
properties (Dependent)
    bias_irr
    dCond_irr
    conds_irr
    n_conds_irr
end
%% Methods 
methods
    function W = DriftIrrSep(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        W.init@Fit.D2.IrrIxn.DriftConst(varargin{:});
        W.init_params0;
    end
    function init_params0(W)
        W.remove_params_all;
        W.init_params0@Fit.D2.Common.Drift;
        n_irr = W.n_conds_irr;
        
        for ii = 1:n_irr
            incl = W.dCond_irr == ii;
            cond = W.Data.cond(incl, W.dim_rel_W);
            ch = W.Data.ch(incl, W.dim_rel_W) == 2;
            b = glmfit(cond, ch, 'binomial');
            bias0 = -b(1) / b(2);
            
            name = str_con('bias_irr', ii);
            W.add_params({
                {name, bias0, -1, 1}
                });
            W.th0.k = b(2); 
            W.th.k = b(2);
            
            if W.to_fix_bias_irr
                W.fix_to_th_(name);
            end
        end
    end
    function v = get.bias_irr(W)
        n_irr = W.n_conds_irr;
        v = zeros(n_irr, 1);
        for ii = 1:n_irr
            v(ii) = W.th.(str_con('bias_irr', ii));
        end
    end
    function v = get.n_conds_irr(W)
        v = numel(unique(W.conds_irr));
    end
    function v = get.conds_irr(W)
        if W.to_sep_by_abs
            v = unique(W.Data.aConds{W.dim_irr_W});
        else
            v = unique(W.Data.conds{W.dim_irr_W});
        end
    end
    function v = get.dCond_irr(W)
        if W.to_sep_by_abs
            v = W.Data.adCond(:, W.dim_irr_W);
        else
            v = W.Data.dCond(:, W.dim_irr_W);
        end
    end
    function drift = cond2drift(W, cond_rel, cond_irr)
        k = W.get_th_('k');
        bias = W.bias_irr;
        
        if W.to_sep_by_abs
            dCond_irr = bsxFind(abs(cond_irr), W.conds_irr);
        else
            dCond_irr = bsxFind(cond_irr, W.conds_irr);
        end
        
        drift = k .* (cond_rel - vVec(bias(dCond_irr)));
    end
    function b = get_cond_bias(W)
        b = W.bias_irr(:);
    end
end
%% Plotting
methods
    function h = plot_drift_by_cond(W)
        [drift_vec, cond_rel, cond_irr] = W.get_drift_vec;
        [~,~,d_cond_rel] = unique(cond_rel);
        [~,~,d_cond_irr] = unique(cond_irr);
        
        drift_mat = accumarray([d_cond_rel, d_cond_irr], drift_vec);
        cond_mat = accumarray([d_cond_rel, d_cond_irr], cond_rel);
        
        h = plot(cond_mat, drift_mat, '.-');
        crossLine('v', 0, 'k--');
        crossLine('h', 0, 'k--');
        xlabel('cond');
        ylabel('drift');
    end
    function h = plot_drift_by_cond_irr(W)
        [drift_vec, cond_rel, cond_irr] = W.get_drift_vec;
        [~,~,d_cond_rel] = unique(cond_rel);
        [~,~,d_cond_irr] = unique(cond_irr);
        
        drift_mat = accumarray([d_cond_irr, d_cond_rel], drift_vec);
        cond_mat = accumarray([d_cond_irr, d_cond_rel], cond_irr);
        
        h = plot(cond_mat, drift_mat, '.-');
        crossLine('v', 0, 'k--');
        crossLine('h', 0, 'k--');
        
        xlabel(sprintf('cond_%s', Data.Consts.dimNames{W.dim_irr_W}));
        ylabel(sprintf('drift_%s', Data.Consts.dimNames{W.dim_rel_W}));
    end
end
end