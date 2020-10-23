classdef MissLocalDimRel < Fit.D2.Common.Miss & Fit.D2.Common.CommonWorkspace
    % Fit.D2.Common.MissLocalDimRel
    %
    % Estimates the range of plausible miss from data.
    %
    % 2015 YK wrote the initial version.
methods
    function customize_th_for_Data(W, dim_rel_W)
        W.set_dim_rel_W(dim_rel_W);
        W.calc_miss;
    end
    function calc_miss(W)
        [miss, lb, ub] = W.get_miss;
        W.th.miss = miss;
        W.th0.miss = miss;
        W.lb.miss = lb;
        W.ub.miss = ub;
    end
    function [miss, lb, ub] = get_miss(W)
        [n_accu, n_tot] = W.get_easiest_cond_accu;
        [accu, ci] = binofit(n_accu, n_tot, W.miss_ci_alpha);
        miss = 1 - accu;
        lb = 1 - ci(2);
        ub = 1 - ci(1);
    end
end
methods (Hidden)
    function [n_accu, n_tot] = get_easiest_cond_accu(W)
        cond = W.Data.get_cond;
        cond = cond(:, W.get_dim_rel_W);
        
        accu = W.Data.get_accu;
        accu = accu(:, W.get_dim_rel_W);
        
        easiest_cond = max(abs(cond));
        tf_easiest_cond = abs(cond) == easiest_cond;
        
        accu = accu(tf_easiest_cond);
        
        n_accu = sum(accu);
        n_tot = length(accu);
    end
end
methods
    function set_miss_ci_alpha(W, v)
        assert(isnumeric(v));
        assert(isscalar(v));
        assert((v >= 0) && (v <= 1));
        W.miss_ci_alpha = v;
    end
end
%% Test
methods (Static)
    function [tf, W] = test_W
        %%
        W = Fit.D2.Common.MissLocalDimRel;
        W.set_Data;
        W.Data.set_path;
        W.Data.load_data;
        
        W.set_miss_ci_alpha(1e-3);
        W.customize_th_for_Data(2);
        
        miss_vec = [W.lb.miss, W.th0.miss, W.th.miss, W.ub.miss];
        disp(miss_vec);
        
        tf = isequal_within( ...
            miss_vec, ...
            [0.0006, 0.0045, 0.0045, 0.0156], ...
            1e-4);
        
        assert(tf);
    end
end
end