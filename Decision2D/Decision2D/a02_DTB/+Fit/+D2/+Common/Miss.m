classdef Miss < Fit.Common.Miss & Fit.D2.Common.CommonWorkspace
    % Fit.D2.Common.Miss
    %
    % Estimates a miss from both dimensions of data.
    %
    % 2015 YK wrote the initial version.
    
%% Customize for data
methods
    function customize_th_for_Data(W)
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
        
        easiest_cond = max(abs(cond));
        tf_easiest_cond = all(bsxfun(@eq, abs(cond), easiest_cond), 2);
        
        accu = all(W.Data.get_accu, 2);
        accu = accu(tf_easiest_cond);
        
        n_accu = sum(accu);
        n_tot = length(accu);
    end
end
%% Data interface
methods
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        obj_or_name = W.enforce_class('Fit.D2.Common.DataChRtPdf', ...
            obj_or_name);
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end    
end
%% Test
methods (Static)
    function [tf, W] = test_W
        %%
        W = Fit.D2.Common.Miss;
        W.set_Data;
        W.Data.set_path;
        W.Data.load_data;
        
        W.set_miss_ci_alpha(1e-3);
        W.customize_th_for_Data;
        
        miss_vec = [W.lb.miss, W.th0.miss, W.th.miss, W.ub.miss];
        disp(miss_vec);
        
        %%
        tf = isequal_within( ...
            miss_vec, ...
            [0 0 0 0.0303], ...
            1e-4);
        
        assert(tf);
    end
end
end