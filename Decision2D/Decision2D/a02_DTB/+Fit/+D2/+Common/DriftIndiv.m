classdef DriftIndiv < Fit.D2.Common.Drift & Fit.D1.Bounded.DriftIndiv
    % Fit.D2.Common.DriftIndiv
    %
    % 2015 YK wrote the initial version.
methods
    function W = DriftIndiv(varargin)
        W = W@Fit.D2.Common.Drift(varargin{:});
    end
    function customize_th_for_Data(W, dim_rel_W)
        W.set_dim_rel_W(dim_rel_W);
        W.remove_params_all;
        W.add_params0;
    end
    function drift_vec = get_drift_vec(W)
        conds = W.get_conds_rel;
        drift_vec = W.get_drift_vec@Fit.D1.Bounded.DriftIndiv(conds);
    end
end
%% Test
methods
    function W = demo
        %%
        W = Fit.D2.Common.DriftIndiv;
        W.Data.load_demo;
        
        W.customize_th_for_Data(1);
        disp(W);
    end
    function W = demo_w_DtbInhDensity
        %%
        W = Fit.D2.Inh.DtbDensity;
        W.Data.load_demo;
        
        W.set_Drifts(rep_deep_copy(Fit.D2.Common.DriftIndiv, [1, 2]));
        disp(W);
    end
end
end