classdef DtbConstFanoSharedDriftFac < Fit.D2.Inh.DtbConstFano
    %
    % 2015 YK wrote the initial version.
methods
    function add_params0(W)
        W.add_params0@Fit.D2.Inh.DtbConstFano;
        
        lb_priority = [1, W.lb.drift_fac_together_dim1_2];
        for priority = 1:2
            W.add_params({
                {sprintf('drift_fac_together_priority%d', priority), ...
                    1, lb_priority(priority), 1}
                });
        end
        
        for dim = 1:2
            for dim_prioritized = 1:2
                W.remove_params({
                    sprintf('drift_fac_together_dim%d_%d', ...
                        dim, dim_prioritized)
                    });
            end
        end        
    end
    function v = get_drift_fac_together(W)
        % dimK_M : factor of K-th dim when M-th dim is prioritized first.
        th = W.th;
        v = [
            th.drift_fac_together_priority1, th.drift_fac_together_priority2
            th.drift_fac_together_priority2, th.drift_fac_together_priority1
            ];
    end
end
end