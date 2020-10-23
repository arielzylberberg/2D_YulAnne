classdef DtbConstSigmaSqSharedDriftFac < Fit.D2.Inh.Dtb
    %
    % 2015 YK wrote the initial version.
methods
    function add_params0(W)
        W.add_params0@Fit.D2.Inh.Dtb;        
        
        n_dim = 2;
        for dim = 1:n_dim
            for dim_prioritized = 1:n_dim
%                 dim = n_dim + 1 - dim_prioritized;
                
                fix_name = sprintf('sigmaSq_fac_together_dim%d_%d', ...
                    dim, dim_prioritized);
                W.th0.(fix_name) = 1;
                W.fix_(fix_name);
            end
        end
        
        for dim = 1:n_dim
            for dim_prioritized = 1:n_dim
                W.remove_params({
                    sprintf('drift_fac_together_dim%d_%d', ...
                        dim, dim_prioritized)
                    });
            end
        end
        
        lb_priority = [1, 0]; % Deprioritized can go to zero
        for priority = 1:n_dim
            W.add_params({
                {sprintf('drift_fac_together_priority%d', priority), ...
                    1, lb_priority(priority), 1}
                });
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