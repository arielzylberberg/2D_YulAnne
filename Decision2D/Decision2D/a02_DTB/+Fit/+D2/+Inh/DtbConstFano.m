classdef DtbConstFano < Fit.D2.Inh.Dtb
    %
    % 2015 YK wrote the initial version.
methods
    function add_params0(W)
        W.add_params0@Fit.D2.Inh.Dtb;        
        for dim = 1:2
            for dim_prioritized = 1:2
                drift_name = sprintf('drift_fac_together_dim%d_%d', ...
                        dim, dim_prioritized);
                sigma_name = sprintf('sigmaSq_fac_together_dim%d_%d', ...
                        dim, dim_prioritized);
                
                W.lb.(drift_name) = W.lb.(sigma_name);
                W.remove_params({sigma_name});                
            end
        end
    end
    function v = get_sigmaSq_fac_together(W)
        % dimK_M : factor of K-th dim when M-th dim is prioritized first.
        %
        % Give sqrt of drift fac, i.e., const coeff var.
        v = sqrt(W.get_drift_fac_together);
    end
end
end